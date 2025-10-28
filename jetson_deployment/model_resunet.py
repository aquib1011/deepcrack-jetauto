# model_resunet.py
# Complete model definition matching your notebook (inference-ready).
# - Keeps preprocessing assumptions: RGB, 256x256, divide by 256.0
# - ResUNet returns (segmentation, dar_reg) for compatibility with your export code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# These two are needed for the dynamic kernels used in EEM
from scipy.ndimage import gaussian_filter, laplace

# LoRA for attention layers
import loralib as lora

# Rearrangement helper for attention/patching
from einops import rearrange


# ---------------------------
# MobileViT-style components
# ---------------------------

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Multi-head self-attention with LoRA on Q/K/V projection and output proj.
    """
    def __init__(self, dim, heads=8, dim_head=64, lora_r=2, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        # LoRA-enabled merged QKV
        self.to_qkv = lora.MergedLinear(
            dim, inner_dim * 3, r=lora_r, enable_lora=[True, True, True], bias=False
        )
        self.to_out = (nn.Sequential(
            lora.Linear(inner_dim, dim, r=lora_r, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity())

    def forward(self, x):
        # x: [B, N, C]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., lora_r=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, lora_r, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    """
    Convolutional tokenization + local transformer on non-overlapping patches,
    then fold back to feature map.
    """
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., lora_r=2):
        super().__init__()
        ph, pw = patch_size
        self.ph, self.pw = ph, pw
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, heads=4, dim_head=dim//4,
                                       mlp_dim=mlp_dim, dropout=dropout, lora_r=lora_r)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)  # [B, C=dim, H, W]
        b, c, H, W = x.shape
        ph, pw = self.ph, self.pw
        assert H % ph == 0 and W % pw == 0, "H and W must be divisible by patch size"
        hP, wP = H // ph, W // pw

        # [B, C, H, W] -> [B*hP*wP, ph*pw, C]
        x_p = rearrange(x, 'b c (hp ph) (wp pw) -> (b hp wp) (ph pw) c', ph=ph, pw=pw)
        x_p = self.transformer(x_p)
        # back to [B, C, H, W]
        x = rearrange(x_p, '(b hp wp) (ph pw) c -> b c (hp ph) (wp pw)',
                      b=b, hp=hP, wp=wP, ph=ph, pw=pw)
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        return self.conv4(x)


# --------------------------------------
# Edge-Enhancement & helper building blocks
# --------------------------------------

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize), dtype=np.float32)
    mid = kernelsize // 2
    n[:, :, mid, mid] = kernelvalue
    g = gaussian_filter(n, sigma)
    return torch.from_numpy(g)

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize), dtype=np.float32)
    mid = kernelsize // 2
    n[:, :, mid, mid] = kernelvalue
    l = laplace(n)
    return torch.from_numpy(l)

class SEM(nn.Module):
    """Squeeze-Excite-like channel attention."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                               groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return self.act(self.bn(x))

class DynamicMS_EEM(nn.Module):
    """
    EEM variant with:
      - Depthwise Sobel (mag) instead of DoG
      - Frozen LoG as before
      - Two DSCs (5×5 & 7×7)
      - Summation fusion
    """
    def __init__(self, ch_in, ch_out, kernel=3, groups=1, reduction=16):
        super().__init__()
        self.groups = groups
        self.kernel_size = int(kernel)      # ensure plain int
        self.pad = self.kernel_size // 2    # <- fixed Python int padding

        # prepare depthwise Sobel kernels
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1,1,3,3).repeat(ch_in,1,1,1))
        self.register_buffer("sobel_y", sobel_y.view(1,1,3,3).repeat(ch_in,1,1,1))

        # frozen LoG kernel (build once, keep as buffer)
        lk = laplaceiankernel(ch_in, ch_in//groups, self.kernel_size, 0.9).float()
        # if laplaceiankernel returns a torch.Tensor already, keep it as-is
        if not isinstance(lk, torch.Tensor):
            lk = torch.from_numpy(lk).float()
        self.register_buffer("lk", lk)  # shape [ch_in, ch_in//groups, k, k]

        # original EEM conv1 & conv2
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, 1, groups=groups, bias=False),
            nn.PReLU(ch_out//2),
            nn.InstanceNorm2d(ch_out//2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, 1, groups=groups, bias=False),
            nn.PReLU(ch_out//2),
            nn.InstanceNorm2d(ch_out//2)
        )

        # original EEM conv3: maxpool + 1×1 proj + GN
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(ch_out//2, ch_out, 1, groups=groups, bias=False),
            nn.PReLU(ch_out),
            nn.GroupNorm(4, ch_out)
        )

        # dynamic DSCs
        self.d5 = DepthwiseSeparableConv(ch_out, ch_out, kernel_size=5, padding=2)
        self.d7 = DepthwiseSeparableConv(ch_out, ch_out, kernel_size=7, padding=3)

        # SEMs
        self.sem1 = SEM(ch_out, reduction)
        self.sem2 = SEM(ch_in, reduction)

        self.prelu = nn.PReLU(ch_out)

    def forward(self, x):
        b, c, *_ = x.shape
        # 1) depthwise Sobel
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=c)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=c)
        sobel_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

        # 2) frozen LoG – use fixed int padding
        LoG = F.conv2d(sobel_mag, self.lk.to(x.device), padding=self.pad, groups=self.groups)

        # 3) original EEM branches
        sobel_feat = self.conv1(sobel_mag - x)
        LoG_feat   = self.conv2(LoG)
        edge       = sobel_feat * LoG_feat

        # 4) pool + proj
        tot = self.pool(edge)
        tot = self.proj(tot)

        # 5) dynamic multi-scale DSCs
        m5  = self.d5(tot)
        m7  = self.d7(tot)
        tot = m5 + m7

        # 6) SEM + residual
        tot1 = self.sem1(tot)
        x1   = self.sem2(x)
        return self.prelu(x + x1 + tot + tot1)


# ---------------------------
# UNet-like encoder/decoder
# ---------------------------

class UpsampleConcatBlock(nn.Module):
    """Upsample by 2×, crop to match skip if needed, then concat."""
    def __init__(self):
        super().__init__()
    def forward(self, x, x_skip):
        u = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # crop if off-by-one
        diffY = u.size(2) - x_skip.size(2)
        diffX = u.size(3) - x_skip.size(3)
        if diffY != 0 or diffX != 0:
            u = u[:, :, :u.size(2) - diffY, :u.size(3) - diffX]
        return torch.cat([u, x_skip], dim=1)

class DepthwisePointwiseConv(nn.Module):
    """Depthwise(3×3)+BN+ReLU → Pointwise(1×1)+BN → optional MaxPool(2)."""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.pool(x)


# -------------
# Main network
# -------------

class ResUNet(nn.Module):
    def __init__(self, img_dim=(256, 256), reg_coeff: float = 0.0):
        super().__init__()
        self.reg_coeff = reg_coeff
        f = [16, 32, 64]  # feature sizes

        # Encoder
        self.e1 = DepthwisePointwiseConv(3,  f[0], pool=False)
        self.e1_eem = DynamicMS_EEM(f[0], f[0], kernel=3, groups=1, reduction=16)

        self.e2 = DepthwisePointwiseConv(f[0], f[1], pool=True)
        self.e2_eem = DynamicMS_EEM(f[1], f[1], kernel=3, groups=1, reduction=16)

        self.e3 = DepthwisePointwiseConv(f[1], f[2], pool=True)
        self.e3_eem = DynamicMS_EEM(f[2], f[2], kernel=3, groups=1, reduction=16)

        # Middle blocks (with MobileViT-style blocks)
        self.e4 = DepthwisePointwiseConv(f[2], f[0], pool=True)
        self.mvit1 = MobileViTBlock(dim=f[2], channel=f[0], kernel_size=3, depth=2,
                                    patch_size=(2,2), mlp_dim=128)

        self.e5 = DepthwisePointwiseConv(f[0], f[1], pool=True)
        self.mvit2 = MobileViTBlock(dim=f[2], channel=f[1], kernel_size=3, depth=2,
                                    patch_size=(2,2), mlp_dim=128)

        self.e6 = DepthwisePointwiseConv(f[1], f[2], pool=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f[2], f[2], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(f[2], f[2], kernel_size=3, padding=1),
            nn.PReLU()
        )

        # CA-DAR weights
        self.alpha = nn.Parameter(torch.zeros(f[2]))

        # Decoder (UNet-style)
        self.up0 = UpsampleConcatBlock()
        self.d0 = DepthwisePointwiseConv(f[2]*2, f[2], pool=False)

        self.up1 = UpsampleConcatBlock()
        self.d1 = DepthwisePointwiseConv(f[2]+f[1], f[1], pool=False)

        self.up2 = UpsampleConcatBlock()
        self.d2 = DepthwisePointwiseConv(f[1]+f[0], f[0], pool=False)

        self.up3 = UpsampleConcatBlock()
        self.d3 = DepthwisePointwiseConv(f[0]+f[2], f[2], pool=False)
        self.mvit3 = MobileViTBlock(dim=f[2], channel=f[2], kernel_size=3, depth=2,
                                    patch_size=(2,2), mlp_dim=128)

        self.up4 = UpsampleConcatBlock()
        self.d4 = DepthwisePointwiseConv(f[2]+f[1], f[1], pool=False)

        self.up5 = UpsampleConcatBlock()
        self.d5 = DepthwisePointwiseConv(f[1]+f[0], f[0], pool=False)

        # Segmentation head
        self.output = nn.Sequential(
            nn.Conv2d(f[0], 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1_feat = self.e1_eem(self.e1(x))
        e2_feat = self.e2_eem(self.e2(e1_feat))
        e3_feat = self.e3_eem(self.e3(e2_feat))

        # Middle
        e4_feat = self.mvit1(self.e4(e3_feat))
        e5_feat = self.mvit2(self.e5(e4_feat))
        e6_feat = self.e6(e5_feat)

        # Bottleneck
        b = self.bottleneck(e6_feat)

        # Decoder
        d0 = self.d0(self.up0(b, e6_feat))
        d1 = self.mvit2(self.d1(self.up1(d0, e5_feat)))
        d2 = self.mvit1(self.d2(self.up2(d1, e4_feat)))
        d3 = self.mvit3(self.d3(self.up3(d2, e3_feat)))
        d4 = self.e2_eem(self.d4(self.up4(d3, e2_feat)))
        d5 = self.e1_eem(self.d5(self.up5(d4, e1_feat)))

        out = self.output(d5)

        # CA-DAR regularizer (optional, returns value if reg_coeff>0)
        reg_loss = None
        if self.reg_coeff > 0:
            B, C, H, W = e6_feat.shape
            norms = e6_feat.view(B, C, -1).norm(p=2, dim=2)  # [B,C]
            mean_norm = norms.mean(dim=0)                    # [C]
            weights = torch.softmax(self.alpha, dim=0)       # [C]
            reg_loss = (weights * mean_norm).sum()

        return out, reg_loss


# ---------------------------
# Helper to build the model
# ---------------------------

def build_resunet(img_dim=(256, 256), reg_coeff: float = 0.0, device='cuda'):
    """
    Convenience constructor used by export and inference scripts.
    """
    m = ResUNet(img_dim=img_dim, reg_coeff=reg_coeff)
    return m.to(device)


# ---------------------------
# Quick local test (optional)
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_resunet(device=device)
    x = torch.randn(1, 3, 256, 256, device=device)
    y, reg = net(x)
    print("Output:", y.shape, "Reg:", None if reg is None else float(reg))
