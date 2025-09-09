# export_onnx.py
import torch, onnx
from onnxsim import simplify
from model_resunet import build_resunet

CKPT_PATH = "deepcrack.pth"
ONNX_PATH = "deepcrack_256.onnx"
IMG_H, IMG_W = 256, 256

class InferOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        y, _ = self.model(x)   # your ResUNet returns (out, reg_loss)
        return y

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_resunet(img_dim=(IMG_H, IMG_W), reg_coeff=0.0, device=device)
    model = load_checkpoint(model, CKPT_PATH)
    model.eval()

    wrapped = InferOnly(model).to(device)
    dummy = torch.randn(1, 3, IMG_H, IMG_W, device=device)

    torch.onnx.export(
        wrapped, dummy, ONNX_PATH,
        input_names=["input"], output_names=["mask"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}}
    )

    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)
    m_simp, ok = simplify(m)
    assert ok, "ONNX simplify failed"
    onnx.save(m_simp, ONNX_PATH)
    print("Exported:", ONNX_PATH)
