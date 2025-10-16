#!/usr/bin/env python3
"""
Enhanced PTH to ONNX conversion script for DeepCrack ResUNet model
Optimized for Hiwonder JetAuto (Jetson Nano) deployment
"""

import torch
import onnx
import numpy as np
import os
import sys
from pathlib import Path

# Add the model directory to path
sys.path.append(str(Path(__file__).parent.parent / "Robot1"))
from model_resunet import build_resunet

def load_checkpoint(model, path):
    """Load PyTorch checkpoint with error handling"""
    print(f"📂 Loading checkpoint from: {path}")
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("✅ Loaded model_state_dict from checkpoint")
        else:
            model.load_state_dict(ckpt)
            print("✅ Loaded direct state dict from checkpoint")
        return model
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise

def verify_model_outputs(pytorch_model, onnx_model_path, input_shape=(1, 3, 256, 256)):
    """Verify ONNX model outputs match PyTorch model"""
    print("🔍 Verifying ONNX model outputs...")
    
    # Create test input
    test_input = torch.randn(input_shape)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output, _ = pytorch_model(test_input)
        pytorch_output = pytorch_output.numpy()
    
    # ONNX inference
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    onnx_output = session.run([output_name], {input_name: test_input.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"📊 Maximum difference: {diff:.6f}")
    
    if diff < 1e-5:
        print("✅ ONNX model verification passed!")
        return True
    else:
        print("⚠️ ONNX model verification failed - outputs don't match closely")
        return False

def convert_pth_to_onnx(pth_path, onnx_path, img_size=(256, 256), opset_version=11):
    """
    Convert PyTorch model to ONNX with optimizations for Jetson Nano
    """
    print("🚀 Starting PTH to ONNX conversion...")
    print(f"📁 Input: {pth_path}")
    print(f"📁 Output: {onnx_path}")
    print(f"🖼️ Image size: {img_size}")
    print(f"🔧 ONNX opset version: {opset_version}")
    
    # Check if input file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"PyTorch model not found: {pth_path}")
    
    # Set device (use CPU for conversion to ensure compatibility)
    device = 'cpu'  # Always use CPU for ONNX conversion
    print(f"💻 Using device: {device}")
    
    # Build model
    print("🏗️ Building ResUNet model...")
    model = build_resunet(img_dim=img_size, reg_coeff=0.0, device=device)
    
    # Load checkpoint
    model = load_checkpoint(model, pth_path)
    model.eval()
    
    # Create wrapper for ONNX export (only inference, no regularization)
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            output, _ = self.model(x)  # Discard regularization loss
            return output
    
    # Wrap model for inference-only export
    wrapped_model = InferenceWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
    print(f"📐 Dummy input shape: {dummy_input.shape}")
    
    # Export to ONNX
    print("📤 Exporting to ONNX...")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False
        )
        print("✅ ONNX export completed successfully!")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise
    
    # Verify ONNX model
    print("🔍 Verifying ONNX model...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed!")
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        raise
    
    # Simplify ONNX model (optional but recommended)
    print("🔧 Simplifying ONNX model...")
    try:
        from onnxsim import simplify
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, onnx_path)
            print("✅ ONNX model simplified successfully!")
        else:
            print("⚠️ ONNX model simplification failed, using original")
    except ImportError:
        print("⚠️ onnxsim not available, skipping simplification")
    except Exception as e:
        print(f"⚠️ ONNX simplification failed: {e}")
    
    # Verify outputs match
    verify_model_outputs(model, onnx_path, (1, 3, img_size[0], img_size[1]))
    
    # Print model info
    print("\n📊 Model Information:")
    print(f"   Input shape: (batch_size, 3, {img_size[0]}, {img_size[1]})")
    print(f"   Output shape: (batch_size, 1, {img_size[0]}, {img_size[1]})")
    print(f"   Model size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
    
    return onnx_path

def main():
    """Main conversion function"""
    # Paths
    pth_path = "../onnx/BEST.pth"
    onnx_path = "../onnx/BEST_optimized.onnx"
    
    # Image size (should match your training)
    img_size = (256, 256)
    
    # ONNX opset version (11 is well supported on Jetson)
    opset_version = 11
    
    print("🧠 DeepCrack ResUNet - PTH to ONNX Converter")
    print("=" * 50)
    
    try:
        # Convert model
        output_path = convert_pth_to_onnx(
            pth_path=pth_path,
            onnx_path=onnx_path,
            img_size=img_size,
            opset_version=opset_version
        )
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Output file: {output_path}")
        print(f"📏 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # Test inference
        print("\n🧪 Testing ONNX inference...")
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Test with random input
        test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        output = session.run(None, {"input": test_input})[0]
        
        print(f"✅ Test inference successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
