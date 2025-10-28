#!/usr/bin/env python3
"""
Verify Jetson Setup and Environment
Checks all dependencies and system configuration
"""

import sys
import subprocess
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}")

def print_result(name, status, details=""):
    symbols = {"pass": "✓", "fail": "✗", "warn": "⚠️"}
    symbol = symbols.get(status, "?")
    print(f"{symbol} {name:<40} {details}")
    return status == "pass"

def check_python():
    """Check Python version"""
    version = sys.version_info
    status = "pass" if version >= (3, 8) else "fail"
    details = f"v{version.major}.{version.minor}.{version.micro}"
    return print_result("Python Version", status, details)

def check_jetson():
    """Check if running on Jetson"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            release = f.read().strip()
        return print_result("Jetson Device", "pass", release.split(',')[0])
    except FileNotFoundError:
        return print_result("Jetson Device", "warn", "Not detected")

def check_import(module_name, package_name=None, attribute=None):
    """Check if a module can be imported"""
    try:
        module = __import__(module_name)
        if attribute:
            version = getattr(module, attribute, "unknown")
        else:
            version = "OK"
        return print_result(package_name or module_name, "pass", str(version))
    except ImportError as e:
        return print_result(package_name or module_name, "fail", str(e))

def check_torch():
    """Check PyTorch installation"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        status = "pass" if cuda_available else "warn"
        details = f"v{version}, CUDA: {cuda_available}"
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            details += f", {device_name}"
        
        return print_result("PyTorch", status, details)
    except ImportError:
        return print_result("PyTorch", "fail", "Not installed")

def check_torchvision():
    """Check TorchVision"""
    try:
        import torchvision
        version = torchvision.__version__
        return print_result("TorchVision", "pass", f"v{version}")
    except ImportError:
        return print_result("TorchVision", "fail", "Not installed")

def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('release ')[-1].split(',')[0]
            return print_result("CUDA Compiler", "pass", f"v{version}")
        else:
            return print_result("CUDA Compiler", "warn", "nvcc not found")
    except FileNotFoundError:
        return print_result("CUDA Compiler", "warn", "nvcc not in PATH")

def check_tensorrt():
    """Check TensorRT"""
    try:
        import tensorrt
        version = tensorrt.__version__
        return print_result("TensorRT", "pass", f"v{version}")
    except ImportError:
        return print_result("TensorRT", "warn", "Not installed (optional)")

def check_onnxruntime():
    """Check ONNX Runtime"""
    try:
        import onnxruntime
        version = onnxruntime.__version__
        providers = onnxruntime.get_available_providers()
        gpu = "CUDA" if "CUDAExecutionProvider" in providers else "CPU"
        return print_result("ONNX Runtime", "pass", f"v{version} ({gpu})")
    except ImportError:
        return print_result("ONNX Runtime", "warn", "Not installed (optional)")

def check_model_file(path="BEST.pth"):
    """Check if model file exists"""
    exists = os.path.exists(path)
    status = "pass" if exists else "fail"
    size = f"{os.path.getsize(path)/1024/1024:.1f} MB" if exists else "Not found"
    return print_result(f"Model File ({path})", status, size)

def check_tegrastats():
    """Check tegrastats availability"""
    try:
        result = subprocess.run(['which', 'tegrastats'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return print_result("tegrastats", "pass", "Available")
        else:
            return print_result("tegrastats", "warn", "Not found")
    except Exception:
        return print_result("tegrastats", "warn", "Not available")

def check_memory():
    """Check system memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024 / 1024 / 1024
        available_gb = mem.available / 1024 / 1024 / 1024
        details = f"{total_gb:.1f} GB total, {available_gb:.1f} GB available"
        status = "pass" if available_gb > 1.0 else "warn"
        return print_result("System Memory", status, details)
    except:
        return print_result("System Memory", "warn", "Cannot check")

def check_disk_space():
    """Check disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / 1024 / 1024 / 1024
        details = f"{free_gb:.1f} GB free"
        status = "pass" if free_gb > 5.0 else "warn"
        return print_result("Disk Space", status, details)
    except:
        return print_result("Disk Space", "warn", "Cannot check")

def main():
    print_header("JETSON SETUP VERIFICATION")
    
    all_pass = True
    
    # System checks
    print("\n📋 SYSTEM INFORMATION")
    all_pass &= check_python()
    all_pass &= check_jetson()
    all_pass &= check_memory()
    all_pass &= check_disk_space()
    
    # CUDA/GPU checks
    print("\n🎮 GPU & CUDA")
    all_pass &= check_cuda()
    all_pass &= check_torch()
    all_pass &= check_torchvision()
    
    # Optional frameworks
    print("\n🔧 OPTIONAL FRAMEWORKS")
    check_tensorrt()  # Don't fail on optional
    check_onnxruntime()
    
    # Python packages
    print("\n📦 PYTHON PACKAGES")
    all_pass &= check_import("scipy", "SciPy", "__version__")
    all_pass &= check_import("cv2", "OpenCV", "__version__")
    all_pass &= check_import("PIL", "Pillow", "__version__")
    all_pass &= check_import("einops", "Einops", "__version__")
    all_pass &= check_import("loralib", "LoRA lib")
    all_pass &= check_import("psutil", "PSUtil", "__version__")
    all_pass &= check_import("numpy", "NumPy", "__version__")
    
    # Monitoring tools
    print("\n📊 MONITORING TOOLS")
    check_tegrastats()  # Don't fail if not on Jetson
    
    # Model files
    print("\n📁 MODEL FILES")
    all_pass &= check_model_file("BEST.pth")
    check_model_file("sample_image.jpg")  # Don't fail on missing test image
    
    # Model architecture
    print("\n🏗️  MODEL ARCHITECTURE")
    try:
        from model_resunet import build_resunet
        print_result("model_resunet.py", "pass", "Found")
    except ImportError as e:
        all_pass &= print_result("model_resunet.py", "fail", str(e))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    if all_pass:
        print("✅ All critical checks passed!")
        print("\n🚀 You're ready to run profiling:")
        print("   python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg")
    else:
        print("❌ Some checks failed. Please review the setup guide:")
        print("   JETSON_ORIN_NANO_COMPLETE_SETUP.md")
        sys.exit(1)
    
    print()

if __name__ == '__main__':
    main()

