#!/usr/bin/env python3
"""
Jetson-safe installation script for DeepCrack
This script installs packages in a way that avoids illegal instruction errors
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_architecture():
    """Check if we're on the correct architecture"""
    arch = platform.machine()
    print(f"🔍 Architecture: {arch}")
    
    if arch != 'aarch64':
        print(f"⚠️ Warning: Expected aarch64, got {arch}")
        print("   This script is optimized for Jetson Nano (aarch64)")
        return False
    
    print("✅ Architecture check passed")
    return True

def install_system_dependencies():
    """Install system dependencies"""
    print("\n📚 Installing system dependencies...")
    
    commands = [
        ("sudo apt-get update", "Updating package list"),
        ("sudo apt-get install -y python3-dev python3-numpy python3-opencv", "Installing Python development packages"),
        ("sudo apt-get install -y libopenblas-dev liblapack-dev libhdf5-dev", "Installing math libraries"),
        ("sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev", "Installing image libraries"),
        ("sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev", "Installing video libraries"),
        ("sudo apt-get install -y libv4l-dev libxvidcore-dev libx264-dev", "Installing media libraries"),
        ("sudo apt-get install -y libgtk-3-dev libatlas-base-dev gfortran", "Installing GUI and math libraries")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"⚠️ {desc} failed, continuing...")
    
    return True

def create_clean_environment():
    """Create a clean Python environment"""
    print("\n🏗️ Creating clean Python environment...")
    
    # Remove existing venv
    if os.path.exists('venv'):
        print("🗑️ Removing existing virtual environment...")
        run_command("rm -rf venv", "Removing old venv")
    
    # Create new venv
    if not run_command("python3 -m venv venv", "Creating virtual environment"):
        return False
    
    # Activate and upgrade pip
    activate_cmd = "source venv/bin/activate && pip install --upgrade pip"
    if not run_command(activate_cmd, "Activating venv and upgrading pip"):
        return False
    
    return True

def install_python_packages():
    """Install Python packages in the correct order"""
    print("\n🐍 Installing Python packages...")
    
    # Base packages first
    base_packages = [
        "numpy",
        "scipy", 
        "pillow",
        "pyyaml",
        "psutil"
    ]
    
    for package in base_packages:
        cmd = f"source venv/bin/activate && pip install {package}"
        run_command(cmd, f"Installing {package}")
    
    # PyTorch for ARM64 (CPU version)
    print("🔥 Installing PyTorch for ARM64...")
    pytorch_cmd = "source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    if not run_command(pytorch_cmd, "Installing PyTorch"):
        print("⚠️ PyTorch installation failed, trying alternative...")
        run_command("source venv/bin/activate && pip install torch torchvision torchaudio", "Installing PyTorch (alternative)")
    
    # ONNX packages
    onnx_packages = [
        "onnx",
        "onnxruntime"
    ]
    
    for package in onnx_packages:
        cmd = f"source venv/bin/activate && pip install {package}"
        run_command(cmd, f"Installing {package}")
    
    # Try ONNX Runtime GPU
    print("⚡ Trying ONNX Runtime GPU...")
    gpu_cmd = "source venv/bin/activate && pip install onnxruntime-gpu"
    if not run_command(gpu_cmd, "Installing ONNX Runtime GPU"):
        print("⚠️ ONNX Runtime GPU not available, using CPU version")
    
    # OpenCV
    print("📷 Installing OpenCV...")
    opencv_cmd = "source venv/bin/activate && pip install opencv-python"
    if not run_command(opencv_cmd, "Installing OpenCV"):
        print("⚠️ OpenCV pip installation failed, using system OpenCV")
    
    # Model-specific packages
    model_packages = [
        "einops",
        "matplotlib",
        "seaborn"
    ]
    
    for package in model_packages:
        cmd = f"source venv/bin/activate && pip install {package}"
        run_command(cmd, f"Installing {package}")
    
    # Try loralib (may not be available for ARM64)
    print("🧩 Installing loralib...")
    lora_cmd = "source venv/bin/activate && pip install loralib"
    if not run_command(lora_cmd, "Installing loralib"):
        print("⚠️ loralib not available for ARM64, model may need modification")
    
    # Jetson monitoring tools
    print("📊 Installing Jetson monitoring tools...")
    monitor_cmd = "source venv/bin/activate && pip install jetson-stats"
    run_command(monitor_cmd, "Installing jetson-stats")
    
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    test_script = """
import sys
print(f"Python: {sys.version}")
print(f"Architecture: {sys.platform}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
    # Test basic operation
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.dot(a, b)
    print(f"NumPy test: {c}")
except Exception as e:
    print(f"NumPy error: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
    # Test basic operation
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("OpenCV test: Success")
except Exception as e:
    print(f"OpenCV error: {e}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    # Test basic operation
    x = torch.randn(5, 5)
    y = torch.mm(x, x.t())
    print("PyTorch test: Success")
except Exception as e:
    print(f"PyTorch error: {e}")

try:
    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")
except Exception as e:
    print(f"ONNX Runtime error: {e}")

try:
    import onnx
    print(f"ONNX: {onnx.__version__}")
except Exception as e:
    print(f"ONNX error: {e}")
"""
    
    cmd = f"source venv/bin/activate && python3 -c \"{test_script}\""
    return run_command(cmd, "Testing installation")

def main():
    """Main installation function"""
    print("🚀 Jetson-Safe DeepCrack Installation")
    print("=" * 50)
    print("This script installs packages in a way that avoids")
    print("'Illegal Instruction (core dumped)' errors on Jetson Nano.")
    print()
    
    # Check architecture
    if not check_architecture():
        print("⚠️ Architecture check failed, but continuing...")
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create clean environment
    if not create_clean_environment():
        print("❌ Failed to create clean environment")
        return False
    
    # Install Python packages
    install_python_packages()
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("📋 To activate the environment: source venv/bin/activate")
        print("📋 To test: python3 check_python.py")
        return True
    else:
        print("\n⚠️ Installation completed with warnings")
        print("📋 Check the output above for any errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
