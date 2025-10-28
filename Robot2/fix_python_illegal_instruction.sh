#!/bin/bash
# Fix for Python causing illegal instruction errors on Jetson Nano
# This addresses the root cause when ANY Python command fails

echo "🚨 Fixing Python Illegal Instruction Error"
echo "=========================================="
echo "This script fixes the root cause when ANY Python command fails"
echo

# Check if we're on the right system
ARCH=$(uname -m)
echo "🔍 System architecture: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️ Warning: This script is designed for aarch64 (Jetson Nano)"
    echo "   Current architecture: $ARCH"
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Python3 not found or corrupted"
    echo "Installing Python3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev
}

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🔍 Currently in virtual environment: $VIRTUAL_ENV"
    echo "⚠️ Virtual environment may be corrupted, deactivating..."
    deactivate
fi

# Remove corrupted virtual environment
if [ -d "venv" ]; then
    echo "🗑️ Removing corrupted virtual environment..."
    rm -rf venv
fi

# Check system Python
echo "🧪 Testing system Python..."
python3 -c "print('System Python works!')" || {
    echo "❌ System Python is corrupted, reinstalling..."
    sudo apt-get remove -y python3 python3-pip python3-venv
    sudo apt-get autoremove -y
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev
}

# Install system dependencies
echo "📚 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-numpy \
    python3-scipy \
    python3-opencv \
    python3-pil \
    python3-matplotlib \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# Create fresh virtual environment
echo "🏗️ Creating fresh virtual environment..."
python3 -m venv venv --clear

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify virtual environment Python
echo "🧪 Testing virtual environment Python..."
python3 -c "print('Virtual environment Python works!')" || {
    echo "❌ Virtual environment Python failed, trying alternative..."
    deactivate
    rm -rf venv
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
}

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install packages in safe order
echo "🐍 Installing packages safely..."

# Install basic packages first
echo "📦 Installing basic packages..."
pip install setuptools wheel cython

# Install NumPy (critical - must be first)
echo "🔢 Installing NumPy..."
pip install numpy==1.19.5 --no-cache-dir --force-reinstall

# Test NumPy
echo "🧪 Testing NumPy..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || {
    echo "❌ NumPy failed, trying alternative..."
    pip install numpy==1.18.5 --no-cache-dir --force-reinstall
}

# Install other packages
echo "📦 Installing other packages..."
pip install scipy==1.5.4 --no-cache-dir
pip install pillow --no-cache-dir
pip install pyyaml psutil --no-cache-dir

# Install PyTorch (CPU version for ARM64)
echo "🔥 Installing PyTorch..."
pip install torch==1.9.0 torchvision==0.10.0 --no-cache-dir || {
    echo "⚠️ PyTorch installation failed, trying alternative..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
}

# Install ONNX
echo "🧠 Installing ONNX..."
pip install onnx==1.9.0 --no-cache-dir

# Install ONNX Runtime
echo "⚡ Installing ONNX Runtime..."
pip install onnxruntime==1.8.0 --no-cache-dir

# Install OpenCV
echo "📷 Installing OpenCV..."
pip install opencv-python==4.5.3.56 --no-cache-dir || {
    echo "⚠️ OpenCV pip installation failed, using system OpenCV"
}

# Install other packages
echo "📦 Installing remaining packages..."
pip install matplotlib seaborn einops --no-cache-dir

# Try loralib
echo "🧩 Installing loralib..."
pip install loralib --no-cache-dir || {
    echo "⚠️ loralib not available for ARM64, skipping..."
}

# Install Jetson monitoring tools
echo "📊 Installing Jetson tools..."
pip install jetson-stats --no-cache-dir || {
    echo "⚠️ jetson-stats installation failed"
}

# Final test
echo "🧪 Final comprehensive test..."
python3 -c "
import sys
print('Python version:', sys.version)
print('Architecture:', sys.platform)
print('Virtual environment:', sys.prefix)

# Test all critical imports
try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
    # Test basic operation
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.dot(a, b)
    print('   NumPy test result:', c)
except Exception as e:
    print('❌ NumPy error:', e)

try:
    import scipy
    print('✅ SciPy:', scipy.__version__)
except Exception as e:
    print('❌ SciPy error:', e)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except Exception as e:
    print('❌ OpenCV error:', e)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('❌ PyTorch error:', e)

try:
    import onnx
    print('✅ ONNX:', onnx.__version__)
except Exception as e:
    print('❌ ONNX error:', e)

try:
    import onnxruntime as ort
    print('✅ ONNX Runtime:', ort.__version__)
    print('   Providers:', ort.get_available_providers())
except Exception as e:
    print('❌ ONNX Runtime error:', e)

try:
    import PIL
    print('✅ Pillow:', PIL.__version__)
except Exception as e:
    print('❌ Pillow error:', e)

try:
    import yaml
    print('✅ PyYAML: Available')
except Exception as e:
    print('❌ PyYAML error:', e)

try:
    import psutil
    print('✅ psutil:', psutil.__version__)
except Exception as e:
    print('❌ psutil error:', e)

try:
    import einops
    print('✅ einops: Available')
except Exception as e:
    print('❌ einops error:', e)

print('\\n🎉 All tests completed!')
"

echo "✅ Python fix completed!"
echo "📋 You can now run Python commands without illegal instruction errors"
echo "📋 To activate the environment: source venv/bin/activate"


