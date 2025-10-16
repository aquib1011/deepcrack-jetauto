#!/bin/bash
# Fix for "Illegal Instruction (core dumped)" error on Jetson Nano
# This script addresses common ARM64 architecture issues

echo "🔧 Fixing Illegal Instruction Error on Jetson Nano"
echo "=================================================="

# Check if we're on ARM64
echo "🔍 Checking system architecture..."
ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️ Warning: This script is designed for aarch64 (Jetson Nano)"
    echo "   Current architecture: $ARCH"
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "🗑️ Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "🏗️ Creating fresh virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install system dependencies first
echo "🔧 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-opencv \
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

# Install PyTorch for ARM64 (CPU version)
echo "🔥 Installing PyTorch for ARM64..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ONNX Runtime for ARM64
echo "🧠 Installing ONNX Runtime for ARM64..."
pip install onnxruntime

# Try to install ONNX Runtime GPU for Jetson
echo "⚡ Installing ONNX Runtime GPU for Jetson..."
pip install onnxruntime-gpu || {
    echo "⚠️ ONNX Runtime GPU installation failed, using CPU version"
}

# Install other dependencies
echo "📚 Installing additional dependencies..."
pip install \
    onnx \
    onnxsim \
    opencv-python \
    numpy \
    scipy \
    pillow \
    pyyaml \
    matplotlib \
    seaborn \
    psutil

# Install model-specific dependencies
echo "🧩 Installing model dependencies..."
pip install \
    einops \
    loralib || {
    echo "⚠️ loralib installation failed, trying alternative..."
    pip install loralib --no-deps || {
        echo "⚠️ loralib not available, model may not work properly"
    }
}

# Install Jetson-specific tools
echo "📊 Installing Jetson monitoring tools..."
pip install jetson-stats || {
    echo "⚠️ jetson-stats installation failed"
}

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Architecture: {sys.platform}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import error: {e}')

try:
    import onnxruntime as ort
    print(f'ONNX Runtime: {ort.__version__}')
    print(f'Available providers: {ort.get_available_providers()}')
except ImportError as e:
    print(f'ONNX Runtime import error: {e}')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV import error: {e}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy import error: {e}')
"

echo "🎉 Installation completed!"
echo "📋 To activate the environment: source venv/bin/activate"
echo "📋 To test: python3 check_python.py"
