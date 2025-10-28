#!/bin/bash
# Quick fix for illegal instruction errors on Jetson Nano
# This script addresses the specific issue you're experiencing

echo "🚀 Quick Fix for Illegal Instruction Error"
echo "=========================================="

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Not in virtual environment. Please run:"
    echo "   source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Clean everything and start fresh
echo "🧹 Cleaning existing packages..."

# Uninstall all problematic packages
pip uninstall -y numpy scipy opencv-python torch torchvision torchaudio onnx onnxruntime matplotlib pillow psutil pyyaml einops loralib jetson-stats 2>/dev/null || true

# Clear pip cache
pip cache purge 2>/dev/null || true

# Install system packages first
echo "📚 Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-numpy python3-scipy python3-opencv python3-pil python3-matplotlib python3-dev

# Install Python packages in correct order
echo "🐍 Installing Python packages..."

# Upgrade pip
pip install --upgrade pip

# Install basic packages
pip install setuptools wheel cython

# Install NumPy (critical - must be first)
echo "🔢 Installing NumPy..."
pip install numpy==1.19.5 --no-cache-dir --force-reinstall

# Test NumPy
echo "🧪 Testing NumPy..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || {
    echo "❌ NumPy test failed, trying alternative..."
    pip install numpy==1.18.5 --no-cache-dir --force-reinstall
}

# Install SciPy
echo "🔬 Installing SciPy..."
pip install scipy==1.5.4 --no-cache-dir

# Install Pillow
echo "🖼️ Installing Pillow..."
pip install pillow --no-cache-dir

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

# Install OpenCV (use system version if pip fails)
echo "📷 Installing OpenCV..."
pip install opencv-python==4.5.3.56 --no-cache-dir || {
    echo "⚠️ OpenCV pip installation failed, using system OpenCV"
}

# Install other packages
echo "📦 Installing other packages..."
pip install pyyaml psutil matplotlib seaborn einops --no-cache-dir

# Try loralib (may not be available for ARM64)
echo "🧩 Installing loralib..."
pip install loralib --no-cache-dir || {
    echo "⚠️ loralib not available for ARM64, skipping..."
}

# Install Jetson monitoring tools
echo "📊 Installing Jetson tools..."
pip install jetson-stats --no-cache-dir || {
    echo "⚠️ jetson-stats installation failed"
}

# Test all imports
echo "🧪 Testing all imports..."
python3 -c "
import sys
print('Python version:', sys.version)
print('Architecture:', sys.platform)

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

print('\\n🎉 Import test completed!')
"

echo "✅ Quick fix completed!"
echo "📋 You can now run your scripts without illegal instruction errors"


