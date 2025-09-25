#!/usr/bin/env bash
set -e

echo "🐍 Updating Jetson Nano from Python 2.7 to Python 3..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3 and development tools
echo "🐍 Installing Python 3 and development tools..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python3-distutils

# Install Python 3.8 if available (better for Jetson)
echo "🔍 Checking for Python 3.8..."
if apt-cache show python3.8 >/dev/null 2>&1; then
    echo "📦 Installing Python 3.8..."
    sudo apt-get install -y python3.8 python3.8-pip python3.8-dev python3.8-venv
    PYTHON3_CMD="python3.8"
    PIP3_CMD="pip3.8"
else
    echo "📦 Using default Python 3..."
    PYTHON3_CMD="python3"
    PIP3_CMD="pip3"
fi

# Verify Python 3 installation
echo "✅ Verifying Python 3 installation..."
$PYTHON3_CMD --version
$PIP3_CMD --version

# Upgrade pip
echo "⬆️ Upgrading pip..."
$PYTHON3_CMD -m pip install --upgrade pip

# Set Python 3 as default (optional - creates symlinks)
echo "🔗 Setting up Python 3 aliases..."
if ! command -v python &> /dev/null; then
    echo "Creating python -> python3 symlink..."
    sudo ln -sf $(which $PYTHON3_CMD) /usr/bin/python
fi

# Install ONNX Runtime for Jetson (TensorRT support)
echo "📦 Installing ONNX Runtime with TensorRT support..."
$PIP3_CMD uninstall -y onnxruntime onnxruntime-gpu || true

# Try Jetson-specific ONNX Runtime first
$PIP3_CMD install onnxruntime-gpu==1.16.3 || {
    echo "⚠️  onnxruntime-gpu failed, falling back to CPU version"
    $PIP3_CMD install onnxruntime
}

# Install other dependencies
echo "📦 Installing additional dependencies..."
$PIP3_CMD install opencv-python numpy PyYAML psutil

# Verify installation
echo "🔍 Verifying installation..."
$PYTHON3_CMD - <<'PY'
import onnxruntime as ort
import cv2
import numpy as np
import yaml
import sys

print("✅ Python version:", sys.version)
print("✅ ONNX Runtime providers:", ort.get_available_providers())
print("✅ OpenCV version:", cv2.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ PyYAML version:", yaml.__version__)

# Test TensorRT availability
if 'TensorrtExecutionProvider' in ort.get_available_providers():
    print("🚀 TensorRT is available!")
else:
    print("⚠️  TensorRT not available - check JetPack installation")

# Test CUDA availability
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("🚀 CUDA is available!")
else:
    print("⚠️  CUDA not available - check CUDA installation")
PY

# Create a simple test script
echo "🧪 Creating test script..."
cat > test_python3.py << 'EOF'
#!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime imported successfully")
    print(f"Available providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"❌ ONNX Runtime import failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
EOF

chmod +x test_python3.py

echo ""
echo "🎉 Python 3 setup complete!"
echo ""
echo "📋 Summary:"
echo "   Python 3 command: $PYTHON3_CMD"
echo "   Pip 3 command: $PIP3_CMD"
echo ""
echo "🧪 Test your installation:"
echo "   $PYTHON3_CMD test_python3.py"
echo ""
echo "🚀 Ready to run inference with:"
echo "   $PYTHON3_CMD run_image_py3.py --input ../sample_image.jpg --output result.png"
echo ""
