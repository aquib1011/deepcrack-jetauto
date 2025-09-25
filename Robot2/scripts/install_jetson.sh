#!/usr/bin/env bash
set -e

echo "🚀 Setting up Jetson Nano for ONNX inference..."

# Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# Upgrade pip
python3 -m pip install --upgrade pip

# Install ONNX Runtime for Jetson (TensorRT support)
echo "📦 Installing ONNX Runtime with TensorRT support..."
pip3 uninstall -y onnxruntime onnxruntime-gpu || true

# Try Jetson-specific ONNX Runtime first
pip3 install onnxruntime-gpu==1.16.3 || {
    echo "⚠️  onnxruntime-gpu failed, falling back to CPU version"
    pip3 install onnxruntime
}

# Install other dependencies
echo "📦 Installing additional dependencies..."
pip3 install opencv-python numpy PyYAML psutil

# Verify installation
echo "🔍 Verifying installation..."
python3 - <<'PY'
import onnxruntime as ort
import cv2
import numpy as np
import yaml
print("✅ ONNX Runtime providers:", ort.get_available_providers())
print("✅ OpenCV version:", cv2.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ PyYAML version:", yaml.__version__)
PY

echo "✅ Setup complete! Ready to run inference on Jetson Nano."
