#!/bin/bash
# Complete environment setup for Hiwonder JetAuto (Jetson Nano)
# Run this script on your Jetson Nano

echo "🚀 Setting up Hiwonder JetAuto environment for DeepCrack inference"
echo "================================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential Python packages
echo "🐍 Installing Python packages..."
sudo apt install -y python3-pip python3-dev python3-venv python3-setuptools
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev libatlas-base-dev gfortran
sudo apt install -y libhdf5-serial-dev hdf5-tools
sudo apt install -y libopencv-dev python3-opencv

# Install PyTorch for Jetson (CPU version for conversion)
echo "🔥 Installing PyTorch for Jetson..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ONNX and related packages
echo "🧠 Installing ONNX packages..."
pip3 install onnx onnxruntime onnxsim
pip3 install onnxruntime-gpu  # For GPU acceleration on Jetson

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip3 install opencv-python numpy scipy pillow
pip3 install pyyaml matplotlib seaborn
pip3 install einops loralib  # For your model architecture

# Install monitoring tools
echo "📊 Installing monitoring tools..."
pip3 install jetson-stats  # For jtop monitoring
pip3 install psutil  # For system monitoring

# Install TensorRT (if not already installed)
echo "⚡ Checking TensorRT installation..."
if ! command -v trtexec &> /dev/null; then
    echo "Installing TensorRT..."
    # TensorRT is usually pre-installed with JetPack
    # If not, install from NVIDIA's repository
    sudo apt install -y tensorrt
fi

# Create virtual environment for the project
echo "🏗️ Creating virtual environment..."
cd ~/deepcrack-jetauto/Robot2
python3 -m venv venv
source venv/bin/activate

# Install packages in virtual environment
echo "📦 Installing packages in virtual environment..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxruntime-gpu onnxsim
pip install opencv-python numpy scipy pillow pyyaml matplotlib seaborn
pip install einops loralib jetson-stats psutil

echo "✅ Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To monitor system performance, run: jtop"
