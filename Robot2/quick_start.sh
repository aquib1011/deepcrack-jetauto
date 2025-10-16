#!/bin/bash
# Quick start script for DeepCrack deployment on Hiwonder JetAuto
# Run this script on your Jetson Nano

echo "🚀 DeepCrack Quick Start for Hiwonder JetAuto"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "deploy_and_test.py" ]; then
    echo "❌ Please run this script from the Robot2 directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxruntime-gpu onnxsim
pip install opencv-python numpy scipy pillow pyyaml matplotlib seaborn
pip install einops loralib jetson-stats psutil

# Make scripts executable
chmod +x *.py

# Run deployment
echo "🚀 Starting deployment..."
python3 deploy_and_test.py --benchmark-duration 30 --monitoring-duration 60

echo "✅ Quick start completed!"
echo "📁 Check the benchmark_results directory for results"
