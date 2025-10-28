#!/bin/bash
# Quick Start Script for Jetson Orin Nano Edge Profiling
# This script automates the complete setup and testing process

set -e  # Exit on error

echo "============================================================"
echo "  Jetson Orin Nano - Quick Start Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    print_warning "Not running on Jetson device - some features may not work"
fi

# Step 1: Check Python version
echo ""
echo "[1/8] Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if (( $(echo "$PYTHON_VERSION >= 3.8" | bc -l) )); then
    print_status "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

# Step 2: Create project directory
echo ""
echo "[2/8] Setting up project directory..."
PROJECT_DIR="$HOME/jetson_crack_detection"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
print_status "Project directory: $PROJECT_DIR"

# Step 3: Create virtual environment
echo ""
echo "[3/8] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_status "Pip upgraded"

# Step 4: Check PyTorch installation
echo ""
echo "[4/8] Checking PyTorch installation..."
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    print_status "PyTorch $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
else
    print_error "PyTorch not installed!"
    echo ""
    echo "Please install PyTorch for Jetson manually:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo ""
    echo "For JetPack 6.0, Python 3.10:"
    echo '  export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl'
    echo '  pip install --no-cache-dir $TORCH_INSTALL'
    exit 1
fi

# Step 5: Install dependencies
echo ""
echo "[5/8] Installing Python dependencies..."

# Check if requirements file exists
if [ -f "requirements_jetson.txt" ]; then
    pip install -r requirements_jetson.txt
    print_status "Dependencies installed from requirements_jetson.txt"
else
    print_warning "requirements_jetson.txt not found, installing core packages..."
    pip install numpy scipy opencv-python Pillow einops loralib psutil pyyaml
    print_status "Core dependencies installed"
fi

# Step 6: Verify model files
echo ""
echo "[6/8] Checking model files..."

# Check for model architecture
if [ -f "model_resunet.py" ]; then
    print_status "Model architecture found: model_resunet.py"
else
    print_error "model_resunet.py not found in $PROJECT_DIR"
    echo "Please copy model_resunet.py to this directory"
    exit 1
fi

# Check for model weights
if [ -f "BEST.pth" ]; then
    SIZE=$(du -h BEST.pth | cut -f1)
    print_status "Model weights found: BEST.pth ($SIZE)"
else
    print_error "BEST.pth not found in $PROJECT_DIR"
    echo "Please copy BEST.pth to this directory"
    exit 1
fi

# Check for test image
if [ -f "sample_image.jpg" ]; then
    print_status "Test image found: sample_image.jpg"
else
    print_warning "sample_image.jpg not found (optional)"
fi

# Step 7: Run verification
echo ""
echo "[7/8] Running setup verification..."
if [ -f "verify_setup.py" ]; then
    python3 verify_setup.py
else
    print_warning "verify_setup.py not found, skipping detailed verification"
fi

# Step 8: Set performance mode
echo ""
echo "[8/8] Configuring performance settings..."
if [ -f /etc/nv_tegra_release ]; then
    # Check if we can run sudo without password (or cache is valid)
    if sudo -n true 2>/dev/null; then
        print_status "Setting maximum performance mode..."
        sudo nvpmodel -m 0 2>/dev/null || print_warning "Could not set power mode (may need sudo)"
        sudo jetson_clocks 2>/dev/null || print_warning "Could not enable jetson_clocks (may need sudo)"
        print_status "Performance mode enabled"
    else
        print_warning "Sudo access needed for performance mode"
        echo "  Run manually: sudo nvpmodel -m 0 && sudo jetson_clocks"
    fi
else
    print_warning "Not on Jetson device, skipping performance configuration"
fi

# Summary
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "📍 Project directory: $PROJECT_DIR"
echo ""
echo "🚀 Run profiling with:"
echo "   source venv/bin/activate"
echo "   python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg"
echo ""
echo "📊 Available scripts:"
echo "   • verify_setup.py - Verify environment"
echo "   • profile_edge_device.py - Complete profiling"
echo "   • visualize_results.py - Visualize profiling results"
echo ""
echo "📖 Documentation:"
echo "   • JETSON_ORIN_NANO_COMPLETE_SETUP.md"
echo ""

