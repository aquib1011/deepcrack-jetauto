# Complete Jetson Orin Nano Developer Kit Setup Guide

## Table of Contents
1. [Initial Hardware Setup](#initial-hardware-setup)
2. [JetPack Installation](#jetpack-installation)
3. [System Configuration](#system-configuration)
4. [Python Environment Setup](#python-environment-setup)
5. [PyTorch & Dependencies Installation](#pytorch-dependencies-installation)
6. [Model Deployment](#model-deployment)
7. [Testing Framework](#testing-framework)

---

## 1. Initial Hardware Setup

### What You'll Need
- Jetson Orin Nano Developer Kit
- USB-C power supply (15W or 25W/30W depending on power mode)
- microSD card (64GB or larger, UHS-1 recommended)
- USB keyboard and mouse
- HDMI monitor
- Ethernet cable (recommended) or WiFi
- Host computer for initial setup

### Power Supply Recommendations
- **7W/15W modes**: USB-C power supply (5V/3A)
- **25W/30W modes**: DC barrel jack power supply (9-20V/3A minimum)

### First Boot
1. **Flash JetPack to microSD card** (see next section)
2. Insert microSD card into Jetson Orin Nano
3. Connect monitor, keyboard, mouse
4. Connect power supply
5. Device will boot automatically

---

## 2. JetPack Installation

### Option A: Using NVIDIA SDK Manager (Recommended for New Device)

**On your host computer (Ubuntu 18.04/20.04/22.04):**

```bash
# Download SDK Manager from NVIDIA website
# https://developer.nvidia.com/sdk-manager

# Install SDK Manager
sudo apt install ./sdkmanager_[version].deb

# Launch SDK Manager
sdkmanager
```

**Steps in SDK Manager:**
1. Select target hardware: Jetson Orin Nano Developer Kit
2. Select JetPack version: **JetPack 6.0** (or latest stable)
3. Select components:
   - Jetson OS
   - CUDA Toolkit
   - cuDNN
   - TensorRT
   - OpenCV
   - VPI (Vision Programming Interface)
4. Flash and install

### Option B: Using SD Card Image

```bash
# Download JetPack SD Card Image
# https://developer.nvidia.com/embedded/jetpack

# Flash using Balena Etcher or dd command
sudo dd if=jetson-orin-nano-sd-card-image.img of=/dev/sdX bs=4M status=progress
```

### Verify Installation After First Boot

```bash
# Check JetPack version
sudo apt-cache show nvidia-jetpack

# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi
# Note: nvidia-smi might not work on Jetson, use tegrastats instead

# Check system info
cat /etc/nv_tegra_release
```

---

## 3. System Configuration

### Update System

```bash
# Update package lists
sudo apt update

# Upgrade packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    nano \
    vim \
    htop \
    python3-pip \
    python3-dev \
    libhdf5-dev \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    gfortran
```

### Configure Power Mode

```bash
# Check available power modes
sudo nvpmodel -q

# Set to maximum performance (15W for Orin Nano)
sudo nvpmodel -m 0

# For 7W low power mode
# sudo nvpmodel -m 1

# Set CPU governor to performance
sudo jetson_clocks

# Verify current mode
sudo nvpmodel -q verbose
```

### Enable Fan Control (if applicable)

```bash
# Check fan status
cat /sys/devices/pwm-fan/target_pwm

# Set fan to maximum (255 = 100%)
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

# Create automatic fan control script
sudo nano /etc/rc.local

# Add before 'exit 0':
# echo 150 > /sys/devices/pwm-fan/target_pwm
```

### Increase Swap Space (Recommended for compilation)

```bash
# Disable current swap
sudo swapoff -a

# Remove old swap file
sudo rm /swapfile

# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

---

## 4. Python Environment Setup

### Install Python 3.10+ (if needed)

```bash
# Check Python version
python3 --version

# If Python < 3.10, add deadsnakes PPA
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.10
sudo apt install -y python3.10 python3.10-dev python3.10-venv

# Update alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### Create Virtual Environment

```bash
# Install venv
sudo apt install -y python3.10-venv

# Create project directory
mkdir -p ~/jetson_projects/crack_detection
cd ~/jetson_projects/crack_detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## 5. PyTorch & Dependencies Installation

### Install PyTorch for Jetson

```bash
# Activate virtual environment
source ~/jetson_projects/crack_detection/venv/bin/activate

# Install dependencies
sudo apt install -y \
    libopenblas-dev \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Install numpy first
pip install numpy==1.24.3

# Download and install PyTorch wheel for Jetson
# Check your JetPack version and architecture
# For JetPack 6.0, Python 3.10, aarch64:

export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

pip install --no-cache-dir $TORCH_INSTALL

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install TorchVision (matching PyTorch version)

```bash
# Install dependencies
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Clone torchvision
cd ~/jetson_projects
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision

# Build from source
cd torchvision
export BUILD_VERSION=0.18.0
python3 setup.py install --user

# Verify
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

### Install Model Dependencies

```bash
# Activate environment
source ~/jetson_projects/crack_detection/venv/bin/activate

# Install required packages
pip install \
    scipy==1.10.1 \
    opencv-python==4.8.0.74 \
    Pillow==10.0.0 \
    einops==0.6.1 \
    pyyaml==6.0 \
    tqdm==4.66.1

# Install loralib for LoRA layers
pip install loralib==0.1.2

# Install ONNX Runtime (for ONNX inference)
pip install onnxruntime-gpu==1.16.3

# Install monitoring tools
pip install \
    psutil==5.9.5 \
    pandas==2.0.3 \
    matplotlib==3.7.2
```

### Verify All Dependencies

```bash
python3 << EOF
import torch
import torchvision
import scipy
import cv2
import loralib
import einops
import onnxruntime
import psutil

print("✓ PyTorch:", torch.__version__)
print("✓ TorchVision:", torchvision.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ SciPy:", scipy.__version__)
print("✓ OpenCV:", cv2.__version__)
print("✓ LoRA lib: OK")
print("✓ Einops: OK")
print("✓ ONNX Runtime:", onnxruntime.__version__)
print("✓ PSUtil:", psutil.__version__)
print("\nAll dependencies installed successfully!")
EOF
```

---

## 6. Model Deployment

### Transfer Files to Jetson

**Option A: Using SCP from host computer**

```bash
# From your host computer
scp -r /path/to/Robot/Robot1/model_resunet.py jetson@jetson-ip:~/jetson_projects/crack_detection/
scp -r /path/to/Robot/onnx/BEST.pth jetson@jetson-ip:~/jetson_projects/crack_detection/
scp -r /path/to/Robot/sample_image.jpg jetson@jetson-ip:~/jetson_projects/crack_detection/
```

**Option B: Using USB drive**

```bash
# On Jetson
cd ~/jetson_projects/crack_detection
cp /media/jetson/USB_DRIVE/model_resunet.py .
cp /media/jetson/USB_DRIVE/BEST.pth .
cp /media/jetson/USB_DRIVE/sample_image.jpg .
```

**Option C: Using Git (if in repository)**

```bash
cd ~/jetson_projects/crack_detection
git clone <your-repo-url> .
```

---

## 7. Testing Framework

The testing scripts will be created in the next step to measure:
- **Memory Usage**: RAM and GPU memory
- **Inference Time**: Average, min, max, std
- **Power Consumption**: Using tegrastats

See the testing scripts created alongside this guide.

---

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA installation
ls /usr/local/cuda/bin/nvcc

# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

### Memory Issues

```bash
# Clear cache
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Monitor memory
watch -n 1 free -h
```

### Permission Issues

```bash
# Add user to necessary groups
sudo usermod -aG video $USER
sudo usermod -aG i2c $USER

# Logout and login again
```

---

## Next Steps

1. Run the setup verification script: `python3 verify_setup.py`
2. Run baseline benchmarks: `python3 benchmark_system.py`
3. Test model inference: `python3 test_inference.py`
4. Run comprehensive profiling: `python3 profile_edge_device.py`

---

## Additional Resources

- [NVIDIA Jetson Orin Nano Documentation](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide)
- [JetPack SDK Documentation](https://docs.nvidia.com/jetson/jetpack/index.html)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Zoo (Pre-built packages)](https://elinux.org/Jetson_Zoo)

