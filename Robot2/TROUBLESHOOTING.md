# 🔧 Troubleshooting Guide: Illegal Instruction Error

This guide helps you fix the "Illegal Instruction (core dumped)" error on your Hiwonder JetAuto (Jetson Nano).

## 🚨 Quick Fix (Try This First)

```bash
# On your Jetson Nano
cd ~/deepcrack-jetauto/Robot2

# Run the diagnostic script
python3 diagnose_illegal_instruction.py

# Run the fix script
chmod +x fix_illegal_instruction.sh
./fix_illegal_instruction.sh

# Or use the Python installer
python3 install_jetson_safe.py
```

## 🔍 What Causes "Illegal Instruction" Errors?

The "Illegal Instruction (core dumped)" error on ARM64 systems like the Jetson Nano typically occurs due to:

1. **Architecture Mismatch**: Packages compiled for x86_64 instead of aarch64
2. **Missing CPU Features**: Code using instructions not supported by your CPU
3. **Incompatible Libraries**: System libraries not compatible with your architecture
4. **Corrupted Installation**: Incomplete or corrupted package installations

## 🛠️ Step-by-Step Solutions

### Solution 1: Clean Installation

```bash
# Remove existing virtual environment
rm -rf venv

# Create fresh environment
python3 -m venv venv
source venv/bin/activate

# Install packages in correct order
pip install --upgrade pip
pip install numpy scipy pillow pyyaml psutil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime
pip install opencv-python
pip install einops matplotlib seaborn
```

### Solution 2: Use System Packages

```bash
# Install system packages first
sudo apt-get update
sudo apt-get install -y python3-numpy python3-opencv python3-scipy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install only essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime
pip install einops loralib
```

### Solution 3: Check CPU Features

```bash
# Check your CPU features
cat /proc/cpuinfo | grep -i flags

# Look for these features:
# - asimd (NEON support)
# - fp (FPU support)
# - aes (AES support)
```

### Solution 4: Use Pre-compiled Jetson Packages

```bash
# Install Jetson-specific packages
sudo apt-get install -y python3-pytorch python3-torchvision
sudo apt-get install -y python3-opencv python3-numpy python3-scipy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install remaining packages
pip install onnx onnxruntime
pip install einops loralib
```

## 🔍 Diagnostic Commands

### Check Architecture
```bash
uname -m
# Should output: aarch64
```

### Check Python Version
```bash
python3 --version
# Should be Python 3.6+ (recommended: 3.8+)
```

### Check Installed Packages
```bash
pip list | grep -E "(torch|onnx|numpy|opencv)"
```

### Check System Libraries
```bash
ldconfig -p | grep -E "(openblas|lapack|hdf5|jpeg|png|tiff)"
```

### Test Basic Operations
```bash
python3 -c "
import numpy as np
print('NumPy test:', np.dot([1,2,3], [4,5,6]))
"
```

## 🚨 Common Error Patterns

### Error: "Illegal instruction" when importing numpy
**Cause**: NumPy compiled for wrong architecture
**Fix**: 
```bash
pip uninstall numpy
pip install numpy --no-cache-dir
```

### Error: "Illegal instruction" when importing torch
**Cause**: PyTorch compiled for wrong architecture
**Fix**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error: "Illegal instruction" when importing cv2
**Cause**: OpenCV compiled for wrong architecture
**Fix**:
```bash
pip uninstall opencv-python
sudo apt-get install python3-opencv
```

### Error: "Illegal instruction" when importing onnxruntime
**Cause**: ONNX Runtime compiled for wrong architecture
**Fix**:
```bash
pip uninstall onnxruntime
pip install onnxruntime --no-cache-dir
```

## 🔧 Advanced Troubleshooting

### Check Library Dependencies
```bash
# Check what libraries a package is linked against
ldd venv/lib/python3.8/site-packages/numpy/core/_multiarray_umath.cpython-38-aarch64-linux-gnu.so
```

### Check for Mixed Architectures
```bash
# Check if you have mixed x86_64 and aarch64 packages
find venv -name "*.so" -exec file {} \; | grep -v aarch64
```

### Check CPU Capabilities
```bash
# Check if your CPU supports required instructions
python3 -c "
import numpy as np
print('NumPy CPU features:', np.show_config())
"
```

### Check Memory Issues
```bash
# Check available memory
free -h
# Check swap usage
swapon -s
```

## 🎯 Jetson-Specific Solutions

### Use JetPack Packages
```bash
# Install JetPack packages
sudo apt-get install -y python3-pytorch python3-torchvision
sudo apt-get install -y python3-opencv python3-numpy python3-scipy
```

### Use TensorRT
```bash
# Install TensorRT (usually pre-installed with JetPack)
sudo apt-get install -y tensorrt
```

### Check Jetson Stats
```bash
# Install jetson-stats for monitoring
pip install jetson-stats
# Run monitoring
jtop
```

## 🚀 Prevention Tips

1. **Always use virtual environments** to isolate packages
2. **Install packages in the correct order** (system packages first)
3. **Use architecture-specific package indexes** when available
4. **Check package compatibility** before installation
5. **Monitor system resources** during installation

## 📞 Getting Help

If you're still experiencing issues:

1. **Run the diagnostic script**: `python3 diagnose_illegal_instruction.py`
2. **Check the logs** in the `benchmark_results/` directory
3. **Review system resources** with `htop` and `tegrastats`
4. **Check for error patterns** in the output above

## 🔗 Additional Resources

- [Jetson Nano Developer Kit Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [PyTorch for Jetson](https://pytorch.org/get-started/locally/)
- [ONNX Runtime for ARM64](https://onnxruntime.ai/)
- [OpenCV for Jetson](https://opencv.org/)

---

**Remember**: The "Illegal Instruction" error is almost always fixable with the right approach. Start with the quick fix and work through the solutions systematically.
