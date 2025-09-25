# Python 3 Update Guide for Jetson Nano

Your Jetson Nano currently has Python 2.7, but modern ONNX inference requires Python 3. Follow these steps to update.

## 🚨 Important Notes

- **Python 2.7 is deprecated** and doesn't support modern libraries
- **ONNX Runtime requires Python 3.6+**
- **Your Jetson Nano supports Python 3.8** (recommended)

## Step-by-Step Update Process

### Step 1: Check Current Python Version

```bash
# Check what's currently installed
python --version
python3 --version
which python
which python3
```

### Step 2: Install Python 3

```bash
# Navigate to your project directory
cd /path/to/your/Robot/Robot2

# Make the Python 3 install script executable
chmod +x scripts/install_python3_jetson.sh

# Run the Python 3 installation
./scripts/install_python3_jetson.sh
```

### Step 3: Verify Installation

```bash
# Test the installation
python3 test_python3.py

# Check Python 3 version
python3 --version

# Check pip3 version
pip3 --version
```

### Step 4: Install ONNX Dependencies

```bash
# Now install ONNX Runtime and other dependencies
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh
```

### Step 5: Test ONNX Inference

```bash
# Test with a sample image
python3 run_image_py3.py --input ../sample_image.jpg --output result.png --verbose
```

## Alternative Manual Installation

If the script doesn't work, install manually:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.8 (recommended for Jetson)
sudo apt-get install -y python3.8 python3.8-pip python3.8-dev python3.8-venv

# Set Python 3.8 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.8 1

# Upgrade pip
python3.8 -m pip install --upgrade pip

# Install ONNX Runtime
python3.8 -m pip install onnxruntime-gpu

# Install other dependencies
python3.8 -m pip install opencv-python numpy PyYAML psutil
```

## Troubleshooting

### Issue: "python3: command not found"

```bash
# Install Python 3
sudo apt-get install python3 python3-pip

# Check installation
python3 --version
```

### Issue: "pip3: command not found"

```bash
# Install pip3
sudo apt-get install python3-pip

# Check installation
pip3 --version
```

### Issue: "Permission denied" when installing packages

```bash
# Use --user flag for user installation
python3 -m pip install --user onnxruntime-gpu

# Or install system-wide with sudo
sudo python3 -m pip install onnxruntime-gpu
```

### Issue: "No module named 'onnxruntime'"

```bash
# Verify installation
python3 -c "import onnxruntime; print('ONNX Runtime installed')"

# If not installed, install it
python3 -m pip install onnxruntime-gpu
```

## Expected Output After Installation

When you run `python3 test_python3.py`, you should see:

```
Python version: 3.8.10 (default, ...)
Python executable: /usr/bin/python3
✅ ONNX Runtime imported successfully
Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
✅ OpenCV version: 4.5.5
✅ NumPy version: 1.21.0
```

## Running Your Inference Code

After successful Python 3 installation:

```bash
# Single image inference
python3 run_image_py3.py --input ../sample_image.jpg --output result.png

# Webcam inference
python3 run_webcam_py3.py --display

# With verbose output
python3 run_image_py3.py --input ../sample_image.jpg --output result.png --verbose
```

## Performance Tips

1. **First run is slow**: TensorRT builds optimized engine on first use
2. **Use Python 3.8**: Best compatibility with Jetson Nano
3. **Monitor resources**: Use `tegrastats` to monitor GPU/CPU usage
4. **Enable performance mode**: `sudo nvpmodel -m 0 && sudo jetson_clocks`

## Next Steps

1. ✅ Update to Python 3
2. ✅ Install ONNX Runtime with TensorRT
3. ✅ Test with sample image
4. ✅ Run webcam inference
5. ✅ Optimize for your specific use case

Your Jetson Nano will now be ready for high-performance ONNX inference!
