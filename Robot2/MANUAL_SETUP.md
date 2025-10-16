# Manual Setup for Jetson Nano ONNX Inference

## Step 1: Create Virtual Environment

```bash
# Navigate to your project
cd ~/deepcrack-jetauto/Robot2

# First, make sure Python 3 is installed
python3 --version
# If not found, install it:
# sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment with explicit Python 3
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify you're using Python 3 in the venv
python --version
# Should show Python 3.x.x, not Python 2.7
```

## Step 2: Install Dependencies (One by One)

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install ONNX Runtime (CPU version for compatibility)
pip install onnxruntime

# Install OpenCV
pip install opencv-python

# Install NumPy
pip install numpy

# Install PyYAML (if needed for config files)
pip install PyYAML
```

## Step 3: Test Installation

```bash
# Test Python and packages
python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
```

## Step 4: Run Simple Inference

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the simple inference script
python3 simple_inference.py
```

## Step 5: Check Results

```bash
# Check if output files were created
ls -la result.png mask.png
```

## Troubleshooting

### If you get "command not found" errors:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Check Python version
python3 --version
```

### If ONNX Runtime fails:
```bash
# Try CPU-only version
pip uninstall onnxruntime
pip install onnxruntime==1.12.1
```

### If OpenCV fails:
```bash
# Install system OpenCV first
sudo apt-get install python3-opencv
```

## Expected Output

You should see:
```
🚀 Starting simple ONNX inference...
📦 Loading ONNX model...
✅ Model loaded successfully
📊 Input: input, Output: output
📷 Loading image...
   Original size: 640x480
   Preprocessed size: (1, 3, 256, 256)
🔄 Running inference...
   Inference time: 45.23 ms
🎨 Post-processing...
✅ Result saved: result.png
✅ Mask saved: mask.png
🎉 Inference completed successfully!
```

## File Structure

```
Robot2/
├── venv/                    # Virtual environment
├── simple_inference.py      # Simple inference script
├── ../onnx/BEST.onnx       # Your ONNX model
├── ../sample_image.jpg     # Input image
├── result.png              # Output with overlay
└── mask.png                # Raw mask
```
