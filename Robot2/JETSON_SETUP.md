# NVIDIA Jetson Nano ONNX Inference Setup Guide

This guide will help you set up and run ONNX inference on your NVIDIA Jetson Nano with optimal performance using TensorRT acceleration.

## System Specifications
- **Device**: NVIDIA Jetson Nano (Developer Kit Version)
- **Jetpack**: UNKNOWN [L4T 32.7.6]
- **CUDA**: 10.2.300
- **TensorRT**: 8.2.1.8
- **OpenCV**: 4.5.5 (with CUDA support)

## Step-by-Step Setup

### 1. Initial System Setup

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3 development tools
sudo apt-get install -y python3-pip python3-dev python3-venv

# Verify Python 3 version (should be 3.6+)
python3 --version
```

### 2. Install Dependencies

```bash
# Navigate to your project directory
cd /path/to/your/Robot/Robot2

# Make the install script executable
chmod +x scripts/install_jetson.sh

# Run the installation script
./scripts/install_jetson.sh
```

### 3. Verify Installation

```bash
# Test ONNX Runtime providers
python3 -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
print('TensorRT available:', 'TensorrtExecutionProvider' in ort.get_available_providers())
print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())
"
```

Expected output should include:
- `TensorrtExecutionProvider`
- `CUDAExecutionProvider` 
- `CPUExecutionProvider`

### 4. Prepare Your Model

Ensure your ONNX model is in the correct location:
```bash
# Check if model exists
ls -la ../onnx/BEST.onnx

# If not found, copy your model to the onnx directory
# cp /path/to/your/model.onnx ../onnx/BEST.onnx
```

## Running Inference

### Option 1: Single Image Inference

```bash
# Basic usage
python3 run_image_py3.py --input ../sample_image.jpg --output result.png

# With verbose output
python3 run_image_py3.py --input ../sample_image.jpg --output result.png --verbose

# Save both overlay and raw mask
python3 run_image_py3.py --input ../sample_image.jpg --output result.png --save_mask mask.png
```

### Option 2: Webcam/Video Stream Inference

```bash
# Use default camera (index 0)
python3 run_webcam_py3.py --display

# Use specific camera index
python3 run_webcam_py3.py --source 1 --display

# Use RTSP stream
python3 run_webcam_py3.py --source "rtsp://your-camera-ip/stream" --display

# Limit FPS for better performance
python3 run_webcam_py3.py --display --fps 15
```

### Option 3: Batch Processing

```bash
# Process all images in a folder
for img in /path/to/images/*.jpg; do
    python3 run_image_py3.py --input "$img" --output "results/$(basename "$img")"
done
```

## Performance Optimization

### 1. Enable Maximum Performance Mode

```bash
# Set to MAXN mode for maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. Monitor Performance

```bash
# Monitor system resources
tegrastats

# Monitor GPU usage
sudo tegrastats --interval 1000
```

### 3. Optimize Model for TensorRT

The first run with TensorRT will be slower as it builds the optimized engine. Subsequent runs will be much faster.

## Troubleshooting

### Common Issues

#### 1. "No module named 'onnxruntime'"
```bash
# Reinstall ONNX Runtime
pip3 uninstall onnxruntime onnxruntime-gpu
pip3 install onnxruntime-gpu
```

#### 2. "TensorRT not available"
```bash
# Check TensorRT installation
dpkg -l | grep tensorrt

# If missing, install JetPack SDK
# Follow NVIDIA's JetPack installation guide
```

#### 3. "CUDA out of memory"
```bash
# Reduce image size in config.yaml
# Or use CPU fallback
python3 run_image_py3.py --input image.jpg --output result.png
# Edit config.yaml to use only CPUExecutionProvider
```

#### 4. Low FPS Performance
```bash
# Enable performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Reduce image resolution in config.yaml
# Use smaller img_size: [128, 128] instead of [256, 256]
```

### Performance Tips

1. **First Run**: The first inference with TensorRT will be slow as it builds the engine
2. **Image Size**: Smaller images = faster inference
3. **Batch Processing**: Process multiple images in sequence for better TensorRT utilization
4. **Memory**: Monitor memory usage with `tegrastats`
5. **Temperature**: Ensure adequate cooling for sustained performance

## Expected Performance

With your Jetson Nano specifications, you should expect:
- **Image inference**: 50-200ms per image (depending on size)
- **Webcam FPS**: 10-30 FPS (depending on resolution and model complexity)
- **Memory usage**: 1-2GB RAM, 500MB-1GB GPU memory

## Configuration Options

Edit `config.yaml` to customize:

```yaml
model_path: "../onnx/BEST.onnx"  # Path to your ONNX model
img_size: [256, 256]             # Input image size (smaller = faster)
threshold: 0.5                   # Segmentation threshold
providers: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
draw_overlay: true               # Show overlay on output
alpha: 0.4                       # Overlay transparency
```

## Next Steps

1. Test with your specific model and images
2. Adjust `img_size` and `threshold` for your use case
3. Monitor performance and optimize as needed
4. Consider model quantization for even better performance

For more advanced usage, see the original README.md file.
