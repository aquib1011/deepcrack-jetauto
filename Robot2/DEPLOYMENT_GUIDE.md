# 🚀 DeepCrack Deployment Guide for Hiwonder JetAuto

Complete step-by-step guide to convert your `.pth` model to ONNX and deploy it on your Hiwonder JetAuto robot with comprehensive performance monitoring.

## 📋 System Overview

| Component | Details |
|-----------|---------|
| **Platform** | Hiwonder JetAuto (Jetson Nano Developer Kit) |
| **OS** | Ubuntu 18.04.6 LTS |
| **JetPack** | 4.6.4/4.6.5 |
| **Architecture** | ARM64 (aarch64) |
| **GPU** | 128-core Maxwell |
| **RAM** | ~3.9 GB LPDDR4 |
| **Storage** | 32 GB micro-SD (25 GB used) |

## 🎯 What This Guide Covers

1. **Environment Setup** - Python environment and dependencies
2. **Model Conversion** - PTH → ONNX with Jetson optimization
3. **Performance Benchmarking** - Memory, inference time, power consumption
4. **Real-time Monitoring** - System metrics during inference
5. **Deployment Testing** - End-to-end validation

## 🚀 Quick Start (Recommended)

### Option 1: Automated Deployment

```bash
# On your Jetson Nano, navigate to Robot2 directory
cd ~/deepcrack-jetauto/Robot2

# Make the script executable
chmod +x quick_start.sh

# Run automated deployment
./quick_start.sh
```

This will:
- Set up Python environment
- Install all dependencies
- Convert your PTH model to ONNX
- Run comprehensive benchmarks
- Generate performance reports

### Option 2: Manual Step-by-Step

## 📦 Step 1: Environment Setup

### 1.1 System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-dev python3-venv python3-setuptools
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev libatlas-base-dev gfortran
sudo apt install -y libhdf5-serial-dev hdf5-tools
sudo apt install -y libopencv-dev python3-opencv
```

### 1.2 Python Environment

```bash
# Navigate to Robot2 directory
cd ~/deepcrack-jetauto/Robot2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for conversion)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ONNX packages
pip install onnx onnxruntime onnxruntime-gpu onnxsim

# Install additional dependencies
pip install opencv-python numpy scipy pillow pyyaml matplotlib seaborn
pip install einops loralib jetson-stats psutil
```

## 🔄 Step 2: Model Conversion (PTH → ONNX)

### 2.1 Convert Your Model

```bash
# Activate virtual environment
source venv/bin/activate

# Run conversion script
python3 convert_pth_to_onnx.py
```

This will:
- Load your `BEST.pth` model
- Convert it to optimized ONNX format
- Verify the conversion
- Save as `BEST_optimized.onnx`

### 2.2 Verify Conversion

The script will automatically verify the conversion by:
- Checking ONNX model validity
- Comparing PyTorch vs ONNX outputs
- Reporting conversion success/failure

## 🧪 Step 3: Performance Benchmarking

### 3.1 Run Comprehensive Benchmark

```bash
# Run full benchmark suite
python3 benchmark_onnx_inference.py \
    --model ../onnx/BEST_optimized.onnx \
    --config config.yaml \
    --image ../sample_image.jpg \
    --output benchmark_results.json \
    --duration 30 \
    --warmup 5
```

### 3.2 Benchmark Types

The script runs three types of benchmarks:

1. **Single Inference** - Measures single image processing
2. **Batch Inference** - Tests multiple image processing
3. **Continuous Inference** - Long-term performance and stability

### 3.3 Metrics Collected

- **Inference Time** - Per-image processing time
- **FPS** - Frames per second
- **Memory Usage** - RAM consumption
- **CPU Usage** - CPU utilization
- **GPU Metrics** - GPU frequency and usage
- **Power Consumption** - Via tegrastats
- **Temperature** - System thermal monitoring

## 📊 Step 4: Real-time Monitoring

### 4.1 Start Monitoring

```bash
# Start real-time monitoring
python3 realtime_monitor.py \
    --interval 0.5 \
    --log monitoring.csv \
    --duration 60
```

### 4.2 Monitor During Inference

```bash
# Terminal 1: Start monitoring
python3 realtime_monitor.py --interval 0.5 --log monitoring.csv

# Terminal 2: Run inference
python3 run_image.py --config config.yaml --input ../sample_image.jpg --output test_output.png
```

## 🚀 Step 5: Complete Deployment

### 5.1 Automated Deployment

```bash
# Run complete deployment pipeline
python3 deploy_and_test.py \
    --benchmark-duration 30 \
    --monitoring-duration 60
```

### 5.2 Manual Testing

```bash
# Test inference
python3 run_image.py \
    --config config.yaml \
    --input ../sample_image.jpg \
    --output test_output.png

# Test with different images
python3 run_image.py \
    --config config.yaml \
    --input /path/to/your/image.jpg \
    --output result.png
```

## 📈 Step 6: Performance Analysis

### 6.1 Review Results

After running benchmarks, check the `benchmark_results/` directory:

```bash
ls -la benchmark_results/
# You'll see:
# - benchmark_*.json (detailed results)
# - monitoring_*.csv (real-time metrics)
# - deployment_summary.json (summary report)
```

### 6.2 Key Metrics to Monitor

| Metric | Target | Notes |
|--------|--------|-------|
| **Inference Time** | < 100ms | Single image processing |
| **FPS** | > 10 | Real-time performance |
| **Memory Usage** | < 80% | Avoid OOM errors |
| **CPU Usage** | < 90% | Leave headroom for other tasks |
| **Temperature** | < 70°C | Thermal throttling threshold |

## 🔧 Troubleshooting

### Common Issues

#### 1. ONNX Conversion Fails

```bash
# Check PyTorch model
python3 -c "import torch; print(torch.__version__)"

# Check model file
ls -la ../onnx/BEST.pth

# Run with verbose output
python3 convert_pth_to_onnx.py --verbose
```

#### 2. Inference Fails

```bash
# Check ONNX model
python3 -c "import onnx; model = onnx.load('../onnx/BEST_optimized.onnx'); onnx.checker.check_model(model)"

# Check providers
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

#### 3. Performance Issues

```bash
# Check system resources
htop
tegrastats

# Monitor during inference
python3 realtime_monitor.py --interval 0.1
```

### Memory Optimization

If you encounter memory issues:

1. **Reduce batch size** in inference scripts
2. **Use CPU-only inference** for testing
3. **Close unnecessary applications**
4. **Monitor memory usage** with `htop`

### Performance Optimization

1. **Use TensorRT** for maximum performance
2. **Optimize image preprocessing**
3. **Use appropriate input sizes**
4. **Monitor thermal throttling**

## 📁 File Structure

```
Robot2/
├── convert_pth_to_onnx.py      # PTH to ONNX conversion
├── benchmark_onnx_inference.py # Performance benchmarking
├── realtime_monitor.py         # Real-time monitoring
├── deploy_and_test.py          # Complete deployment pipeline
├── quick_start.sh              # Automated setup script
├── config.yaml                 # Configuration file
├── utils.py                    # Utility functions
├── run_image.py               # Image inference script
├── venv/                      # Python virtual environment
└── benchmark_results/         # Results directory
    ├── benchmark_*.json       # Benchmark results
    ├── monitoring_*.csv       # Monitoring logs
    └── deployment_summary.json # Summary report
```

## 🎯 Next Steps

After successful deployment:

1. **Integrate with ROS** - Use the ONNX model in your ROS nodes
2. **Optimize for your use case** - Adjust parameters based on your requirements
3. **Monitor in production** - Use the monitoring tools during real-world usage
4. **Scale up** - Deploy on multiple robots if needed

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the benchmark results for performance insights
3. Monitor system resources during inference
4. Check the logs in the `benchmark_results/` directory

## 🔗 Additional Resources

- [Jetson Nano Developer Kit Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Hiwonder JetAuto Documentation](https://www.hiwonder.com/)

---

**Happy Deploying! 🚀**
