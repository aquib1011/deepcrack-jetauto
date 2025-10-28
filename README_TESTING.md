# Jetson Orin Nano Edge Device Testing Framework

Complete testing framework for measuring performance, memory, and power consumption of the ResUNet crack detection model on Jetson Orin Nano Developer Kit.

## 📋 Overview

This framework provides comprehensive tools to:
- **Measure inference time** (mean, std, percentiles, FPS)
- **Monitor memory usage** (RAM and GPU memory)
- **Track power consumption** (Watts, Joules per inference)
- **Analyze GPU/CPU utilization**
- **Visualize results** with plots and reports

## 🚀 Quick Start

### On Your Jetson Device

```bash
# 1. Run the quick start script
chmod +x quick_start_jetson.sh
./quick_start_jetson.sh

# 2. Activate virtual environment
source ~/jetson_crack_detection/venv/bin/activate

# 3. Copy your files to the project directory
cd ~/jetson_crack_detection
# Copy model_resunet.py, BEST.pth, sample_image.jpg here

# 4. Verify setup
python3 verify_setup.py

# 5. Run profiling
python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg

# 6. Visualize results
python3 visualize_results.py --input profile_results_*.json --report --show
```

## 📁 Files in This Framework

### Setup & Documentation
- **`JETSON_ORIN_NANO_COMPLETE_SETUP.md`** - Complete setup guide from scratch
- **`README_TESTING.md`** - This file
- **`requirements_jetson.txt`** - Python dependencies for Jetson
- **`quick_start_jetson.sh`** - Automated setup script

### Testing Scripts
- **`verify_setup.py`** - Verify environment and dependencies
- **`profile_edge_device.py`** - Main profiling script (memory, time, power)
- **`test_single_inference.py`** - Quick single image test
- **`visualize_results.py`** - Create plots and reports from results

### Model Files (you need to provide)
- **`model_resunet.py`** - ResUNet model architecture
- **`BEST.pth`** - Trained model weights
- **`sample_image.jpg`** - Test image

## 📊 Detailed Usage

### 1. Verify Setup

Check that all dependencies are installed correctly:

```bash
python3 verify_setup.py
```

This will verify:
- Python version
- PyTorch with CUDA
- All required packages
- Model files
- Jetson-specific tools (tegrastats)

### 2. Test Single Inference

Quick test to verify the model works:

```bash
python3 test_single_inference.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --output test_result.png
```

This will:
- Load the model
- Run inference on one image
- Save visualization showing input, mask, and overlay
- Print inference time and crack detection statistics

### 3. Comprehensive Profiling

Run complete performance profiling:

```bash
python3 profile_edge_device.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --iterations 100 \
    --output my_results.json
```

**Parameters:**
- `--model_path`: Path to `.pth` model weights
- `--image_path`: Path to test image
- `--iterations`: Number of inference iterations (default: 100)
- `--output`: Output JSON file path (optional, auto-generated if not specified)

**What it measures:**
- **Inference Time**: mean, std, min, max, p95, p99, FPS
- **Memory**: Process RAM, System RAM, GPU memory (allocated, reserved, peak)
- **Power**: Mean, max, min power in Watts (using tegrastats)
- **Utilization**: GPU and CPU utilization percentages
- **Energy**: Joules per inference

**Example output:**
```
PROFILING SUMMARY
==============================================================

📊 INFERENCE PERFORMANCE
  Mean Time: 45.23 ms
  Std Dev: 2.15 ms
  Min Time: 42.10 ms
  Max Time: 51.87 ms
  95th Percentile: 48.92 ms
  FPS: 22.11

💾 MEMORY USAGE
  RAM (Process): 523.45 MB
  GPU Memory (Allocated): 342.18 MB
  GPU Memory (Peak): 358.92 MB

⚡ POWER CONSUMPTION
  Mean: 8.45 W
  Max: 10.23 W
  Min: 7.82 W

🔋 ENERGY PER INFERENCE
  0.3823 Joules (382.30 mJ)
```

### 4. Visualize Results

Create plots and reports from profiling results:

```bash
# Single run visualization
python3 visualize_results.py \
    --input profile_results_20250101_120000.json \
    --output summary.png \
    --report

# Compare multiple runs
python3 visualize_results.py \
    --input "run1.json,run2.json,run3.json" \
    --labels "FP32,FP16,INT8" \
    --output comparison.png \
    --report
```

**Parameters:**
- `--input`: JSON file(s) from profiling (comma-separated for comparison)
- `--labels`: Labels for comparison plots (comma-separated)
- `--output`: Output PNG file path
- `--report`: Generate detailed text report
- `--show`: Display plots interactively

**Outputs:**
- Summary plot with 4 subplots:
  - Inference time distribution
  - Memory usage (RAM + GPU)
  - Power consumption
  - Performance metrics summary
- Text report with detailed statistics
- Comparison plots (when comparing multiple runs)

## 🔧 Advanced Usage

### Running Multiple Test Scenarios

Test different power modes:

```bash
# Set 15W mode
sudo nvpmodel -m 0
sudo jetson_clocks

python3 profile_edge_device.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --output results_15w.json

# Set 7W mode
sudo nvpmodel -m 1

python3 profile_edge_device.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --output results_7w.json

# Compare
python3 visualize_results.py \
    --input "results_15w.json,results_7w.json" \
    --labels "15W,7W" \
    --output power_mode_comparison.png
```

### Batch Testing on Multiple Images

```bash
# Create a simple batch script
for img in images/*.jpg; do
    echo "Testing $img..."
    python3 profile_edge_device.py \
        --model_path BEST.pth \
        --image_path "$img" \
        --iterations 50 \
        --output "results_$(basename $img .jpg).json"
done
```

### Custom Iterations for Speed vs. Accuracy

```bash
# Quick test (10 iterations)
python3 profile_edge_device.py --iterations 10 --output quick_test.json

# Standard test (100 iterations) - recommended
python3 profile_edge_device.py --iterations 100 --output standard_test.json

# Thorough test (1000 iterations)
python3 profile_edge_device.py --iterations 1000 --output thorough_test.json
```

## 📈 Understanding the Results

### Inference Time Metrics

- **Mean**: Average inference time across all iterations
- **Median**: Middle value (less affected by outliers)
- **Std Dev**: Variability in inference times
- **Min/Max**: Best and worst case times
- **P95/P99**: 95th and 99th percentile (useful for latency requirements)
- **FPS**: Frames per second (1000 / mean_ms)

### Memory Metrics

- **Process RSS**: Actual RAM used by the process
- **Process VMS**: Virtual memory size
- **System Used**: Total system RAM used
- **GPU Allocated**: GPU memory actively used by tensors
- **GPU Reserved**: GPU memory reserved by PyTorch
- **GPU Peak**: Maximum GPU memory used during profiling

### Power Metrics

- **Mean Power**: Average power consumption during inference
- **Energy per Inference**: Power × Time (in Joules or mJ)
- **GPU/CPU Utilization**: How much of the hardware is being used

### Typical Values for Jetson Orin Nano

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Inference Time | 30-80 ms | Depends on model complexity |
| FPS | 12-30 | Real-time applications need >15 FPS |
| GPU Memory | 200-500 MB | For ResUNet-sized models |
| Power (15W mode) | 8-12 W | During active inference |
| Power (7W mode) | 5-8 W | Lower performance mode |
| Energy/Inference | 200-500 mJ | Lower is better for battery life |

## 🐛 Troubleshooting

### "CUDA not available"
```bash
# Check PyTorch installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch for Jetson
pip uninstall torch
# Download correct wheel from NVIDIA forums
pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

### "tegrastats not found"
- Only works on actual Jetson hardware
- Power metrics will be unavailable on non-Jetson systems
- Other metrics (time, memory) will still work

### "Out of memory"
```bash
# Reduce batch size or iterations
python3 profile_edge_device.py --iterations 10

# Clear cache before running
python3 -c "import torch; torch.cuda.empty_cache()"

# Check available memory
free -h
```

### Model loading errors
```bash
# Verify checkpoint format
python3 -c "import torch; print(torch.load('BEST.pth').keys())"

# Check model architecture matches
python3 -c "from model_resunet import build_resunet; m = build_resunet()"
```

## 📚 Additional Resources

- [Jetson Orin Nano User Guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide)
- [JetPack Documentation](https://docs.nvidia.com/jetson/jetpack/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [tegrastats Documentation](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

## 📧 Support

If you encounter issues:
1. Check `JETSON_ORIN_NANO_COMPLETE_SETUP.md` for detailed setup
2. Run `verify_setup.py` to identify missing dependencies
3. Check tegrastats logs for power monitoring issues
4. Review error messages and stack traces

## 📝 Citation

If you use this framework in your research, please cite:
- DeepCrack dataset and original paper
- Jetson Orin Nano documentation
- PyTorch framework

---

**Version**: 1.0  
**Last Updated**: 2025-01-28  
**Compatible with**: JetPack 6.0, Python 3.8+, PyTorch 2.0+

