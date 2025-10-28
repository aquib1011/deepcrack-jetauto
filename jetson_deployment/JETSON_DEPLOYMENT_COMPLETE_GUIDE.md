# 🚀 Complete Jetson Orin Nano Deployment Guide

## Overview

This guide walks you through deploying and testing your ResUNet crack detection model on a **brand new Jetson Orin Nano Developer Kit**, from initial hardware setup to comprehensive performance profiling.

---

## 📦 What You Have

### Model Information
- **Architecture**: ResUNet with MobileViT blocks, LoRA attention, and Dynamic MS-EEM
- **Training Dataset**: DeepCrack (crack detection)
- **Model File**: `model_resunet.py`
- **Weights**: `onnx/BEST.pth` (2.24 MB)
- **Input Size**: 256×256×3 (RGB)
- **Output**: 256×256×1 (Binary crack mask)

### Deployment Package
- **Location**: `jetson_deployment.zip` (1.89 MB)
- **Contents**: Model, scripts, documentation, dependencies
- **Ready to transfer**: Yes ✅

---

## 🎯 What You'll Measure

1. **Inference Time**
   - Mean, median, standard deviation
   - Min, max, 95th/99th percentiles
   - Frames per second (FPS)

2. **Memory Usage**
   - Process RAM consumption
   - System RAM usage
   - GPU memory (allocated, reserved, peak)

3. **Power Consumption** (Jetson-specific)
   - Mean, min, max power in Watts
   - Energy per inference (Joules/mJ)
   - GPU and CPU utilization percentages

---

## 🛠️ Hardware Setup (Step-by-Step)

### What You Need
- ✅ Jetson Orin Nano Developer Kit
- ✅ microSD card (64GB+ recommended, UHS-1)
- ✅ USB-C power supply (15W) or DC barrel jack (25W/30W for higher power modes)
- ✅ HDMI monitor
- ✅ USB keyboard & mouse
- ✅ Ethernet cable (recommended) or WiFi
- ✅ Host computer (for flashing JetPack)

### Initial Setup

#### 1. Flash JetPack to microSD Card

**Option A: Using Balena Etcher (Easiest)**
1. Download JetPack SD Card Image from: https://developer.nvidia.com/embedded/jetpack
2. Download Balena Etcher: https://www.balena.io/etcher/
3. Insert microSD card into your computer
4. Open Balena Etcher
5. Select JetPack image file
6. Select your microSD card
7. Click "Flash"

**Option B: Using NVIDIA SDK Manager (More control)**
1. Download SDK Manager: https://developer.nvidia.com/sdk-manager
2. Install on Ubuntu host: `sudo apt install ./sdkmanager_*.deb`
3. Run: `sdkmanager`
4. Select "Jetson Orin Nano Developer Kit"
5. Select JetPack 6.0 (or latest)
6. Follow on-screen instructions

#### 2. First Boot
1. Insert flashed microSD card into Jetson
2. Connect monitor (HDMI), keyboard, mouse
3. Connect power supply
4. Device boots automatically
5. Follow Ubuntu setup wizard:
   - Set username and password (remember these!)
   - Configure WiFi/network
   - Complete initial setup

#### 3. Verify Basic Operation
```bash
# Check system info
cat /etc/nv_tegra_release

# Check GPU
nvidia-smi  # May not work on Jetson, use tegrastats instead
tegrastats

# Check CUDA
nvcc --version

# Check Python
python3 --version
```

---

## 📥 Transfer Deployment Package to Jetson

### Method 1: SCP (Network Transfer) - Recommended

**Step 1**: Find Jetson's IP address
```bash
# On Jetson terminal
hostname -I
# Or
ifconfig
```

**Step 2**: Transfer from your computer
```bash
# On your Windows computer (PowerShell):
scp jetson_deployment.zip jetson@<JETSON_IP>:~/

# Example:
# scp jetson_deployment.zip jetson@192.168.1.100:~/
```

**Step 3**: Extract on Jetson
```bash
# On Jetson terminal
cd ~
unzip jetson_deployment.zip
cd jetson_deployment
```

### Method 2: USB Drive

**Step 1**: Copy to USB drive
- Copy `jetson_deployment.zip` to a USB drive from your computer

**Step 2**: Transfer from USB
```bash
# On Jetson, insert USB drive
# It will auto-mount to /media/<username>/<drive_name>

cd ~
cp /media/jetson/*/jetson_deployment.zip .
unzip jetson_deployment.zip
cd jetson_deployment
```

### Method 3: Git Repository (If you have one)
```bash
# On Jetson
cd ~
git clone <your-repo-url> jetson_deployment
cd jetson_deployment
```

---

## ⚙️ Complete Jetson Setup

### 1. Run Automated Setup Script

```bash
cd ~/jetson_deployment
chmod +x quick_start_jetson.sh
./quick_start_jetson.sh
```

This script will:
- ✅ Check Python version
- ✅ Create project directory (`~/jetson_crack_detection`)
- ✅ Set up virtual environment
- ✅ Verify PyTorch installation
- ✅ Install dependencies
- ✅ Copy model files
- ✅ Configure performance mode

### 2. Manual Setup (If script fails)

See `JETSON_ORIN_NANO_COMPLETE_SETUP.md` for detailed manual setup instructions.

**Key Steps:**
```bash
# Create project directory
mkdir -p ~/jetson_crack_detection
cd ~/jetson_crack_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Jetson (if not already installed)
# Check: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# For JetPack 6.0:
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install --no-cache-dir $TORCH_INSTALL

# Install dependencies
pip install -r ~/jetson_deployment/requirements_jetson.txt

# Copy model files
cp ~/jetson_deployment/model_resunet.py .
cp ~/jetson_deployment/BEST.pth .
cp ~/jetson_deployment/sample_image.jpg .
cp ~/jetson_deployment/*.py .
```

### 3. Verify Setup

```bash
cd ~/jetson_crack_detection
source venv/bin/activate
python3 verify_setup.py
```

Expected output:
```
JETSON SETUP VERIFICATION
==============================================================

📋 SYSTEM INFORMATION
✓ Python Version                             v3.10.x
✓ Jetson Device                              NVIDIA Jetson Orin Nano
✓ System Memory                              7.5 GB total, 6.2 GB available
✓ Disk Space                                 42.3 GB free

🎮 GPU & CUDA
✓ CUDA Compiler                              v12.x
✓ PyTorch                                    v2.3.0, CUDA: True, NVIDIA GPU
✓ TorchVision                                v0.18.0

📦 PYTHON PACKAGES
✓ SciPy                                      1.10.1
✓ OpenCV                                     4.8.0.74
✓ Pillow                                     10.0.0
✓ Einops                                     0.6.1
✓ LoRA lib                                   OK
✓ PSUtil                                     5.9.5
✓ NumPy                                      1.24.3

📁 MODEL FILES
✓ Model File (BEST.pth)                      pass    2.2 MB

🏗️  MODEL ARCHITECTURE
✓ model_resunet.py                           pass    Found

==============================================================
✅ All critical checks passed!

🚀 You're ready to run profiling:
   python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg
```

---

## 🧪 Running Tests

### Test 1: Quick Single Inference Test

Verify the model works on a single image:

```bash
cd ~/jetson_crack_detection
source venv/bin/activate

python3 test_single_inference.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --output test_result.png
```

**Expected Output:**
```
Single Image Inference Test
==============================================================
Device: cuda
Model: BEST.pth
Image: sample_image.jpg
==============================================================

[1/4] Loading model...
✓ Model loaded successfully

[2/4] Preprocessing image...
✓ Image preprocessed: (400, 300) -> (256, 256)

[3/4] Running inference...
✓ Inference completed in 45.23 ms
  Crack coverage: 12.34% (8123/65536 pixels)

[4/4] Creating visualization...
✓ Visualization saved to: test_result.png
✓ Mask saved to: test_result_mask.png

==============================================================
✓ Test completed successfully!
==============================================================
```

**What you get:**
- `test_result.png`: Side-by-side visualization (input, mask, overlay)
- `test_result_mask.png`: Binary crack mask

### Test 2: Set Performance Mode

For consistent and maximum performance:

```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance (15W for Orin Nano)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify
sudo nvpmodel -q verbose
```

**Power Modes:**
- Mode 0: 15W (Max Performance)
- Mode 1: 10W (Balanced)
- Mode 2: 7W (Low Power)

### Test 3: Comprehensive Profiling

Run complete performance analysis:

```bash
cd ~/jetson_crack_detection
source venv/bin/activate

# Ensure performance mode is enabled
sudo nvpmodel -m 0
sudo jetson_clocks

# Run profiling (100 iterations - takes ~1-2 minutes)
python3 profile_edge_device.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --iterations 100 \
    --output results_15w_mode.json
```

**What happens:**
1. Loads model and weights
2. Warms up GPU (10 iterations)
3. Runs 100 timed inferences
4. Monitors RAM and GPU memory
5. Records power consumption via tegrastats
6. Calculates statistics

**Sample Output:**
```
#############################################################
# JETSON ORIN NANO - COMPREHENSIVE EDGE PROFILING
#############################################################
Timestamp: 2025-01-28 14:30:00

============================================================
Model Profiler Initialized
============================================================
Device: cuda
Model Path: BEST.pth

[1/4] Loading Model...
✓ Model loaded successfully
  Total parameters: 1,234,567
  Trainable parameters: 1,234,567
  Model size: 4.71 MB (FP32)

[2/4] Warming up GPU (10 iterations)...
✓ Warmup complete

[3/4] Benchmarking Inference (100 iterations)...
  Progress: 20/100
  Progress: 40/100
  Progress: 60/100
  Progress: 80/100
  Progress: 100/100
✓ Inference benchmark complete
  Mean: 45.23 ms
  FPS: 22.11

[4/4] Profiling Memory Usage...
✓ Memory profiling complete
  RAM (Process): 523.45 MB
  GPU Memory: 342.18 MB

✓ Started tegrastats logging to tegrastats_20250128_143000.log
✓ Stopped tegrastats

==============================================================
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
  RAM (System Used): 3245.12 MB / 7654.32 MB (42.4%)
  GPU Memory (Allocated): 342.18 MB
  GPU Memory (Reserved): 512.00 MB
  GPU Memory (Peak): 358.92 MB

⚡ POWER CONSUMPTION
  Mean: 8.45 W
  Max: 10.23 W
  Min: 7.82 W
  Std Dev: 0.45 W

🎮 GPU UTILIZATION
  Mean: 78.5%
  Max: 95.2%

🖥️  CPU UTILIZATION
  Mean: 35.2%
  Max: 58.7%

🔋 ENERGY PER INFERENCE
  0.3823 Joules (382.30 mJ)

==============================================================

💾 Results saved to: results_15w_mode.json

✓ Profiling complete!
```

### Test 4: Compare Different Power Modes

Test performance across different power modes:

```bash
# Test 15W mode
sudo nvpmodel -m 0
sudo jetson_clocks
python3 profile_edge_device.py --iterations 100 --output results_15w.json

# Test 10W mode
sudo nvpmodel -m 1
python3 profile_edge_device.py --iterations 100 --output results_10w.json

# Test 7W mode
sudo nvpmodel -m 2
python3 profile_edge_device.py --iterations 100 --output results_7w.json
```

---

## 📊 Visualizing Results

### Generate Summary Plot and Report

```bash
cd ~/jetson_crack_detection
source venv/bin/activate

# Single run visualization
python3 visualize_results.py \
    --input results_15w_mode.json \
    --output summary_plot.png \
    --report

# This creates:
# - summary_plot.png (4-panel visualization)
# - summary_plot_report.txt (detailed text report)
```

**What you get:**

**`summary_plot.png`** contains:
1. Inference time distribution histogram
2. Memory usage bar chart (RAM + GPU)
3. Power consumption metrics
4. Performance summary text box

**`summary_plot_report.txt`** contains:
- Device information
- Model architecture details
- Complete inference statistics
- Memory breakdown
- Power consumption analysis
- Energy per inference calculations

### Compare Multiple Runs

```bash
python3 visualize_results.py \
    --input "results_15w.json,results_10w.json,results_7w.json" \
    --labels "15W Mode,10W Mode,7W Mode" \
    --output power_mode_comparison.png \
    --report
```

This creates comparison plots showing inference time, memory, and power across different configurations.

---

## 📈 Understanding Your Results

### Typical Performance Expectations

For ResUNet on Jetson Orin Nano:

| Metric | 15W Mode | 10W Mode | 7W Mode |
|--------|----------|----------|---------|
| Inference Time | 30-50 ms | 40-70 ms | 60-100 ms |
| FPS | 20-33 | 14-25 | 10-16 |
| GPU Memory | 300-500 MB | 300-500 MB | 300-500 MB |
| Power | 8-12 W | 6-9 W | 4-7 W |
| Energy/Inference | 300-500 mJ | 300-500 mJ | 350-550 mJ |

### Key Metrics Explained

**Inference Time**
- **Mean**: Average time across all runs
- **Std Dev**: Consistency (lower is better)
- **P95**: 95% of inferences are faster than this
- **FPS**: For real-time applications (need >15 FPS typically)

**Memory**
- **Process RSS**: Actual RAM your model uses
- **GPU Allocated**: GPU memory for tensors/weights
- **GPU Peak**: Maximum GPU memory needed

**Power & Energy**
- **Mean Power**: Average consumption during inference
- **Energy/Inference**: Power × Time (important for battery life)
- Lower power mode = lower performance but better efficiency

### What's Good vs. Concerning

✅ **Good Signs:**
- Mean inference time < 50ms (>20 FPS)
- Std deviation < 10% of mean (consistent)
- GPU memory < 1GB (leaves room for other tasks)
- Power consumption matches expected mode

⚠️ **Concerning Signs:**
- High std deviation (>20% of mean) - indicates inconsistency
- GPU memory usage close to device limit (4GB on Orin Nano)
- Very high power (>15W in 15W mode) - thermal throttling risk
- FPS < 10 - too slow for real-time applications

---

## 🔧 Troubleshooting

### Issue: CUDA Not Available

**Symptom:**
```
Device: cpu (CUDA not available)
```

**Solution:**
```bash
# Check PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch
pip uninstall torch
# Get correct wheel from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
pip install <torch_wheel_for_jetpack_6.0.whl>

# Verify CUDA
nvcc --version
```

### Issue: Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# 1. Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# 2. Reduce iterations
python3 profile_edge_device.py --iterations 10

# 3. Check memory
free -h
# If low, increase swap

# 4. Close other applications
pkill -f firefox  # Example
```

### Issue: tegrastats Not Found

**Symptom:**
```
⚠️  Not running on Jetson - power monitoring disabled
```

**This is OK if:**
- You're testing on non-Jetson hardware first
- Power metrics won't be available, but time/memory will work

**On actual Jetson:**
```bash
# Check tegrastats
which tegrastats
# Should output: /usr/bin/tegrastats

# If not found, reinstall JetPack
```

### Issue: Slow Inference

**Symptom:**
- Mean inference time > 100ms
- FPS < 10

**Solutions:**
```bash
# 1. Enable performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. Check GPU is being used
python3 -c "import torch; print(torch.cuda.is_available())"

# 3. Check system load
htop
# Kill unnecessary processes

# 4. Check thermal throttling
tegrastats
# Look for temp warnings
```

### Issue: Model Loading Error

**Symptom:**
```
✗ Error loading model: <various errors>
```

**Solutions:**
```bash
# Check file exists
ls -lh BEST.pth

# Check file integrity
python3 -c "import torch; print(torch.load('BEST.pth').keys())"

# Verify model architecture
python3 -c "from model_resunet import build_resunet; build_resunet()"

# Check dependencies
pip install scipy einops loralib
```

---

## 📚 Complete File Reference

### In `jetson_deployment/` Package

| File | Purpose | When to Use |
|------|---------|-------------|
| `README.md` | Package overview | First read |
| `TRANSFER_INSTRUCTIONS.txt` | How to transfer files | Before copying to Jetson |
| `model_resunet.py` | Model architecture | Required for all tests |
| `BEST.pth` | Trained weights | Required for all tests |
| `sample_image.jpg` | Test image | For quick testing |
| `quick_start_jetson.sh` | Automated setup | First time setup |
| `verify_setup.py` | Environment check | After setup |
| `test_single_inference.py` | Quick test | Verify model works |
| `profile_edge_device.py` | **Main profiling** | **Primary testing** |
| `visualize_results.py` | Results visualization | After profiling |
| `requirements_jetson.txt` | Python dependencies | For manual setup |
| `JETSON_ORIN_NANO_COMPLETE_SETUP.md` | Detailed setup guide | Manual setup reference |
| `README_TESTING.md` | Testing framework docs | Detailed usage |

---

## 🎯 Recommended Testing Workflow

### Phase 1: Initial Setup (30-60 min)
1. Flash JetPack to microSD
2. Boot Jetson and complete setup wizard
3. Transfer deployment package
4. Run `quick_start_jetson.sh`
5. Run `verify_setup.py`

### Phase 2: Quick Validation (5 min)
1. Set performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. Run single inference test: `python3 test_single_inference.py`
3. Verify output looks correct

### Phase 3: Comprehensive Profiling (10 min)
1. Run 100-iteration profiling: `python3 profile_edge_device.py --iterations 100`
2. Review console output for immediate feedback
3. Check `profile_results_*.json` file created

### Phase 4: Analysis & Visualization (5 min)
1. Generate plots: `python3 visualize_results.py --input results.json --report`
2. Review plots and text report
3. Document your findings

### Phase 5: Extended Testing (Optional, 30 min)
1. Test different power modes
2. Compare multiple configurations
3. Test on additional images
4. Generate comparison plots

---

## 📊 Expected Timeline

| Task | Estimated Time |
|------|----------------|
| Hardware setup & JetPack flash | 30-60 minutes |
| Software setup (automated) | 10-15 minutes |
| Dependency installation | 15-30 minutes |
| First inference test | 2-5 minutes |
| Single profiling run (100 iter) | 5-10 minutes |
| Visualization generation | 1-2 minutes |
| **Total (first time)** | **~90 minutes** |
| **Subsequent tests** | **~10 minutes each** |

---

## 🎓 Next Steps

After completing testing:

1. **Document Your Results**
   - Save all JSON result files
   - Keep generated plots
   - Note any issues encountered

2. **Optimize if Needed**
   - Try FP16 inference for faster speed
   - Consider model quantization (INT8)
   - Explore TensorRT conversion for maximum performance

3. **Deploy to Application**
   - Integrate model into your pipeline
   - Add preprocessing/postprocessing
   - Implement real-time inference loop

4. **Monitor in Production**
   - Track inference times over time
   - Monitor memory leaks
   - Watch for thermal throttling

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Verification**: Run `verify_setup.py`
2. **Review Logs**: Check `tegrastats_*.log` files
3. **Read Documentation**:
   - `JETSON_ORIN_NANO_COMPLETE_SETUP.md` for setup
   - `README_TESTING.md` for testing details
4. **NVIDIA Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
5. **PyTorch for Jetson**: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

---

## ✅ Success Checklist

Before considering deployment complete:

- [ ] JetPack installed and Jetson boots properly
- [ ] Network connectivity working
- [ ] Deployment package transferred successfully
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA working
- [ ] All dependencies installed
- [ ] `verify_setup.py` passes all checks
- [ ] Model loads without errors
- [ ] Single inference test produces output
- [ ] Comprehensive profiling completes
- [ ] Results JSON files generated
- [ ] Visualization plots created
- [ ] Performance meets requirements
- [ ] Results documented

---

**You're now ready to deploy and test your crack detection model on Jetson Orin Nano! 🚀**

**Package Version**: 1.0  
**Last Updated**: 2025-01-28  
**Tested On**: Jetson Orin Nano Developer Kit, JetPack 6.0

