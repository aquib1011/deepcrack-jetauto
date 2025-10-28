# Jetson Orin Nano Deployment Package

**Generated**: 2025-10-27 23:34:33

## 📦 Package Contents

This package contains everything needed to deploy and test the ResUNet crack detection model on Jetson Orin Nano Developer Kit.

### Model Files
- `model_resunet.py` - ResUNet architecture with MobileViT blocks
- `BEST.pth` - Trained model weights (DeepCrack dataset)
- `sample_image.jpg` - Test image for verification

### Testing Scripts
- `profile_edge_device.py` - Comprehensive profiling (memory, time, power)
- `test_single_inference.py` - Quick single image test
- `verify_setup.py` - Environment verification
- `visualize_results.py` - Results visualization and reporting

### Setup Files
- `quick_start_jetson.sh` - Automated setup script
- `requirements_jetson.txt` - Python dependencies
- `JETSON_ORIN_NANO_COMPLETE_SETUP.md` - Complete setup guide
- `README_TESTING.md` - Testing framework documentation

## 🚀 Quick Start on Jetson

### 1. Transfer Package to Jetson

**Option A: Using SCP**
```bash
# On your host computer
scp jetson_deployment.zip jetson@<jetson-ip>:~/
```

**Option B: Using USB drive**
- Copy `jetson_deployment.zip` to USB drive
- Insert USB drive into Jetson
- Copy from mounted drive

### 2. Extract and Setup

```bash
# On Jetson device
cd ~
unzip jetson_deployment.zip
cd jetson_deployment

# Run setup
chmod +x quick_start_jetson.sh
./quick_start_jetson.sh
```

### 3. Run Tests

```bash
# Activate environment
source ~/jetson_crack_detection/venv/bin/activate

# Quick test
python3 test_single_inference.py

# Full profiling
python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg

# Visualize results
python3 visualize_results.py --input profile_results_*.json --report
```

## 📚 Documentation

See included documentation for detailed instructions:
- **JETSON_ORIN_NANO_COMPLETE_SETUP.md** - Step-by-step setup from scratch
- **README_TESTING.md** - Complete testing framework guide

## ⚙️ System Requirements

- Jetson Orin Nano Developer Kit
- JetPack 6.0 or later
- Python 3.8+
- 8GB RAM minimum
- 10GB free storage

## 📊 What Will Be Measured

- **Inference Time**: mean, std, min, max, percentiles, FPS
- **Memory Usage**: RAM and GPU memory consumption
- **Power Consumption**: Watts and energy per inference
- **Utilization**: GPU and CPU usage percentages

## 🔧 Support

For issues or questions:
1. Check JETSON_ORIN_NANO_COMPLETE_SETUP.md for setup issues
2. Run `verify_setup.py` to diagnose problems
3. Review README_TESTING.md for usage examples

## 📝 Notes

- Ensure Jetson is in maximum performance mode for consistent results
- Power measurements require tegrastats (Jetson-specific)
- First run will be slower due to GPU initialization

---

**Package Version**: 1.0
**Compatible with**: JetPack 6.0+, PyTorch 2.0+
