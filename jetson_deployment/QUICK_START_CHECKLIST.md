# 🚀 Jetson Orin Nano Quick Start Checklist

## Before You Start
- [ ] Jetson Orin Nano Developer Kit (new, unboxed)
- [ ] microSD card (64GB+) with JetPack 6.0 flashed
- [ ] Power supply (USB-C 15W or DC 25W/30W)
- [ ] Monitor, keyboard, mouse
- [ ] `jetson_deployment.zip` file ready

---

## Part 1: Hardware Setup (30 min)

### Step 1: Flash microSD Card
- [ ] Download JetPack 6.0 SD image
- [ ] Use Balena Etcher to flash microSD card
- [ ] Safely eject microSD card

### Step 2: Physical Setup
- [ ] Insert microSD card into Jetson
- [ ] Connect HDMI monitor
- [ ] Connect USB keyboard & mouse
- [ ] Connect Ethernet cable (or prepare WiFi credentials)
- [ ] Connect power supply last

### Step 3: First Boot
- [ ] Complete Ubuntu setup wizard
- [ ] Set username: `_______` and password: `_______`
- [ ] Connect to network (note IP: `_____________`)
- [ ] Update system: `sudo apt update && sudo apt upgrade`

---

## Part 2: Transfer Files (5 min)

### Find Jetson IP Address
```bash
hostname -I
```
My Jetson IP: `_____________`

### Transfer Package
**Method 1 - SCP (from your computer):**
```bash
scp jetson_deployment.zip jetson@<JETSON_IP>:~/
```

**Method 2 - USB Drive:**
- [ ] Copy `jetson_deployment.zip` to USB drive
- [ ] Insert USB into Jetson
- [ ] Copy from `/media/jetson/*/jetson_deployment.zip` to `~/`

### Extract Package
```bash
cd ~
unzip jetson_deployment.zip
cd jetson_deployment
ls  # Verify files are present
```

---

## Part 3: Software Setup (20 min)

### Run Automated Setup
```bash
cd ~/jetson_deployment
chmod +x quick_start_jetson.sh
./quick_start_jetson.sh
```

- [ ] Script completes without errors
- [ ] Virtual environment created at `~/jetson_crack_detection/venv`
- [ ] PyTorch installed with CUDA
- [ ] All dependencies installed

### Verify Setup
```bash
cd ~/jetson_crack_detection
source venv/bin/activate
python3 verify_setup.py
```

- [ ] All critical checks pass ✓
- [ ] CUDA available: True
- [ ] Model files found
- [ ] tegrastats available

---

## Part 4: Performance Mode (2 min)

### Enable Maximum Performance
```bash
# Check current mode
sudo nvpmodel -q

# Set to 15W mode (Mode 0)
sudo nvpmodel -m 0

# Lock clocks to max
sudo jetson_clocks

# Verify
sudo nvpmodel -q verbose
```

- [ ] Mode set to 0 (15W MAX)
- [ ] Clocks locked to maximum

---

## Part 5: Quick Test (5 min)

### Single Inference Test
```bash
cd ~/jetson_crack_detection
source venv/bin/activate

python3 test_single_inference.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --output test_result.png
```

- [ ] Model loads successfully
- [ ] Inference completes (record time: `____` ms)
- [ ] Output images created:
  - [ ] `test_result.png`
  - [ ] `test_result_mask.png`

---

## Part 6: Comprehensive Profiling (10 min)

### Run Profiling
```bash
cd ~/jetson_crack_detection
source venv/bin/activate

# Ensure performance mode is on
sudo nvpmodel -m 0 && sudo jetson_clocks

# Run profiling (100 iterations)
python3 profile_edge_device.py \
    --model_path BEST.pth \
    --image_path sample_image.jpg \
    --iterations 100 \
    --output my_results.json
```

Wait ~5-10 minutes for completion...

### Record Results
- [ ] Profiling completes successfully
- [ ] JSON file created: `my_results.json`

**Key Metrics** (write from console output):
```
Mean Inference Time: _______ ms
FPS: _______
RAM Usage: _______ MB
GPU Memory: _______ MB
Mean Power: _______ W
Energy per Inference: _______ mJ
```

---

## Part 7: Visualization (2 min)

### Generate Plots and Report
```bash
python3 visualize_results.py \
    --input my_results.json \
    --output summary_plot.png \
    --report
```

- [ ] `summary_plot.png` created
- [ ] `summary_plot_report.txt` created

---

## Part 8: Additional Tests (Optional)

### Test Different Power Modes

**15W Mode:**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
python3 profile_edge_device.py --iterations 100 --output results_15w.json
```
- [ ] Results: `____` ms, `____` W

**10W Mode:**
```bash
sudo nvpmodel -m 1
python3 profile_edge_device.py --iterations 100 --output results_10w.json
```
- [ ] Results: `____` ms, `____` W

**7W Mode:**
```bash
sudo nvpmodel -m 2
python3 profile_edge_device.py --iterations 100 --output results_7w.json
```
- [ ] Results: `____` ms, `____` W

### Compare Power Modes
```bash
python3 visualize_results.py \
    --input "results_15w.json,results_10w.json,results_7w.json" \
    --labels "15W,10W,7W" \
    --output power_comparison.png \
    --report
```
- [ ] Comparison plot created

---

## Final Checklist

- [ ] Hardware setup complete
- [ ] Software installed and verified
- [ ] Model loads and runs successfully
- [ ] Single inference test passed
- [ ] Comprehensive profiling completed
- [ ] Results documented
- [ ] Visualizations generated
- [ ] (Optional) Multiple power modes tested
- [ ] All result files backed up

---

## Result Files to Keep

Location: `~/jetson_crack_detection/`

- [ ] `my_results.json` - Profiling data
- [ ] `summary_plot.png` - Visualization
- [ ] `summary_plot_report.txt` - Detailed report
- [ ] `test_result.png` - Sample inference output
- [ ] `tegrastats_*.log` - Power monitoring logs

**Backup these files to another location!**

---

## Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| CUDA not available | Reinstall PyTorch: See setup guide |
| Out of memory | Reduce iterations: `--iterations 10` |
| Model won't load | Check file: `ls -lh BEST.pth` |
| tegrastats not found | Only works on actual Jetson hardware |
| Slow inference | Enable performance: `sudo nvpmodel -m 0 && sudo jetson_clocks` |

---

## Performance Expectations

**For ResUNet on Jetson Orin Nano (15W mode):**

| Metric | Expected Range | Your Result |
|--------|----------------|-------------|
| Inference Time | 30-50 ms | ______ ms |
| FPS | 20-33 | ______ |
| GPU Memory | 300-500 MB | ______ MB |
| Power | 8-12 W | ______ W |
| Energy/Inference | 300-500 mJ | ______ mJ |

✅ **If your results are in these ranges, you're good to go!**

---

## Next Steps

After completing this checklist:

1. **Document Findings**
   - Save all result files
   - Take screenshots of visualizations
   - Note any anomalies

2. **Share Results**
   - Copy result files back to development machine
   - Review with team
   - Compare with requirements

3. **Optimize (if needed)**
   - Try FP16 precision
   - Consider model quantization
   - Explore TensorRT

4. **Deploy to Production**
   - Integrate into application
   - Set up monitoring
   - Plan for maintenance

---

## Contact & Support

**Documentation:**
- `JETSON_DEPLOYMENT_COMPLETE_GUIDE.md` - Comprehensive guide
- `README_TESTING.md` - Testing framework details
- `JETSON_ORIN_NANO_COMPLETE_SETUP.md` - Setup reference

**Resources:**
- NVIDIA Jetson Forums: https://forums.developer.nvidia.com/
- PyTorch for Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

---

**Date Started**: ___/___/20___  
**Date Completed**: ___/___/20___  
**Total Time**: _______ minutes

✅ **DEPLOYMENT COMPLETE!** 🎉

