#!/usr/bin/env python3
"""
Prepare Deployment Package for Jetson Orin Nano
Creates a zip file with all necessary files for edge deployment
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import zipfile


def create_deployment_package(output_dir='jetson_deployment'):
    """Create deployment package"""
    
    print("="*60)
    print("Creating Jetson Deployment Package")
    print("="*60)
    print()
    
    # Files to include
    required_files = {
        'model_resunet.py': 'Model architecture',
        'onnx/BEST.pth': 'Trained weights',
        'sample_image.jpg': 'Test image',
    }
    
    script_files = {
        'profile_edge_device.py': 'Main profiling script',
        'verify_setup.py': 'Setup verification',
        'test_single_inference.py': 'Single inference test',
        'visualize_results.py': 'Results visualization',
        'requirements_jetson.txt': 'Python dependencies',
        'quick_start_jetson.sh': 'Quick start script',
        'JETSON_ORIN_NANO_COMPLETE_SETUP.md': 'Setup guide',
        'README_TESTING.md': 'Testing guide',
    }
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"⚠️  Output directory exists: {output_dir}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track missing files
    missing_files = []
    copied_files = []
    
    # Copy required files
    print("\n📋 Copying required files...")
    for src, description in required_files.items():
        if os.path.exists(src):
            dest = output_path / Path(src).name
            shutil.copy2(src, dest)
            size = os.path.getsize(src) / 1024 / 1024
            print(f"  ✓ {Path(src).name:<30} ({size:.2f} MB) - {description}")
            copied_files.append(Path(src).name)
        else:
            print(f"  ✗ {Path(src).name:<30} - {description} [MISSING]")
            missing_files.append(src)
    
    # Copy script files
    print("\n📜 Copying scripts and documentation...")
    for src, description in script_files.items():
        if os.path.exists(src):
            dest = output_path / src
            shutil.copy2(src, dest)
            print(f"  ✓ {src:<40} - {description}")
            copied_files.append(src)
        else:
            print(f"  ⚠️  {src:<40} - {description} [MISSING]")
    
    # Create README for package
    readme_content = f"""# Jetson Orin Nano Deployment Package

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""
    
    readme_path = output_path / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\n📄 Created package README: {readme_path}")
    
    # Create instructions file
    instructions = f"""
QUICK TRANSFER INSTRUCTIONS
{'='*60}

Method 1: SCP (Network Transfer)
---------------------------------
On your computer (in this directory):

    scp -r {output_dir} jetson@<JETSON_IP>:~/

Replace <JETSON_IP> with your Jetson's IP address (find with ifconfig on Jetson)

Method 2: USB Drive
-------------------
1. Create zip file (see below)
2. Copy to USB drive
3. Insert USB into Jetson
4. Copy from /media/<username>/<drive_name>/

Method 3: Git Repository
-------------------------
1. Commit these files to a git repository
2. On Jetson: git clone <repo_url>

Creating Zip File:
------------------
Windows:
    Right-click {output_dir} folder → Send to → Compressed (zipped) folder

Linux/Mac:
    zip -r jetson_deployment.zip {output_dir}/

{'='*60}

After Transfer:
1. cd {output_dir}
2. chmod +x quick_start_jetson.sh
3. ./quick_start_jetson.sh
"""
    
    instructions_path = output_path / 'TRANSFER_INSTRUCTIONS.txt'
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"📄 Created transfer instructions: {instructions_path}")
    
    # Create zip file
    print(f"\n📦 Creating zip archive...")
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_path.parent)
                zipf.write(file_path, arcname)
    
    zip_size = os.path.getsize(zip_path) / 1024 / 1024
    print(f"✓ Created: {zip_path} ({zip_size:.2f} MB)")
    
    # Summary
    print(f"\n{'='*60}")
    print("DEPLOYMENT PACKAGE SUMMARY")
    print(f"{'='*60}")
    print(f"\n📁 Directory: {output_dir}/")
    print(f"📦 Archive: {zip_path}")
    print(f"📊 Total files: {len(copied_files)}")
    
    if missing_files:
        print(f"\n⚠️  Missing files ({len(missing_files)}):")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPackage created but may be incomplete!")
    else:
        print("\n✅ All files included successfully!")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("\n1. Transfer package to Jetson Orin Nano:")
    print(f"   scp -r {output_dir} jetson@<JETSON_IP>:~/")
    print("   OR")
    print(f"   Copy {zip_path} to USB drive\n")
    
    print("2. On Jetson device:")
    print(f"   cd ~/{output_dir}")
    print("   chmod +x quick_start_jetson.sh")
    print("   ./quick_start_jetson.sh\n")
    
    print("3. Run profiling:")
    print("   source ~/jetson_crack_detection/venv/bin/activate")
    print("   python3 profile_edge_device.py\n")
    
    print(f"See {output_dir}/README.md for detailed instructions.")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'jetson_deployment'
    
    try:
        create_deployment_package(output_dir)
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

