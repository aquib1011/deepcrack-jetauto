#!/usr/bin/env python3
"""
Targeted fix for illegal instruction errors in Python packages
This script specifically addresses architecture mismatches in installed packages
"""

import subprocess
import sys
import os
import shutil

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("❌ Not running in virtual environment")
        print("   Please activate your virtual environment first:")
        print("   source venv/bin/activate")
        return False

def clean_package_cache():
    """Clean pip cache to force fresh downloads"""
    print("\n🧹 Cleaning package cache...")
    
    # Clear pip cache
    run_command("pip cache purge", "Clearing pip cache")
    
    # Remove any cached wheels
    cache_dir = os.path.expanduser("~/.cache/pip")
    if os.path.exists(cache_dir):
        run_command(f"rm -rf {cache_dir}", "Removing pip cache directory")
    
    print("✅ Package cache cleaned")

def uninstall_problematic_packages():
    """Uninstall packages that commonly cause illegal instruction errors"""
    print("\n🗑️ Uninstalling problematic packages...")
    
    problematic_packages = [
        "numpy",
        "scipy", 
        "opencv-python",
        "torch",
        "torchvision",
        "torchaudio",
        "onnxruntime",
        "onnx",
        "matplotlib",
        "pillow"
    ]
    
    for package in problematic_packages:
        run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")

def install_system_packages():
    """Install system packages first"""
    print("\n📚 Installing system packages...")
    
    system_packages = [
        "python3-numpy",
        "python3-scipy", 
        "python3-opencv",
        "python3-pil",
        "python3-matplotlib",
        "python3-dev",
        "libopenblas-dev",
        "liblapack-dev",
        "libhdf5-dev",
        "libjpeg-dev",
        "libpng-dev",
        "libtiff-dev"
    ]
    
    for package in system_packages:
        run_command(f"sudo apt-get install -y {package}", f"Installing {package}")

def install_python_packages_safely():
    """Install Python packages in a safe order for ARM64"""
    print("\n🐍 Installing Python packages safely...")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install basic packages first
    basic_packages = [
        "setuptools",
        "wheel",
        "cython"
    ]
    
    for package in basic_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Install NumPy (critical for everything else)
    print("\n🔢 Installing NumPy...")
    run_command("pip install numpy --no-cache-dir --force-reinstall", "Installing NumPy")
    
    # Test NumPy
    if not test_import("numpy", "NumPy"):
        print("❌ NumPy installation failed, trying alternative...")
        run_command("pip install numpy==1.19.5 --no-cache-dir", "Installing NumPy 1.19.5")
    
    # Install SciPy
    print("\n🔬 Installing SciPy...")
    run_command("pip install scipy --no-cache-dir", "Installing SciPy")
    
    # Install Pillow
    print("\n🖼️ Installing Pillow...")
    run_command("pip install pillow --no-cache-dir", "Installing Pillow")
    
    # Install PyTorch (CPU version for ARM64)
    print("\n🔥 Installing PyTorch...")
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir"
    if not run_command(pytorch_cmd, "Installing PyTorch"):
        print("⚠️ PyTorch installation failed, trying alternative...")
        run_command("pip install torch==1.9.0 torchvision==0.10.0 --no-cache-dir", "Installing PyTorch 1.9.0")
    
    # Install ONNX
    print("\n🧠 Installing ONNX...")
    run_command("pip install onnx --no-cache-dir", "Installing ONNX")
    
    # Install ONNX Runtime
    print("\n⚡ Installing ONNX Runtime...")
    run_command("pip install onnxruntime --no-cache-dir", "Installing ONNX Runtime")
    
    # Install OpenCV (try pip first, fallback to system)
    print("\n📷 Installing OpenCV...")
    if not run_command("pip install opencv-python --no-cache-dir", "Installing OpenCV"):
        print("⚠️ OpenCV pip installation failed, using system OpenCV")
    
    # Install other packages
    other_packages = [
        "pyyaml",
        "psutil",
        "matplotlib",
        "seaborn"
    ]
    
    for package in other_packages:
        run_command(f"pip install {package} --no-cache-dir", f"Installing {package}")
    
    # Install model-specific packages
    print("\n🧩 Installing model packages...")
    run_command("pip install einops --no-cache-dir", "Installing einops")
    
    # Try loralib (may not be available for ARM64)
    if not run_command("pip install loralib --no-cache-dir", "Installing loralib"):
        print("⚠️ loralib not available for ARM64, skipping...")
    
    # Install Jetson monitoring tools
    run_command("pip install jetson-stats --no-cache-dir", "Installing jetson-stats")

def test_import(module_name, display_name):
    """Test if a module can be imported without errors"""
    try:
        __import__(module_name)
        print(f"✅ {display_name}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {display_name}: Import failed - {e}")
        return False

def test_all_imports():
    """Test all critical imports"""
    print("\n🧪 Testing all imports...")
    
    modules_to_test = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("psutil", "psutil")
    ]
    
    success_count = 0
    for module, display in modules_to_test:
        if test_import(module, display):
            success_count += 1
    
    print(f"\n📊 Import test results: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

def main():
    """Main fix function"""
    print("🔧 Fixing Illegal Instruction Errors in Python Packages")
    print("=" * 60)
    print("This script will fix architecture mismatches in your Python packages")
    print()
    
    # Check virtual environment
    if not check_virtual_environment():
        return False
    
    # Clean package cache
    clean_package_cache()
    
    # Uninstall problematic packages
    uninstall_problematic_packages()
    
    # Install system packages
    install_system_packages()
    
    # Install Python packages safely
    install_python_packages_safely()
    
    # Test all imports
    if test_all_imports():
        print("\n🎉 All packages installed successfully!")
        print("✅ No more illegal instruction errors!")
        return True
    else:
        print("\n⚠️ Some packages may still have issues")
        print("📋 Check the output above for specific errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


