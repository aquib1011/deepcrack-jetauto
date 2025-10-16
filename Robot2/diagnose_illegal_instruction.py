#!/usr/bin/env python3
"""
Diagnostic script for "Illegal Instruction (core dumped)" error on Jetson Nano
This script helps identify the specific cause of the error
"""

import sys
import os
import subprocess
import platform
import struct

def check_system_info():
    """Check basic system information"""
    print("🔍 System Information")
    print("=" * 40)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[0]}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment")
    
    print()

def check_cpu_features():
    """Check CPU features and capabilities"""
    print("🔧 CPU Features Check")
    print("=" * 40)
    
    try:
        # Read CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        # Extract key information
        lines = cpuinfo.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['processor', 'model name', 'cpu implementer', 'cpu variant', 'cpu part', 'cpu revision', 'features', 'flags']):
                print(f"  {line.strip()}")
        
        # Check for ARM64 specific features
        if 'aarch64' in platform.machine().lower():
            print("\n🔍 ARM64 Specific Checks:")
            
            # Check for NEON support
            if 'asimd' in cpuinfo.lower():
                print("  ✅ NEON (Advanced SIMD) support detected")
            else:
                print("  ❌ NEON support not detected")
            
            # Check for FPU support
            if 'fp' in cpuinfo.lower():
                print("  ✅ FPU support detected")
            else:
                print("  ❌ FPU support not detected")
            
            # Check for AES support
            if 'aes' in cpuinfo.lower():
                print("  ✅ AES support detected")
            else:
                print("  ❌ AES support not detected")
        
    except Exception as e:
        print(f"❌ Error reading CPU info: {e}")
    
    print()

def check_python_packages():
    """Check Python packages and their compatibility"""
    print("📦 Python Package Check")
    print("=" * 40)
    
    packages_to_check = [
        'torch',
        'torchvision', 
        'onnxruntime',
        'onnx',
        'numpy',
        'cv2',
        'scipy',
        'einops',
        'loralib'
    ]
    
    for package in packages_to_check:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ {package}: {cv2.__version__}")
            elif package == 'onnxruntime':
                import onnxruntime as ort
                print(f"✅ {package}: {ort.__version__}")
                print(f"   Providers: {ort.get_available_providers()}")
            elif package == 'torch':
                import torch
                print(f"✅ {package}: {torch.__version__}")
                print(f"   CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA version: {torch.version.cuda}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: Not installed - {e}")
        except Exception as e:
            print(f"⚠️ {package}: Error - {e}")
    
    print()

def check_system_libraries():
    """Check system libraries"""
    print("📚 System Libraries Check")
    print("=" * 40)
    
    libraries_to_check = [
        'libopenblas.so.0',
        'liblapack.so.3',
        'libhdf5.so.103',
        'libjpeg.so.8',
        'libpng16.so.16',
        'libtiff.so.5',
        'libavcodec.so.58',
        'libavformat.so.58',
        'libswscale.so.5',
        'libv4l2.so.0',
        'libgtk-3.so.0',
        'libatlas.so.3'
    ]
    
    for lib in libraries_to_check:
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"✅ {lib}: Found")
            else:
                print(f"❌ {lib}: Not found")
        except Exception as e:
            print(f"⚠️ {lib}: Error checking - {e}")
    
    print()

def test_basic_operations():
    """Test basic operations that might cause illegal instruction"""
    print("🧪 Basic Operations Test")
    print("=" * 40)
    
    try:
        # Test NumPy operations
        import numpy as np
        print("Testing NumPy operations...")
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([6, 7, 8, 9, 10])
        c = np.dot(a, b)
        print(f"✅ NumPy dot product: {c}")
        
        # Test matrix operations
        matrix = np.random.rand(100, 100)
        result = np.linalg.inv(matrix)
        print("✅ NumPy matrix inversion: Success")
        
    except Exception as e:
        print(f"❌ NumPy operations failed: {e}")
    
    try:
        # Test OpenCV operations
        import cv2
        print("Testing OpenCV operations...")
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV color conversion: Success")
        
    except Exception as e:
        print(f"❌ OpenCV operations failed: {e}")
    
    try:
        # Test PyTorch operations
        import torch
        print("Testing PyTorch operations...")
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print("✅ PyTorch matrix multiplication: Success")
        
    except Exception as e:
        print(f"❌ PyTorch operations failed: {e}")
    
    print()

def check_environment_variables():
    """Check relevant environment variables"""
    print("🌍 Environment Variables Check")
    print("=" * 40)
    
    env_vars = [
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'CUDA_HOME',
        'CUDA_PATH',
        'TENSORRT_ROOT',
        'OPENCV_DIR',
        'BLAS',
        'LAPACK'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    print()

def main():
    """Main diagnostic function"""
    print("🔍 Illegal Instruction Error Diagnostic Tool")
    print("=" * 60)
    print("This tool helps identify the cause of 'Illegal Instruction (core dumped)' errors")
    print("on ARM64 systems like the Jetson Nano.")
    print()
    
    check_system_info()
    check_cpu_features()
    check_python_packages()
    check_system_libraries()
    check_environment_variables()
    test_basic_operations()
    
    print("🎯 Diagnostic Summary")
    print("=" * 40)
    print("If you're still getting 'Illegal Instruction' errors, try:")
    print("1. Run: ./fix_illegal_instruction.sh")
    print("2. Check if you're using the correct architecture (aarch64)")
    print("3. Ensure all packages are compiled for ARM64")
    print("4. Check for missing system libraries")
    print("5. Verify your Python environment is clean")
    print()
    print("For more help, check the DEPLOYMENT_GUIDE.md file.")

if __name__ == "__main__":
    main()
