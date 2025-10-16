#!/usr/bin/env python3
"""
Simple script to check Python version and environment
"""

import sys
import os

print("🐍 Python Environment Check")
print("=" * 40)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[0]}")
print(f"Virtual environment: {'VIRTUAL_ENV' in os.environ}")

if 'VIRTUAL_ENV' in os.environ:
    print(f"VENV path: {os.environ['VIRTUAL_ENV']}")

print("=" * 40)

# Check if this is Python 3
if sys.version_info[0] == 3:
    print("✅ Python 3 detected - Good to go!")
else:
    print("❌ Python 2 detected - Need to use Python 3")
    print("   Try: python3 check_python.py")
    sys.exit(1)

# Check if we can import required packages
print("\n📦 Package Check:")
try:
    import onnxruntime
    print(f"✅ ONNX Runtime: {onnxruntime.__version__}")
except ImportError:
    print("❌ ONNX Runtime: Not installed")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV: Not installed")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except ImportError:
    print("❌ NumPy: Not installed")

print("\n🎉 Environment check complete!")
