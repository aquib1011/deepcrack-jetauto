#!/usr/bin/env python3
"""
Basic Python test to verify the interpreter is working
This script tests the most basic Python operations
"""

print("🧪 Basic Python Test")
print("=" * 30)

# Test 1: Basic print
print("✅ Print statement works")

# Test 2: Basic arithmetic
try:
    result = 2 + 2
    print(f"✅ Basic arithmetic: 2 + 2 = {result}")
except Exception as e:
    print(f"❌ Arithmetic failed: {e}")

# Test 3: Basic string operations
try:
    text = "Hello, World!"
    print(f"✅ String operations: {text}")
except Exception as e:
    print(f"❌ String operations failed: {e}")

# Test 4: Basic list operations
try:
    my_list = [1, 2, 3, 4, 5]
    print(f"✅ List operations: {my_list}")
except Exception as e:
    print(f"❌ List operations failed: {e}")

# Test 5: Basic function
try:
    def add_numbers(a, b):
        return a + b
    
    result = add_numbers(5, 3)
    print(f"✅ Function definition: add_numbers(5, 3) = {result}")
except Exception as e:
    print(f"❌ Function definition failed: {e}")

# Test 6: Basic import (built-in modules only)
try:
    import os
    import sys
    print(f"✅ Built-in imports: os, sys")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {sys.platform}")
except Exception as e:
    print(f"❌ Built-in imports failed: {e}")

print("\n🎉 Basic Python test completed!")
print("If you see this message, Python is working correctly.")

