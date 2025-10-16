#!/bin/bash
set -e

echo "🐍 Creating Python 3 virtual environment for Jetson Nano..."

# Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Installing Python 3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Verify Python 3 version
echo "🔍 Python 3 version:"
python3 --version

# Create virtual environment with explicit Python 3
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify we're using Python 3 in the venv
echo "🔍 Virtual environment Python version:"
python --version

# Upgrade pip
pip install --upgrade pip

echo "✅ Virtual environment created and activated"
echo "📋 To activate manually: source venv/bin/activate"
echo "📋 To verify Python version: python --version"
