#!/usr/bin/env bash
set -e
python3 -m pip install --upgrade pip
# Jetson-friendly ORT GPU wheel; change version if your JetPack needs a different one.
# If this fails on your JetPack version, comment it out and fall back to CPU.
pip uninstall -y onnxruntime onnxruntime-gpu || true
pip install --extra-index-url https://download.pytorch.org/whl/cu118 onnxruntime-gpu==1.17.0 || \
pip install onnxruntime  # fallback CPU
pip install opencv-python numpy PyYAML
python - <<'PY'
import onnxruntime as ort
print("Providers available:", ort.get_available_providers())
PY
