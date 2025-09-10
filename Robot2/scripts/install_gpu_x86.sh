#!/usr/bin/env bash
set -e
python3 -m pip install --upgrade pip
# Replace CPU ORT with GPU ORT
pip uninstall -y onnxruntime || true
pip install onnxruntime-gpu>=1.17.0
pip install -r requirements.txt
python - <<'PY'
import onnxruntime as ort
print("Providers available:", ort.get_available_providers())
PY
