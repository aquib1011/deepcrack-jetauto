#!/usr/bin/env bash
set -e
python3 -m pip install --upgrade pip
pip install -r requirements.txt
# Ensure CPU provider (default)
python - <<'PY'
import onnxruntime as ort
print("Providers available:", ort.get_available_providers())
PY
