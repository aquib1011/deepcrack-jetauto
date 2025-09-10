# Edge Device ONNX Profiler

This tool profiles **inference latency, memory usage, and power consumption** when running a single image through an exported **ONNX model** on edge devices.

It is designed to be **lightweight** and **portable**, working across:
- CPU-only systems
- x86 systems with NVIDIA GPUs (via `pynvml`)
- NVIDIA Jetson boards (via `tegrastats`)

---

## Features

- Loads and runs **.onnx** models with [onnxruntime](https://onnxruntime.ai/).
- Measures:
  - **Inference latency** (mean, p50, p95)
  - **CPU memory (RSS peak MB)**
  - **GPU memory (MB)** if supported
  - **Power usage (W)** via:
    - `pynvml` (x86 NVIDIA GPUs)
    - `tegrastats` (Jetson boards)
- Works with a **single Python file** (`edge_profile_onefile.py`).
- Runs on **any edge device**.

---

## Requirements

### Core (all devices)
```bash
pip install onnxruntime psutil opencv-python numpy
````

### Optional (for GPU stats and power)

* **x86 NVIDIA GPU**:

  ```bash
  pip install onnxruntime-gpu pynvml
  ```
* **NVIDIA Jetson**:

  * `onnxruntime-gpu` usually comes via JetPack or `pip`.
  * `tegrastats` is typically preinstalled in Jetson OS (`/usr/bin/tegrastats`).

---

## Quick Setup & Test

### 🖥️ On x86 with NVIDIA GPU

1. Create environment:

   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install --upgrade pip
   pip install onnxruntime-gpu pynvml opencv-python psutil numpy
   ```

2. Run profiler:

   ```bash
   python edge_profile_onefile.py \
     --model ./onnx/BEST.onnx \
     --image ./sample.jpg \
     --providers CUDAExecutionProvider CPUExecutionProvider \
     --warmup 10 --repeat 20
   ```

3. Output includes:

   * Latency
   * CPU memory
   * GPU memory usage
   * GPU power (via NVML)

---

### 🤖 On NVIDIA Jetson (Nano, Xavier, Orin, etc.)

1. Ensure dependencies:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip
   pip3 install --upgrade pip
   pip3 install onnxruntime-gpu opencv-python psutil numpy
   ```

   (JetPack includes `tegrastats` at `/usr/bin/tegrastats`.)

2. Run profiler:

   ```bash
   python3 edge_profile_onefile.py \
     --model ./onnx/BEST.onnx \
     --image ./sample.jpg \
     --warmup 10 --repeat 20
   ```

3. Output includes:

   * Latency
   * CPU memory
   * Jetson board power (from `POM_5V_IN` via tegrastats)

---

## Usage (General)

### Basic (CPU only)

```bash
python edge_profile_onefile.py --model ./onnx/BEST.onnx --image ./sample.jpg
```

### On NVIDIA GPU (CUDA)

```bash
python edge_profile_onefile.py \
  --model ./onnx/BEST.onnx --image ./sample.jpg \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup 10 --repeat 20
```

### On Jetson

```bash
python edge_profile_onefile.py \
  --model ./onnx/BEST.onnx --image ./sample.jpg \
  --warmup 10 --repeat 20
```

---

## Arguments

| Argument         | Description                                                       | Default |
| ---------------- | ----------------------------------------------------------------- | ------- |
| `--model`        | Path to `.onnx` model                                             | (req)   |
| `--image`        | Path to input image                                               | (req)   |
| `--img_size W H` | Resize image before inference (width height)                      | 256 256 |
| `--providers`    | Execution providers (CPUExecutionProvider, CUDAExecutionProvider) | CPU     |
| `--warmup`       | Number of warmup runs before profiling                            | 5       |
| `--repeat`       | Number of profiled inference runs                                 | 1       |

---

## Example Output

```text
=== ONNX Single-Image Inference Profile ===
Model             : ./onnx/BEST.onnx
Image             : ./sample.jpg
Providers         : ['CUDAExecutionProvider', 'CPUExecutionProvider']
Runs (measured)   : 20 (warmup=10)
Latency (mean)    : 12.45 ms
Latency (p50/p95) : 12.30 / 13.02 ms
CPU RSS (peak)    : 245.38 MB (process)
GPU Mem (peak)    : 512.00 MB (process)
Power (mean)      : 9.85 W [NVML]
===========================================
```

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'onnxruntime'`

* Install the correct runtime:

  * CPU only: `pip install onnxruntime`
  * NVIDIA GPU: `pip install onnxruntime-gpu`

---

### ❌ `pynvml not found` or no GPU metrics on x86

* Install it:

  ```bash
  pip install pynvml
  ```
* Ensure NVIDIA driver is installed and GPU is visible (`nvidia-smi` works).

---

### ❌ No power stats on Jetson

* Ensure `tegrastats` is available:

  ```bash
  which tegrastats
  ```
* If missing, install JetPack tools or check `/usr/bin/tegrastats`.

---

### ❌ ONNX model shape mismatch

* Adjust `--img_size W H` to match your exported model’s input dimensions.
* You can inspect your ONNX model input with:

  ```bash
  onnxruntime.tools.onnxruntime_test \
    --model ./onnx/BEST.onnx
  ```

---

### ❌ Very high variance in power readings

* Increase `--warmup` and `--repeat` (e.g., `--warmup 20 --repeat 50`) for smoother averages.
* On Jetson, other background processes may affect board-level readings.

---