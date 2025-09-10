# DeepCrack on JetAuto (Jetson) — End-to-End Deployment & Benchmarking

Run your trained **ResUNet (DeepCrack)** model on the **Hiwonder JetAuto** (NVIDIA Jetson), convert it to **TensorRT** for speed, and measure **latency, memory, and energy**. This README is designed for a fresh Jetson setup.

---

## 🚀 At a Glance (Overall Steps)

1. **Set up a workspace & install dependencies**
2. **Place your model code + weights**
3. **Export PyTorch → ONNX**
4. **Build a TensorRT engine (.plan)**
5. **Run a quick inference test (image/webcam)**
6. **Benchmark latency + memory** (CSV logs)
7. **Log and parse energy with tegrastats**

---

## 📁 What Each File Does

- **`model_resunet.py`**  
  Your full model definition (all classes, layers) and a helper `build_resunet()` to instantiate the model.

- **`export_onnx.py`**  
  Loads `deepcrack.pth`, wraps the model to export only the segmentation output, and exports to `deepcrack_256.onnx` (also simplifies/validates ONNX).

- **`infer_tensorrt.py`**  
  Loads the TensorRT engine (`.plan`), does preprocessing (RGB, resize 256×256, **/256.0**), runs inference on an image or webcam, postprocesses (threshold), and shows overlay.

- **`benchmark_trt_stream.py`**  
  Streams camera/video frames, measures per-frame **preprocess / inference / postprocess / total** times, logs **GPU/CPU memory**, writes `bench_stream.csv`.

- **`run_with_tegrastats.sh`**  
  Wrapper to run any command while logging power via `tegrastats` to a log file (e.g., `tegra_200ms.log`).

- **`parse_tegrastats_energy.py`**  
  Parses `tegrastats` log → prints **average power** (board/GPU/CPU) and **energy (J)** integrated across the run.

---

## 🧰 0) One-Time Setup (Jetson)

> Do this on the Jetson (SSH or local terminal).

```bash
# Make a workspace
mkdir -p ~/deepcrack_jetson && cd ~/deepcrack_jetson

# Upgrade pip and install Python deps (TensorRT/CUDA are in JetPack)
python3 -m pip install --upgrade pip
python3 -m pip install onnx onnxsim opencv-python numpy pycuda psutil

# Verify TensorRT installed with JetPack
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"

# Verify PyTorch (should be GPU-enabled). If this prints False, install NVIDIA’s CUDA wheel for your JetPack.
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
````

---

## 📄 1) Put Your Code & Weights

1. Copy your trained weights to the workspace:

```
deepcrack.pth
```

2. Create **`model_resunet.py`** and paste your model code from your notebook.
   Include all required classes:
   `SEM`, `DepthwiseSeparableConv`, `DynamicMS_EEM`, `UpsampleConcatBlock`, `DepthwisePointwiseConv`, `MobileViTBlock`, `ResUNet`.



> **Important preprocessing to match training:** RGB → resize **256×256** → **divide by 256.0** (not 255).

---

## 📜 2) Save the Helper Scripts

Create each file with the exact content below.

### `export_onnx.py`


### `infer_tensorrt.py`

### `benchmark_trt_stream.py`


### `run_with_tegrastats.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
INTERVAL_MS=${1:-200}
LOGFILE=${2:-"tegrastats.log"}
shift 2 || true
tegrastats --interval ${INTERVAL_MS} --logfile ${LOGFILE} &
TS_PID=$!
sleep 0.3
"$@"
kill ${TS_PID} || true
wait ${TS_PID} 2>/dev/null || true
echo "tegrastats saved to ${LOGFILE}"
```

```bash
chmod +x run_with_tegrastats.sh
```

### `parse_tegrastats_energy.py`


## 🔄 3) Export PyTorch → ONNX

```bash
python3 export_onnx.py
# Output: deepcrack_256.onnx
```

**Common errors & fixes**

* `torch.cuda.is_available() == False`
  → Install NVIDIA’s CUDA-enabled PyTorch wheel for your JetPack; reboot.
* `size mismatch while loading state_dict`
  → Ensure `model_resunet.py` matches the architecture that produced `deepcrack.pth`.
* `ONNX simplify failed`
  → Try `opset_version=16` in `export_onnx.py`; or skip simplify (save raw ONNX).

---

## ⚡ 4) Build TensorRT Engine

**Fast no-code path:**

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=deepcrack_256.onnx \
  --saveEngine=deepcrack_256_fp16.plan \
  --explicitBatch --fp16 \
  --shapes=input:1x3x256x256 \
  --workspace=2048
```

**Common errors & fixes**

* `trtexec: command not found`
  → Ensure JetPack installed; `ls /usr/src/tensorrt/bin`. If missing, (re)install TensorRT with JetPack.
* `Unsupported FP16`
  → Remove `--fp16`, rebuild: it will generate a FP32 engine.
* `Shape mismatch / dynamic shapes`
  → Keep `--shapes=input:1x3x256x256` (matches preprocessing). If you change input size, you must retrain/export and adjust code.

---

## 👀 5) Quick Inference Test

### Single Image

```bash
python3 infer_tensorrt.py --image sample.jpg --save overlay.jpg
```

### Webcam

```bash
python3 infer_tensorrt.py --video cam
```

**Common errors & fixes**

* `Failed to open camera/video`
  → Ensure the correct device is `/dev/video0`. Try `v4l2-ctl --list-devices` and `--video /dev/video1` if needed.
  → For CSI cameras, ensure drivers/nodes are running or use the vendor’s camera app to verify.
* `Binding names not found: 'input'/'mask'`
  → The engine must use those names. Rebuild ONNX from `export_onnx.py` (it sets names correctly).
* Wrong colors / poor masks → Verify preprocessing (RGB, **/256.0**). If training used `/255.0`, switch in both train & infer.

---

## 📊 6) Benchmark Latency & Memory

(Optional, fix clocks for repeatability)

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Run streaming benchmark (500 frames) **while logging power**:

```bash
./run_with_tegrastats.sh 200 tegra_200ms.log \
  python3 benchmark_trt_stream.py --video cam --frames 500 --display
```

Output:

* `bench_stream.csv` — per-frame `t_pre_ms, t_inf_ms, t_post_ms, t_total_ms`, GPU/CPU memory
* Terminal — average total latency

**Common errors & fixes**

* `tegrastats: not found`
  → On most Jetsons it’s preinstalled; if not, install via JetPack or use `sudo /usr/bin/tegrastats`.
* High variance in latency
  → Close other apps; fix power/clocks; increase camera exposure stability; use `--frames` ≥ 500.

---

## 🔋 7) Parse Energy Usage

```bash
python3 parse_tegrastats_energy.py tegra_200ms.log --interval_ms 200
```

You’ll see:

```
Samples       : N
Avg Power (mW): IN=xxxx GPU=xxxx CPU=xxxx
Energy (J)    : IN=xx.xx GPU=xx.xx CPU=xx.xx
```

**Common errors & fixes**

* `No POM_* lines found`
  → `tegrastats` output format may differ. Open the log, copy a sample line, and adjust the regex in `parse_tegrastats_energy.py` accordingly.
* Sampling too slow/fast
  → Change interval (e.g., `100` ms) in `run_with_tegrastats.sh` and pass the same to parser.

---



## ✅ Recap: Minimal Command Flow

```bash
cd ~/deepcrack_jetson

# 1) Export ONNX
python3 export_onnx.py

# 2) Build TensorRT engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=deepcrack_256.onnx \
  --saveEngine=deepcrack_256_fp16.plan \
  --explicitBatch --fp16 \
  --shapes=input:1x3x256x256 --workspace=2048

# 3) Test inference
python3 infer_tensorrt.py --image sample.jpg --save overlay.jpg
python3 infer_tensorrt.py --video cam

# 4) Benchmark + energy logging
sudo nvpmodel -m 0 && sudo jetson_clocks
./run_with_tegrastats.sh 200 tegra_200ms.log \
  python3 benchmark_trt_stream.py --video cam --frames 1000 --display
python3 parse_tegrastats_energy.py tegra_200ms.log --interval_ms 200
```
