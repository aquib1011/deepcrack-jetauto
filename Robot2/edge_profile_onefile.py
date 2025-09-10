#!/usr/bin/env python3
"""
Edge ONNX single-image profiler (one-file)
- Loads a .onnx model and one image
- Runs onnxruntime inference once (with optional warmup)
- Reports latency, CPU RSS peak, GPU memory (if available), and power (if available)

Optional backends:
- x86 NVIDIA GPU: NVML via pynvml (GPU mem + power)
- NVIDIA Jetson: tegrastats parsing (board power via POM_5V_IN, optional GPU mem if exposed)

Dependencies:
  - Required: onnxruntime, numpy, opencv-python, psutil
  - Optional: pynvml (for NVIDIA dGPU power/mem on x86), tegrastats (Jetson)
"""

import argparse, os, time, threading, shutil, subprocess, re, sys
from statistics import mean
import numpy as np
import psutil

# ---- soft deps ----
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (opencv-python) is required.", file=sys.stderr)
    raise
try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime is required.", file=sys.stderr)
    raise

try:
    import pynvml  # optional
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# -------------------- image I/O & pre/post --------------------
def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def preprocess(img_rgb: np.ndarray, size_wh) -> np.ndarray:
    """Resize to (W,H), convert to float32, divide by 256.0, HWC->CHW, add batch."""
    W, H = size_wh
    img = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 256.0  # matches your training normalization
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# -------------------- samplers --------------------
class SamplerBase:
    def start(self): ...
    def stop(self): ...
    def stats(self) -> dict: return {}

class RSSPeakSampler(SamplerBase):
    """Process RSS peak (MB) during measurement window."""
    def __init__(self, pid: int, interval: float = 0.005):
        self.pid = pid
        self.interval = interval
        self._running = False
        self._peak = 0
        self._th = None

    def _run(self):
        p = psutil.Process(self.pid)
        while self._running:
            try:
                rss = p.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=1.0)

    def stats(self) -> dict:
        return {"cpu_rss_peak_mb": self._peak / (1024**2)}


class NVMLSampler(SamplerBase):
    """x86 NVIDIA: sample total GPU power (W) and per-process GPU memory (MB)."""
    def __init__(self, pid: int, interval: float = 0.02):
        self.pid = pid
        self.interval = interval
        self._running = False
        self._th = None
        self._mw = []   # milliwatts samples (sum across GPUs)
        self._gmem = [] # MB peak for this pid
        self._handles = []

    def _init(self):
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))

    def _sample_once(self):
        mw_total = 0
        gmem = 0.0
        for h in self._handles:
            try:
                mw_total += pynvml.nvmlDeviceGetPowerUsage(h)  # mW
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h)
                except Exception:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                for p in procs:
                    if getattr(p, "pid", None) == self.pid:
                        gmem += getattr(p, "usedGpuMemory", getattr(p, "memoryUsed", 0)) / (1024**2)
            except Exception:
                pass
        self._mw.append(mw_total)
        if gmem > 0:
            self._gmem.append(gmem)

    def _run(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval)

    def start(self):
        try:
            self._init()
        except Exception:
            return
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=1.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def stats(self) -> dict:
        if self._mw:
            watts_mean = mean(self._mw) / 1000.0
        else:
            watts_mean = None
        gmem_peak = max(self._gmem) if self._gmem else None
        return {
            "power_w_mean": watts_mean,
            "gpu_mem_peak_mb": gmem_peak,
            "power_backend": "NVML"
        }


class TegrastatsSampler(SamplerBase):
    """Jetson: parse tegrastats lines for board power (POM_5V_IN)."""
    def __init__(self, interval_ms: int = 20):
        self.interval_ms = interval_ms
        self._proc = None
        self._running = False
        self._th = None
        self._mw = []
        self._rx_pwr = re.compile(r"POM_5V_IN[:=]\s*([0-9]+)\s*mW", re.IGNORECASE)

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
        except FileNotFoundError:
            return
        for line in self._proc.stdout:
            if not self._running:
                break
            m = self._rx_pwr.search(line)
            if m:
                try:
                    self._mw.append(int(m.group(1)))
                except Exception:
                    pass

    def start(self):
        if shutil.which("tegrastats") is None:
            return
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._th:
            self._th.join(timeout=1.0)

    def stats(self) -> dict:
        if self._mw:
            watts_mean = mean(self._mw) / 1000.0
        else:
            watts_mean = None
        return {
            "power_w_mean": watts_mean,
            "gpu_mem_peak_mb": None,
            "power_backend": "tegrastats" if shutil.which("tegrastats") else None
        }


def pick_power_sampler() -> SamplerBase:
    # Prefer NVML when available
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return NVMLSampler(os.getpid())
        except Exception:
            pass
    # Jetson tegrastats fallback
    if shutil.which("tegrastats") is not None:
        return TegrastatsSampler()
    # none
    return SamplerBase()


# -------------------- main --------------------
def build_session(onnx_path: str, providers: list):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name

def main():
    ap = argparse.ArgumentParser(description="Single-file ONNX profiler for edge devices")
    ap.add_argument("--model", required=True, help="Path to .onnx model")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--img_size", type=int, nargs=2, default=[256, 256], metavar=("W","H"), help="Model input size")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                    help='ONNX Runtime providers, e.g. "CUDAExecutionProvider CPUExecutionProvider"')
    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs before measurement")
    ap.add_argument("--repeat", type=int, default=1, help="Measured runs")
    args = ap.parse_args()

    # Build session
    sess, in_name, out_name = build_session(args.model, args.providers)

    # Load & preprocess
    img = load_rgb(args.image)
    x = preprocess(img, args.img_size)

    # Warmup
    for _ in range(max(0, args.warmup)):
        sess.run([out_name], {in_name: x})

    # Samplers
    rss = RSSPeakSampler(os.getpid(), interval=0.005)
    pwr = pick_power_sampler()

    # Measure latency across repeat runs
    times_ms = []
    rss.start(); pwr.start()
    try:
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            _ = sess.run([out_name], {in_name: x})
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    finally:
        pwr.stop(); rss.stop()

    # Stats
    lat_mean = float(np.mean(times_ms))
    lat_p50  = float(np.median(times_ms))
    lat_p95  = float(np.percentile(times_ms, 95)) if len(times_ms) > 1 else lat_mean
    rss_now_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    s = {
        "providers": args.providers,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "latency_ms_mean": round(lat_mean, 3),
        "latency_ms_p50": round(lat_p50, 3),
        "latency_ms_p95": round(lat_p95, 3),
        **rss.stats(),
        **pwr.stats(),
        "cpu_rss_now_mb": round(rss_now_mb, 2),
        "model": args.model,
        "image": args.image,
    }

    # Pretty print
    print("\n=== ONNX Single-Image Inference Profile ===")
    print(f"Model             : {s['model']}")
    print(f"Image             : {s['image']}")
    print(f"Providers         : {s['providers']}")
    print(f"Runs (measured)   : {s['repeat']} (warmup={s['warmup']})")
    print(f"Latency (mean)    : {s['latency_ms_mean']:.2f} ms")
    print(f"Latency (p50/p95) : {s['latency_ms_p50']:.2f} / {s['latency_ms_p95']:.2f} ms")
    print(f"CPU RSS (peak)    : {s.get('cpu_rss_peak_mb', 0):.2f} MB (process)")
    if s.get("gpu_mem_peak_mb") is not None:
        print(f"GPU Mem (peak)    : {s['gpu_mem_peak_mb']:.2f} MB (process)")
    if s.get("power_w_mean") is not None:
        print(f"Power (mean)      : {s['power_w_mean']:.2f} W [{s.get('power_backend')}]")
    else:
        print("Power             : (no sampler available: install pynvml or use Jetson tegrastats)")
    print("===========================================\n")


if __name__ == "__main__":
    # numpy import deferred to here for minimal startup time on tiny devices
    import numpy as np  # noqa: E402
    main()
