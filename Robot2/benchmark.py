import argparse, yaml, time
import onnxruntime as ort
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W, H = cfg["img_size"]
    providers = cfg["providers"]

    sess = ort.InferenceSession(cfg["model_path"], providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    x = np.random.rand(1,3,H,W).astype(np.float32)

    # warmup
    for _ in range(10):
        sess.run([out_name], {in_name: x})

    t0 = time.time()
    for _ in range(args.repeat):
        sess.run([out_name], {in_name: x})
    dt = time.time() - t0

    print(f"Provider(s): {providers}")
    print(f"Repeat: {args.repeat}, Total: {dt:.3f}s, Avg: {dt/args.repeat*1000:.2f} ms")
    print("✅ Done")

if __name__ == "__main__":
    main()
