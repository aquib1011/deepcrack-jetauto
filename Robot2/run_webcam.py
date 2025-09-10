import argparse, yaml, time
import cv2, onnxruntime as ort
import numpy as np
from utils import preprocess_rgb_uint8_to_nchw_float, postprocess_prob_to_mask, overlay_mask_on_rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=0, help="camera index or RTSP URL")
    ap.add_argument("--display", action="store_true", help="show window")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W, H = cfg["img_size"]
    thr = float(cfg.get("threshold", 0.5))
    alpha = float(cfg.get("alpha", 0.4))

    sess = ort.InferenceSession(cfg["model_path"], providers=cfg["providers"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cap = cv2.VideoCapture(args.source if isinstance(args.source, str) else int(args.source))
    assert cap.isOpened(), f"Cannot open source: {args.source}"

    t0, n = time.time(), 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = preprocess_rgb_uint8_to_nchw_float(rgb, (W, H))
        out = sess.run([out_name], {in_name: inp})[0]
        mask = postprocess_prob_to_mask(out, thr)
        mask_up = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        over = overlay_mask_on_rgb(rgb, mask_up, alpha)
        vis = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
        n += 1
        if args.display:
            cv2.imshow("seg", vis)
            if cv2.waitKey(1) & 0xFF == 27: break  # ESC exits

    fps = n / (time.time() - t0 + 1e-6)
    print(f"✅ Frames: {n}, ~{fps:.2f} FPS")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
