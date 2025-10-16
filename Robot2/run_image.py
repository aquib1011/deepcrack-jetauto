import argparse, yaml, os
import onnxruntime as ort
import numpy as np
from utils import load_rgb, preprocess_rgb_uint8_to_nchw_float, postprocess_prob_to_mask, overlay_mask_on_rgb, save_rgb

def get_sess(cfg):
    providers = cfg.get("providers", ["CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(cfg["model_path"], sess_options=sess_opts, providers=providers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", required=True, help="Path to image")
    ap.add_argument("--output", default="out.png", help="Output path (overlay or mask)")
    ap.add_argument("--save_mask", default=None, help="Optional: save raw mask to this path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W, H = cfg["img_size"]
    thr = float(cfg.get("threshold", 0.5))
    draw_overlay = bool(cfg.get("draw_overlay", True))
    alpha = float(cfg.get("alpha", 0.4))

    sess = get_sess(cfg)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    img = load_rgb(args.input)
    inp = preprocess_rgb_uint8_to_nchw_float(img, (W, H))  # (1,3,H,W)

    # run
    out = sess.run([output_name], {input_name: inp})[0]  # expected (1,1,H,W) in [0,1]
    # squeeze + make mask
    mask = postprocess_prob_to_mask(out, thr)

    if draw_overlay:
        # upsample mask to original size
        import cv2
        mask_up = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask_on_rgb(img, mask_up, alpha)
        save_rgb(args.output, overlay)
    else:
        save_rgb(args.output, np.dstack([mask, mask, mask]))  # 3-ch for viewers

    if args.save_mask:
        import cv2
        cv2.imwrite(args.save_mask, mask)

    print(f" Wrote: {args.output}")
    if args.save_mask:
        print(f"Raw mask: {args.save_mask}")

if __name__ == "__main__":
    main()
