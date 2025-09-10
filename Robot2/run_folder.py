import argparse, yaml, glob, os
import onnxruntime as ort
import numpy as np
import cv2
from utils import load_rgb, preprocess_rgb_uint8_to_nchw_float, postprocess_prob_to_mask, overlay_mask_on_rgb, save_rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    W, H = cfg["img_size"]
    thr = float(cfg.get("threshold", 0.5))
    draw_overlay = bool(cfg.get("draw_overlay", True))
    alpha = float(cfg.get("alpha", 0.4))

    sess = ort.InferenceSession(cfg["model_path"], providers=cfg["providers"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    assert files, f"No files matched in {args.input_dir} with pattern {args.glob}"

    for f in files:
        img = load_rgb(f)
        inp = preprocess_rgb_uint8_to_nchw_float(img, (W, H))
        out = sess.run([output_name], {input_name: inp})[0]
        mask = postprocess_prob_to_mask(out, thr)
        mask_up = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if draw_overlay:
            res = overlay_mask_on_rgb(img, mask_up, alpha)
        else:
            res = np.dstack([mask_up, mask_up, mask_up])
        save_path = os.path.join(args.out_dir, os.path.basename(f).rsplit(".",1)[0] + "_out.png")
        save_rgb(save_path, res)
        print("→", save_path)

    print("✅ Done")

if __name__ == "__main__":
    main()
