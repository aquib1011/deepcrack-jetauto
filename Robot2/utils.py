import cv2
import numpy as np
from typing import Tuple

def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_keep(path_or_img, size: Tuple[int,int]):
    if isinstance(path_or_img, str):
        img = load_rgb(path_or_img)
    else:
        img = path_or_img
    h, w = img.shape[:2]
    if (w, h) == (size[0], size[1]):
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def preprocess_rgb_uint8_to_nchw_float(img_rgb: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    # resize to (W,H) order in OpenCV
    img = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 256.0  # <-- matches your training code
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    img = np.expand_dims(img, axis=0)     # add batch
    return img

def postprocess_prob_to_mask(prob: np.ndarray, thr: float) -> np.ndarray:
    # prob: (1,1,H,W) or (1,H,W) float
    if prob.ndim == 4:
        prob = prob[0,0]
    elif prob.ndim == 3:
        prob = prob[0]
    mask = (prob >= thr).astype(np.uint8) * 255
    return mask

def overlay_mask_on_rgb(img_rgb: np.ndarray, mask_255: np.ndarray, alpha: float=0.4) -> np.ndarray:
    # blue overlay where mask=255
    overlay = img_rgb.copy()
    color = np.zeros_like(img_rgb)
    color[..., 2] = 255   # Red channel (BGR->RGB aware) but we are in RGB so this is red
    # To make it visually distinct on RGB, use red; change to blue by editing channel 2->0
    mask_bool = mask_255.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color[mask_bool]
    return overlay

def save_rgb(path: str, img_rgb: np.ndarray):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
