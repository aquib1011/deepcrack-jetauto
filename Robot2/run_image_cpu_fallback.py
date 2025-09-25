#!/usr/bin/env python3
"""
ONNX Image Inference Script for Jetson Nano - CPU Fallback Version
This version uses CPU-only execution to avoid compatibility issues
"""

import argparse
import yaml
import os
import sys
import onnxruntime as ort
import numpy as np
from utils import load_rgb, preprocess_rgb_uint8_to_nchw_float, postprocess_prob_to_mask, overlay_mask_on_rgb, save_rgb

def get_sess_cpu_only(cfg):
    """Create ONNX Runtime session with CPU-only execution"""
    print("⚠️  Using CPU-only execution to avoid compatibility issues")
    
    # Force CPU-only execution
    providers = ["CPUExecutionProvider"]
    
    # Create session options for better performance
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    try:
        session = ort.InferenceSession(cfg["model_path"], sess_options=sess_opts, providers=providers)
        print(f"✅ Model loaded successfully with providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Available providers:", ort.get_available_providers())
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ONNX Image Inference on Jetson Nano (CPU Fallback)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="out.png", help="Output path (overlay or mask)")
    parser.add_argument("--save_mask", default=None, help="Optional: save raw mask to this path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override providers to CPU-only
    cfg["providers"] = ["CPUExecutionProvider"]
    
    # Validate input image
    if not os.path.exists(args.input):
        print(f"❌ Input image not found: {args.input}")
        sys.exit(1)
    
    # Extract configuration
    W, H = cfg["img_size"]
    thr = float(cfg.get("threshold", 0.5))
    draw_overlay = bool(cfg.get("draw_overlay", True))
    alpha = float(cfg.get("alpha", 0.4))
    
    if args.verbose:
        print(f"📋 Configuration:")
        print(f"   Model: {cfg['model_path']}")
        print(f"   Image size: {W}x{H}")
        print(f"   Threshold: {thr}")
        print(f"   Draw overlay: {draw_overlay}")
        print(f"   Alpha: {alpha}")
        print(f"   Providers: {cfg['providers']}")
    
    # Create ONNX session
    sess = get_sess_cpu_only(cfg)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    if args.verbose:
        print(f"📊 Model info:")
        print(f"   Input: {input_name}, shape: {sess.get_inputs()[0].shape}")
        print(f"   Output: {output_name}, shape: {sess.get_outputs()[0].shape}")
    
    # Load and preprocess image
    print(f"📷 Loading image: {args.input}")
    img = load_rgb(args.input)
    print(f"   Original size: {img.shape[1]}x{img.shape[0]}")
    
    inp = preprocess_rgb_uint8_to_nchw_float(img, (W, H))
    print(f"   Preprocessed size: {inp.shape}")
    
    # Run inference
    print("🔄 Running inference (CPU-only)...")
    import time
    start_time = time.time()
    
    out = sess.run([output_name], {input_name: inp})[0]
    
    inference_time = (time.time() - start_time) * 1000
    print(f"   Inference time: {inference_time:.2f} ms")
    
    # Post-process results
    mask = postprocess_prob_to_mask(out, thr)
    print(f"   Mask shape: {mask.shape}")
    
    # Save results
    if draw_overlay:
        print("🎨 Creating overlay...")
        import cv2
        mask_up = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask_on_rgb(img, mask_up, alpha)
        save_rgb(args.output, overlay)
        print(f"✅ Overlay saved: {args.output}")
    else:
        save_rgb(args.output, np.dstack([mask, mask, mask]))
        print(f"✅ Mask saved: {args.output}")
    
    if args.save_mask:
        import cv2
        cv2.imwrite(args.save_mask, mask)
        print(f"✅ Raw mask saved: {args.save_mask}")
    
    print("🎉 Inference completed successfully!")

if __name__ == "__main__":
    main()
