#!/usr/bin/env python3
"""
ONNX Webcam Inference Script for Jetson Nano
Optimized for Python 3 and TensorRT acceleration
"""

import argparse
import yaml
import time
import cv2
import onnxruntime as ort
import numpy as np
from utils import preprocess_rgb_uint8_to_nchw_float, postprocess_prob_to_mask, overlay_mask_on_rgb

def get_sess(cfg):
    """Create ONNX Runtime session with optimal settings for Jetson"""
    providers = cfg.get("providers", ["CPUExecutionProvider"])
    
    # Create session options for better performance
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Enable TensorRT optimizations if available
    if "TensorrtExecutionProvider" in providers:
        print("🚀 Using TensorRT acceleration for optimal performance")
    elif "CUDAExecutionProvider" in providers:
        print("🚀 Using CUDA acceleration")
    else:
        print("⚠️  Using CPU only - consider enabling GPU acceleration")
    
    try:
        session = ort.InferenceSession(cfg["model_path"], sess_options=sess_opts, providers=providers)
        print(f"✅ Model loaded successfully with providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Available providers:", ort.get_available_providers())
        return None

def main():
    parser = argparse.ArgumentParser(description="ONNX Webcam Inference on Jetson Nano")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--source", default=0, help="Camera index or RTSP URL")
    parser.add_argument("--display", action="store_true", help="Show window")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (0 for unlimited)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Extract configuration
    W, H = cfg["img_size"]
    thr = float(cfg.get("threshold", 0.5))
    alpha = float(cfg.get("alpha", 0.4))
    
    if args.verbose:
        print(f"📋 Configuration:")
        print(f"   Model: {cfg['model_path']}")
        print(f"   Image size: {W}x{H}")
        print(f"   Threshold: {thr}")
        print(f"   Alpha: {alpha}")
        print(f"   Target FPS: {args.fps}")
    
    # Create ONNX session
    sess = get_sess(cfg)
    if sess is None:
        print("❌ Failed to create ONNX session")
        return
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    if args.verbose:
        print(f"📊 Model info:")
        print(f"   Input: {input_name}, shape: {sess.get_inputs()[0].shape}")
        print(f"   Output: {output_name}, shape: {sess.get_outputs()[0].shape}")
    
    # Initialize camera
    print(f"📷 Initializing camera: {args.source}")
    cap = cv2.VideoCapture(args.source if isinstance(args.source, str) else int(args.source))
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera: {args.source}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Camera initialized successfully")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_times = []
    
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            inp = preprocess_rgb_uint8_to_nchw_float(rgb, (W, H))
            
            # Run inference
            inference_start = time.time()
            out = sess.run([output_name], {input_name: inp})[0]
            inference_time = (time.time() - inference_start) * 1000
            
            # Post-process
            mask = postprocess_prob_to_mask(out, thr)
            mask_up = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = overlay_mask_on_rgb(rgb, mask_up, alpha)
            
            # Convert back to BGR for display
            vis = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Add performance info overlay
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add text overlay
            cv2.putText(vis, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis, f"Inference: {inference_time:.1f}ms", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if args.display:
                cv2.imshow("ONNX Segmentation", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    cv2.imwrite(f"capture_{timestamp}.jpg", vis)
                    print(f"💾 Frame saved: capture_{timestamp}.jpg")
            
            # FPS limiting
            if args.fps > 0:
                target_frame_time = 1.0 / args.fps
                elapsed_frame = time.time() - current_time
                if elapsed_frame < target_frame_time:
                    time.sleep(target_frame_time - elapsed_frame)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Performance Summary:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
