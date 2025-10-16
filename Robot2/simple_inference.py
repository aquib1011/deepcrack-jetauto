#!/usr/bin/env python3
"""
Simple ONNX Inference for Jetson Nano
Minimal code to run inference on a single image
"""

import cv2
import numpy as np
import onnxruntime as ort
import sys
import os

def main():
    # Configuration
    model_path = "../onnx/BEST.onnx"
    input_image = "../sample_image.jpg"
    output_image = "result.png"
    img_size = (256, 256)
    threshold = 0.5
    
    print("🚀 Starting simple ONNX inference...")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(input_image):
        print(f"❌ Image not found: {input_image}")
        return
    
    # Load ONNX model (CPU only for compatibility)
    print("📦 Loading ONNX model...")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"📊 Input: {input_name}, Output: {output_name}")
    
    # Load and preprocess image
    print("📷 Loading image...")
    img = cv2.imread(input_image)
    if img is None:
        print(f"❌ Could not load image: {input_image}")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"   Original size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    
    # Resize image
    img_resized = cv2.resize(img_rgb, img_size)
    
    # Normalize to [0, 1] and convert to NCHW format
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_nchw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    img_batch = np.expand_dims(img_nchw, axis=0)  # Add batch dimension
    
    print(f"   Preprocessed size: {img_batch.shape}")
    
    # Run inference
    print("🔄 Running inference...")
    import time
    start_time = time.time()
    
    outputs = session.run([output_name], {input_name: img_batch})
    result = outputs[0]
    
    inference_time = (time.time() - start_time) * 1000
    print(f"   Inference time: {inference_time:.2f} ms")
    
    # Post-process result
    print("🎨 Post-processing...")
    
    # Convert probability to mask
    if result.ndim == 4:
        prob = result[0, 0]  # Remove batch and channel dimensions
    elif result.ndim == 3:
        prob = result[0]
    else:
        prob = result
    
    # Create binary mask
    mask = (prob >= threshold).astype(np.uint8) * 255
    
    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create overlay (red mask on original image)
    overlay = img_rgb.copy()
    overlay[mask_resized > 0] = [255, 0, 0]  # Red color for mask
    
    # Blend overlay with original image
    alpha = 0.4
    final_result = cv2.addWeighted(img_rgb, 1-alpha, overlay, alpha, 0)
    
    # Convert back to BGR for saving
    result_bgr = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    
    # Save result
    cv2.imwrite(output_image, result_bgr)
    print(f"✅ Result saved: {output_image}")
    
    # Also save raw mask
    cv2.imwrite("mask.png", mask_resized)
    print(f"✅ Mask saved: mask.png")
    
    print("🎉 Inference completed successfully!")

if __name__ == "__main__":
    main()
