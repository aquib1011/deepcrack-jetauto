#!/usr/bin/env python3
"""
Simple Single Inference Test
Quick test to verify model works on a single image
"""

import torch
import numpy as np
from PIL import Image
import argparse
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Import model
from model_resunet import build_resunet


def preprocess_image(image_path):
    """Preprocess image for inference"""
    # Read image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize to 256x256
    img_resized = img.resize((256, 256), Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_np = np.array(img_resized).astype(np.float32) / 256.0
    
    # Convert to tensor [1, 3, 256, 256]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img_resized, original_size


def postprocess_output(output_tensor):
    """Convert output tensor to displayable image"""
    # Remove batch dimension and convert to numpy
    output_np = output_tensor.squeeze().cpu().numpy()
    
    # Convert to 0-255 range
    output_np = (output_np * 255).astype(np.uint8)
    
    return output_np


def visualize_results(original_img, output_img, save_path='result.png'):
    """Visualize input and output side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted crack mask
    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title('Predicted Crack Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    original_np = np.array(original_img)
    overlay = original_np.copy()
    
    # Create red overlay for cracks
    mask_binary = output_img > 127
    overlay[mask_binary] = [255, 0, 0]  # Red color for cracks
    
    # Blend with original
    blended = cv2.addWeighted(original_np, 0.7, overlay, 0.3, 0)
    
    axes[2].imshow(blended)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test single image inference')
    parser.add_argument('--model_path', type=str, default='BEST.pth',
                       help='Path to model weights')
    parser.add_argument('--image_path', type=str, default='sample_image.jpg',
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='inference_result.png',
                       help='Output visualization path')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model_path):
        print(f"✗ Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.image_path):
        print(f"✗ Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and device == 'cpu':
        print("⚠️  CUDA requested but not available, using CPU")
    
    print(f"\n{'='*60}")
    print(f"Single Image Inference Test")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"{'='*60}\n")
    
    # Load model
    print("[1/4] Loading model...")
    try:
        model = build_resunet(img_dim=(256, 256), reg_coeff=0.0, device=device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Preprocess image
    print("\n[2/4] Preprocessing image...")
    try:
        img_tensor, img_resized, original_size = preprocess_image(args.image_path)
        img_tensor = img_tensor.to(device)
        print(f"✓ Image preprocessed: {original_size} -> (256, 256)")
    except Exception as e:
        print(f"✗ Error preprocessing image: {e}")
        sys.exit(1)
    
    # Run inference
    print("\n[3/4] Running inference...")
    try:
        import time
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(img_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            # Timed inference
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output, _ = model(img_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
        
        print(f"✓ Inference completed in {inference_time_ms:.2f} ms")
        
        # Analyze output
        output_np = output.squeeze().cpu().numpy()
        crack_pixels = (output_np > 0.5).sum()
        total_pixels = output_np.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        print(f"  Crack coverage: {crack_percentage:.2f}% ({crack_pixels}/{total_pixels} pixels)")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Visualize results
    print("\n[4/4] Creating visualization...")
    try:
        output_img = postprocess_output(output)
        visualize_results(img_resized, output_img, args.output)
        
        # Also save mask separately
        mask_path = args.output.replace('.png', '_mask.png')
        cv2.imwrite(mask_path, output_img)
        print(f"✓ Mask saved to: {mask_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create visualization: {e}")
    
    if args.show:
        plt.show()
    
    print(f"\n{'='*60}")
    print("✓ Test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

