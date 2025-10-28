#!/usr/bin/env python3
"""
Complete Edge Device Profiler for Jetson Orin Nano
Measures: Memory (RAM/GPU), Inference Time, Power Consumption

Usage:
    python3 profile_edge_device.py --model_path BEST.pth --image_path sample_image.jpg
"""

import torch
import torch.nn as nn
import time
import psutil
import subprocess
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import cv2
from PIL import Image

# Import model
from model_resunet import build_resunet


class JetsonMonitor:
    """Monitor Jetson-specific metrics using tegrastats"""
    
    def __init__(self):
        self.is_jetson = self._check_jetson()
        self.tegrastats_process = None
        self.stats_file = f"tegrastats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def _check_jetson(self) -> bool:
        """Check if running on Jetson device"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return True
        except FileNotFoundError:
            return False
    
    def start_monitoring(self):
        """Start tegrastats logging"""
        if not self.is_jetson:
            print("⚠️  Not running on Jetson - power monitoring disabled")
            return
        
        # Start tegrastats in background
        self.tegrastats_process = subprocess.Popen(
            ['tegrastats', '--interval', '100'],  # 100ms interval
            stdout=open(self.stats_file, 'w'),
            stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)  # Let it initialize
        print(f"✓ Started tegrastats logging to {self.stats_file}")
    
    def stop_monitoring(self) -> Dict:
        """Stop tegrastats and parse results"""
        if not self.is_jetson or self.tegrastats_process is None:
            return {}
        
        self.tegrastats_process.terminate()
        self.tegrastats_process.wait()
        time.sleep(0.2)
        
        return self._parse_tegrastats()
    
    def _parse_tegrastats(self) -> Dict:
        """Parse tegrastats log file"""
        power_readings = []
        gpu_util = []
        cpu_util = []
        
        try:
            with open(self.stats_file, 'r') as f:
                for line in f:
                    # Parse power: look for "VDD_IN current" or "POM_5V_IN"
                    if 'POM_5V_IN' in line:
                        parts = line.split('POM_5V_IN')[1].split()[0]
                        power = int(parts.split('/')[0])  # mW
                        power_readings.append(power)
                    elif 'VDD_IN' in line:
                        parts = line.split('VDD_IN')[1].split()[0]
                        power = int(parts.split('/')[0])  # mW
                        power_readings.append(power)
                    
                    # Parse GPU utilization
                    if 'GR3D_FREQ' in line:
                        parts = line.split('GR3D_FREQ')[1].split('%')[0]
                        gpu_util.append(float(parts.strip().split()[-1]))
                    
                    # Parse CPU utilization (average of cores)
                    if 'CPU' in line:
                        cpu_parts = line.split('CPU [')[1].split(']')[0].split(',')
                        cpu_vals = [float(x.split('%')[0]) for x in cpu_parts if '%' in x]
                        if cpu_vals:
                            cpu_util.append(np.mean(cpu_vals))
        
        except Exception as e:
            print(f"⚠️  Error parsing tegrastats: {e}")
            return {}
        
        stats = {}
        if power_readings:
            stats['power_mW'] = {
                'mean': np.mean(power_readings),
                'max': np.max(power_readings),
                'min': np.min(power_readings),
                'std': np.std(power_readings)
            }
            stats['power_W'] = {k: v/1000 for k, v in stats['power_mW'].items()}
        
        if gpu_util:
            stats['gpu_utilization_%'] = {
                'mean': np.mean(gpu_util),
                'max': np.max(gpu_util),
                'min': np.min(gpu_util)
            }
        
        if cpu_util:
            stats['cpu_utilization_%'] = {
                'mean': np.mean(cpu_util),
                'max': np.max(cpu_util),
                'min': np.min(cpu_util)
            }
        
        return stats


class MemoryMonitor:
    """Monitor RAM and GPU memory usage"""
    
    @staticmethod
    def get_ram_usage() -> Dict:
        """Get current RAM usage in MB"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_MB': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_MB': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    @staticmethod
    def get_system_ram() -> Dict:
        """Get system-wide RAM info"""
        mem = psutil.virtual_memory()
        return {
            'total_MB': mem.total / 1024 / 1024,
            'available_MB': mem.available / 1024 / 1024,
            'used_MB': mem.used / 1024 / 1024,
            'percent': mem.percent
        }
    
    @staticmethod
    def get_gpu_memory() -> Dict:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_MB': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_MB': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_MB': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    
    @staticmethod
    def reset_gpu_memory_stats():
        """Reset GPU memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


class ModelProfiler:
    """Profile model inference"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_path = model_path
        
        print(f"\n{'='*60}")
        print(f"Model Profiler Initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model Path: {model_path}")
        
    def load_model(self):
        """Load model and weights"""
        print("\n[1/4] Loading Model...")
        
        try:
            # Build model
            self.model = build_resunet(img_dim=(256, 256), reg_coeff=0.0, device=self.device)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"✓ Model loaded successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        # Read image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 256x256
        img = img.resize((256, 256), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_np = np.array(img).astype(np.float32) / 256.0
        
        # Convert to tensor [1, 3, 256, 256]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def warmup(self, num_iterations: int = 10):
        """Warmup GPU"""
        print(f"\n[2/4] Warming up GPU ({num_iterations} iterations)...")
        
        dummy_input = torch.randn(1, 3, 256, 256, device=self.device)
        
        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        print("✓ Warmup complete")
    
    def benchmark_inference(self, image_path: str, num_iterations: int = 100) -> Dict:
        """Benchmark inference time"""
        print(f"\n[3/4] Benchmarking Inference ({num_iterations} iterations)...")
        
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        
        # Reset memory stats
        MemoryMonitor.reset_gpu_memory_stats()
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                # Sync before timing
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                # Inference
                output, _ = self.model(img_tensor)
                
                # Sync after inference
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{num_iterations}")
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        
        results = {
            'mean_ms': float(np.mean(inference_times)),
            'median_ms': float(np.median(inference_times)),
            'std_ms': float(np.std(inference_times)),
            'min_ms': float(np.min(inference_times)),
            'max_ms': float(np.max(inference_times)),
            'p95_ms': float(np.percentile(inference_times, 95)),
            'p99_ms': float(np.percentile(inference_times, 99)),
            'fps': float(1000 / np.mean(inference_times)),
            'num_iterations': num_iterations
        }
        
        print(f"✓ Inference benchmark complete")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        
        return results
    
    def profile_memory(self, image_path: str) -> Dict:
        """Profile memory usage during inference"""
        print(f"\n[4/4] Profiling Memory Usage...")
        
        # Get initial memory
        MemoryMonitor.reset_gpu_memory_stats()
        initial_ram = MemoryMonitor.get_ram_usage()
        initial_gpu = MemoryMonitor.get_gpu_memory()
        
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output, _ = self.model(img_tensor)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Get final memory
        final_ram = MemoryMonitor.get_ram_usage()
        final_gpu = MemoryMonitor.get_gpu_memory()
        system_ram = MemoryMonitor.get_system_ram()
        
        results = {
            'ram': {
                'process_rss_MB': final_ram['rss_MB'],
                'process_vms_MB': final_ram['vms_MB'],
                'system_total_MB': system_ram['total_MB'],
                'system_used_MB': system_ram['used_MB'],
                'system_available_MB': system_ram['available_MB'],
                'system_percent': system_ram['percent']
            }
        }
        
        if self.device == 'cuda':
            results['gpu'] = {
                'allocated_MB': final_gpu['allocated_MB'],
                'reserved_MB': final_gpu['reserved_MB'],
                'max_allocated_MB': final_gpu['max_allocated_MB'],
            }
        
        print(f"✓ Memory profiling complete")
        print(f"  RAM (Process): {final_ram['rss_MB']:.2f} MB")
        if self.device == 'cuda':
            print(f"  GPU Memory: {final_gpu['allocated_MB']:.2f} MB")
        
        return results


def run_comprehensive_profile(model_path: str, image_path: str, 
                              num_iterations: int = 100) -> Dict:
    """Run comprehensive profiling"""
    
    print(f"\n{'#'*60}")
    print(f"# JETSON ORIN NANO - COMPREHENSIVE EDGE PROFILING")
    print(f"{'#'*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize monitors
    jetson_monitor = JetsonMonitor()
    profiler = ModelProfiler(model_path)
    
    # Start power monitoring
    jetson_monitor.start_monitoring()
    time.sleep(1)
    
    # Load model
    profiler.load_model()
    
    # Warmup
    profiler.warmup(num_iterations=10)
    
    # Benchmark inference
    inference_results = profiler.benchmark_inference(image_path, num_iterations)
    
    # Profile memory
    memory_results = profiler.profile_memory(image_path)
    
    # Stop power monitoring
    time.sleep(1)
    power_results = jetson_monitor.stop_monitoring()
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': {
            'type': 'Jetson Orin Nano' if jetson_monitor.is_jetson else 'Unknown',
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
        },
        'model': {
            'path': model_path,
            'architecture': 'ResUNet with MobileViT',
            'input_size': '256x256x3',
            'output_size': '256x256x1',
        },
        'inference': inference_results,
        'memory': memory_results,
        'power': power_results,
        'test_image': image_path,
        'num_iterations': num_iterations
    }
    
    return results


def print_summary(results: Dict):
    """Print summary of results"""
    print(f"\n{'='*60}")
    print(f"PROFILING SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📊 INFERENCE PERFORMANCE")
    print(f"  Mean Time: {results['inference']['mean_ms']:.2f} ms")
    print(f"  Std Dev: {results['inference']['std_ms']:.2f} ms")
    print(f"  Min Time: {results['inference']['min_ms']:.2f} ms")
    print(f"  Max Time: {results['inference']['max_ms']:.2f} ms")
    print(f"  95th Percentile: {results['inference']['p95_ms']:.2f} ms")
    print(f"  FPS: {results['inference']['fps']:.2f}")
    
    print(f"\n💾 MEMORY USAGE")
    print(f"  RAM (Process): {results['memory']['ram']['process_rss_MB']:.2f} MB")
    print(f"  RAM (System Used): {results['memory']['ram']['system_used_MB']:.2f} MB / {results['memory']['ram']['system_total_MB']:.2f} MB ({results['memory']['ram']['system_percent']:.1f}%)")
    
    if 'gpu' in results['memory']:
        print(f"  GPU Memory (Allocated): {results['memory']['gpu']['allocated_MB']:.2f} MB")
        print(f"  GPU Memory (Reserved): {results['memory']['gpu']['reserved_MB']:.2f} MB")
        print(f"  GPU Memory (Peak): {results['memory']['gpu']['max_allocated_MB']:.2f} MB")
    
    if results['power']:
        print(f"\n⚡ POWER CONSUMPTION")
        print(f"  Mean: {results['power']['power_W']['mean']:.2f} W")
        print(f"  Max: {results['power']['power_W']['max']:.2f} W")
        print(f"  Min: {results['power']['power_W']['min']:.2f} W")
        print(f"  Std Dev: {results['power']['power_W']['std']:.2f} W")
        
        if 'gpu_utilization_%' in results['power']:
            print(f"\n🎮 GPU UTILIZATION")
            print(f"  Mean: {results['power']['gpu_utilization_%']['mean']:.1f}%")
            print(f"  Max: {results['power']['gpu_utilization_%']['max']:.1f}%")
        
        if 'cpu_utilization_%' in results['power']:
            print(f"\n🖥️  CPU UTILIZATION")
            print(f"  Mean: {results['power']['cpu_utilization_%']['mean']:.1f}%")
            print(f"  Max: {results['power']['cpu_utilization_%']['max']:.1f}%")
    
    # Calculate energy per inference
    if results['power'] and 'power_W' in results['power']:
        energy_per_inference = (results['power']['power_W']['mean'] * 
                               results['inference']['mean_ms'] / 1000)  # Joules
        print(f"\n🔋 ENERGY PER INFERENCE")
        print(f"  {energy_per_inference:.4f} Joules ({energy_per_inference*1000:.2f} mJ)")
    
    print(f"\n{'='*60}")


def save_results(results: Dict, output_path: str = None):
    """Save results to JSON file"""
    if output_path is None:
        output_path = f"profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile model on Jetson Orin Nano')
    parser.add_argument('--model_path', type=str, default='BEST.pth',
                       help='Path to model weights')
    parser.add_argument('--image_path', type=str, default='sample_image.jpg',
                       help='Path to test image')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of inference iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model_path):
        print(f"✗ Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.image_path):
        print(f"✗ Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run profiling
    try:
        results = run_comprehensive_profile(
            args.model_path, 
            args.image_path, 
            args.iterations
        )
        
        # Print summary
        print_summary(results)
        
        # Save results
        save_results(results, args.output)
        
        print(f"\n✓ Profiling complete!")
        
    except Exception as e:
        print(f"\n✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

