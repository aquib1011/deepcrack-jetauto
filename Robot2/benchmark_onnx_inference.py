#!/usr/bin/env python3
"""
Comprehensive ONNX inference benchmarking script for Hiwonder JetAuto
Measures memory usage, inference time, and power consumption
"""

import time
import psutil
import numpy as np
import cv2
import onnxruntime as ort
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import queue
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class JetsonMonitor:
    """Monitor Jetson-specific metrics using tegrastats"""
    
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start tegrastats monitoring in background"""
        self.monitoring = True
        self.metrics = []
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        print("📊 Started Jetson monitoring...")
    
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics"""
        if self.monitoring:
            self.stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            self.monitoring = False
        return self.metrics
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        try:
            # Start tegrastats process
            process = subprocess.Popen(
                ['tegrastats', '--interval', str(int(self.sample_interval * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            while not self.stop_event.is_set():
                line = process.stdout.readline()
                if line:
                    metrics = self._parse_tegrastats(line.strip())
                    if metrics:
                        self.metrics.append(metrics)
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
        finally:
            if 'process' in locals():
                process.terminate()
    
    def _parse_tegrastats(self, line: str) -> Dict:
        """Parse tegrastats output line"""
        try:
            # Example line: "RAM 1234/4567MB (lfb 1234x4MB) SWAP 1234/4567MB (cached 1234MB) CPU [12%] EMC_FREQ 12% GR3D_FREQ 12%"
            metrics = {}
            
            # Parse RAM usage
            if 'RAM' in line:
                ram_part = line.split('RAM')[1].split('MB')[0].strip()
                if '/' in ram_part:
                    used, total = ram_part.split('/')
                    metrics['ram_used_mb'] = int(used)
                    metrics['ram_total_mb'] = int(total)
                    metrics['ram_usage_percent'] = (int(used) / int(total)) * 100
            
            # Parse CPU usage
            if 'CPU' in line and '[' in line:
                cpu_part = line.split('CPU')[1].split(']')[0].strip('[]')
                if '%' in cpu_part:
                    metrics['cpu_usage_percent'] = float(cpu_part.replace('%', ''))
            
            # Parse GPU frequency
            if 'GR3D_FREQ' in line:
                gpu_part = line.split('GR3D_FREQ')[1].split('%')[0].strip()
                metrics['gpu_freq_percent'] = float(gpu_part)
            
            # Parse EMC frequency
            if 'EMC_FREQ' in line:
                emc_part = line.split('EMC_FREQ')[1].split('%')[0].strip()
                metrics['emc_freq_percent'] = float(emc_part)
            
            # Add timestamp
            metrics['timestamp'] = time.time()
            
            return metrics
            
        except Exception as e:
            return None

class ONNXBenchmark:
    """Comprehensive ONNX inference benchmarking"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.img_size = (256, 256)
        self.threshold = 0.5
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config()
        
        # Initialize ONNX session
        self._init_session()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.img_size = tuple(config.get('img_size', [256, 256]))
            self.threshold = config.get('threshold', 0.5)
    
    def _init_session(self):
        """Initialize ONNX Runtime session with optimal providers"""
        print("🔧 Initializing ONNX Runtime session...")
        
        # Provider priority for Jetson Nano
        providers = [
            'TensorrtExecutionProvider',  # Best performance on Jetson
            'CUDAExecutionProvider',      # GPU fallback
            'CPUExecutionProvider'        # CPU fallback
        ]
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"✅ ONNX session initialized successfully!")
            print(f"   Providers: {self.session.get_providers()}")
            print(f"   Input: {self.input_name}")
            print(f"   Output: {self.output_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize ONNX session: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0
        
        # Convert HWC to CHW and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run single inference"""
        return self.session.run([self.output_name], {self.input_name: input_data})[0]
    
    def benchmark_single_inference(self, image_path: str, warmup_runs: int = 5) -> Dict:
        """Benchmark single image inference with detailed metrics"""
        print(f"🧪 Benchmarking single inference: {image_path}")
        
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        
        # Warmup runs
        print(f"🔥 Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = self.run_inference(input_data)
        
        # Initialize monitoring
        monitor = JetsonMonitor(sample_interval=0.05)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Measure inference time
        start_time = time.perf_counter()
        output = self.run_inference(input_data)
        end_time = time.perf_counter()
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        inference_time = end_time - start_time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Process Jetson-specific metrics
        jetson_metrics = self._process_jetson_metrics(metrics)
        
        result = {
            'image_path': image_path,
            'inference_time_ms': inference_time * 1000,
            'fps': 1.0 / inference_time,
            'cpu_usage_percent': cpu_percent,
            'memory_usage_mb': memory_info.used / (1024 * 1024),
            'memory_usage_percent': memory_info.percent,
            'input_shape': input_data.shape,
            'output_shape': output.shape,
            'output_range': [float(output.min()), float(output.max())],
            'jetson_metrics': jetson_metrics
        }
        
        print(f"✅ Single inference completed:")
        print(f"   Time: {inference_time*1000:.2f} ms")
        print(f"   FPS: {1.0/inference_time:.2f}")
        print(f"   CPU: {cpu_percent:.1f}%")
        print(f"   Memory: {memory_info.percent:.1f}%")
        
        return result
    
    def benchmark_batch_inference(self, image_paths: List[str], batch_size: int = 1) -> Dict:
        """Benchmark batch inference"""
        print(f"🧪 Benchmarking batch inference: {len(image_paths)} images, batch_size={batch_size}")
        
        # Preprocess all images
        input_batch = []
        for path in image_paths:
            input_data = self.preprocess_image(path)
            input_batch.append(input_data)
        
        # Initialize monitoring
        monitor = JetsonMonitor(sample_interval=0.1)
        monitor.start_monitoring()
        
        # Run batch inference
        start_time = time.perf_counter()
        outputs = []
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            if len(batch) == 1:
                output = self.run_inference(batch[0])
                outputs.append(output)
            else:
                # For true batch processing, you'd need to modify the model
                for item in batch:
                    output = self.run_inference(item)
                    outputs.append(output)
        
        end_time = time.perf_counter()
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(image_paths)
        
        result = {
            'total_images': len(image_paths),
            'batch_size': batch_size,
            'total_time_ms': total_time * 1000,
            'avg_time_per_image_ms': avg_time_per_image * 1000,
            'fps': len(image_paths) / total_time,
            'jetson_metrics': self._process_jetson_metrics(metrics)
        }
        
        print(f"✅ Batch inference completed:")
        print(f"   Total time: {total_time*1000:.2f} ms")
        print(f"   Avg per image: {avg_time_per_image*1000:.2f} ms")
        print(f"   FPS: {len(image_paths)/total_time:.2f}")
        
        return result
    
    def benchmark_continuous_inference(self, image_path: str, duration_seconds: int = 30) -> Dict:
        """Benchmark continuous inference for power/thermal analysis"""
        print(f"🧪 Benchmarking continuous inference for {duration_seconds}s...")
        
        # Preprocess image once
        input_data = self.preprocess_image(image_path)
        
        # Initialize monitoring
        monitor = JetsonMonitor(sample_interval=0.1)
        monitor.start_monitoring()
        
        # Run continuous inference
        start_time = time.time()
        inference_count = 0
        inference_times = []
        
        while time.time() - start_time < duration_seconds:
            iter_start = time.perf_counter()
            _ = self.run_inference(input_data)
            iter_end = time.perf_counter()
            
            inference_times.append(iter_end - iter_start)
            inference_count += 1
        
        end_time = time.time()
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        actual_duration = end_time - start_time
        avg_inference_time = np.mean(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        std_inference_time = np.std(inference_times)
        
        result = {
            'duration_seconds': actual_duration,
            'total_inferences': inference_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'min_inference_time_ms': min_inference_time * 1000,
            'max_inference_time_ms': max_inference_time * 1000,
            'std_inference_time_ms': std_inference_time * 1000,
            'fps': inference_count / actual_duration,
            'jetson_metrics': self._process_jetson_metrics(metrics)
        }
        
        print(f"✅ Continuous inference completed:")
        print(f"   Duration: {actual_duration:.1f}s")
        print(f"   Inferences: {inference_count}")
        print(f"   Avg time: {avg_inference_time*1000:.2f} ms")
        print(f"   FPS: {inference_count/actual_duration:.2f}")
        
        return result
    
    def _process_jetson_metrics(self, metrics: List[Dict]) -> Dict:
        """Process and summarize Jetson-specific metrics"""
        if not metrics:
            return {}
        
        # Extract key metrics
        ram_usage = [m.get('ram_usage_percent', 0) for m in metrics if 'ram_usage_percent' in m]
        cpu_usage = [m.get('cpu_usage_percent', 0) for m in metrics if 'cpu_usage_percent' in m]
        gpu_freq = [m.get('gpu_freq_percent', 0) for m in metrics if 'gpu_freq_percent' in m]
        emc_freq = [m.get('emc_freq_percent', 0) for m in metrics if 'emc_freq_percent' in m]
        
        result = {}
        
        if ram_usage:
            result['ram_usage_avg'] = np.mean(ram_usage)
            result['ram_usage_max'] = np.max(ram_usage)
            result['ram_usage_min'] = np.min(ram_usage)
        
        if cpu_usage:
            result['cpu_usage_avg'] = np.mean(cpu_usage)
            result['cpu_usage_max'] = np.max(cpu_usage)
            result['cpu_usage_min'] = np.min(cpu_usage)
        
        if gpu_freq:
            result['gpu_freq_avg'] = np.mean(gpu_freq)
            result['gpu_freq_max'] = np.max(gpu_freq)
        
        if emc_freq:
            result['emc_freq_avg'] = np.mean(emc_freq)
            result['emc_freq_max'] = np.max(emc_freq)
        
        return result
    
    def generate_report(self, results: Dict, output_path: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report"""
        print(f"📊 Generating benchmark report: {output_path}")
        
        # Add system info
        results['system_info'] = {
            'platform': 'Hiwonder JetAuto (Jetson Nano)',
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'model_path': self.model_path,
            'model_size_mb': os.path.getsize(self.model_path) / (1024*1024)
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Report saved to: {output_path}")
        
        # Generate plots if matplotlib is available
        try:
            self._generate_plots(results, output_path.replace('.json', '_plots.png'))
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
    
    def _generate_plots(self, results: Dict, plot_path: str):
        """Generate performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ONNX Inference Performance on Hiwonder JetAuto', fontsize=16)
        
        # Plot 1: Inference time distribution
        if 'continuous' in results:
            times = results['continuous'].get('inference_times', [])
            if times:
                axes[0, 0].hist(times, bins=20, alpha=0.7)
                axes[0, 0].set_title('Inference Time Distribution')
                axes[0, 0].set_xlabel('Time (ms)')
                axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: System resource usage
        if 'jetson_metrics' in results.get('single', {}):
            jetson = results['single']['jetson_metrics']
            metrics = ['ram_usage_avg', 'cpu_usage_avg', 'gpu_freq_avg']
            values = [jetson.get(m, 0) for m in metrics]
            axes[0, 1].bar(['RAM %', 'CPU %', 'GPU %'], values, alpha=0.7)
            axes[0, 1].set_title('System Resource Usage')
            axes[0, 1].set_ylabel('Percentage')
        
        # Plot 3: Performance comparison
        if 'single' in results and 'batch' in results:
            single_fps = results['single'].get('fps', 0)
            batch_fps = results['batch'].get('fps', 0)
            axes[1, 0].bar(['Single', 'Batch'], [single_fps, batch_fps], alpha=0.7)
            axes[1, 0].set_title('FPS Comparison')
            axes[1, 0].set_ylabel('Frames per Second')
        
        # Plot 4: Memory usage over time
        if 'continuous' in results and 'jetson_metrics' in results['continuous']:
            # This would need time-series data from continuous monitoring
            axes[1, 1].text(0.5, 0.5, 'Memory usage over time\n(requires time-series data)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Memory Usage Over Time')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to: {plot_path}")

def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='ONNX Inference Benchmarking for Hiwonder JetAuto')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--config', help='Path to config YAML file')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--output', default='benchmark_results.json', help='Output report path')
    parser.add_argument('--duration', type=int, default=30, help='Continuous test duration (seconds)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    
    args = parser.parse_args()
    
    print("🚀 ONNX Inference Benchmarking for Hiwonder JetAuto")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ONNXBenchmark(args.model, args.config)
    
    # Run benchmarks
    results = {}
    
    # Single inference benchmark
    print("\n1️⃣ Single Inference Benchmark")
    print("-" * 40)
    results['single'] = benchmark.benchmark_single_inference(args.image, args.warmup)
    
    # Batch inference benchmark
    print("\n2️⃣ Batch Inference Benchmark")
    print("-" * 40)
    results['batch'] = benchmark.benchmark_batch_inference([args.image] * 10, batch_size=1)
    
    # Continuous inference benchmark
    print("\n3️⃣ Continuous Inference Benchmark")
    print("-" * 40)
    results['continuous'] = benchmark.benchmark_continuous_inference(args.image, args.duration)
    
    # Generate report
    print("\n📊 Generating Report")
    print("-" * 40)
    benchmark.generate_report(results, args.output)
    
    print(f"\n🎉 Benchmarking completed! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
