#!/usr/bin/env python3
"""
Complete deployment and testing script for DeepCrack on Hiwonder JetAuto
Handles PTH to ONNX conversion, benchmarking, and performance monitoring
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
import shutil

class DeepCrackDeployer:
    """Complete deployment pipeline for DeepCrack on Hiwonder JetAuto"""
    
    def __init__(self, workspace_root="."):
        self.workspace_root = Path(workspace_root).resolve()
        self.robot1_dir = self.workspace_root / "Robot1"
        self.robot2_dir = self.workspace_root / "Robot2"
        self.onnx_dir = self.workspace_root / "onnx"
        
        # Paths
        self.pth_path = self.onnx_dir / "BEST.pth"
        self.onnx_path = self.onnx_dir / "BEST_optimized.onnx"
        self.config_path = self.robot2_dir / "config.yaml"
        self.sample_image = self.workspace_root / "sample_image.jpg"
        
        # Results directory
        self.results_dir = self.robot2_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check if PTH file exists
        if not self.pth_path.exists():
            print(f"❌ PyTorch model not found: {self.pth_path}")
            return False
        
        # Check if sample image exists
        if not self.sample_image.exists():
            print(f"❌ Sample image not found: {self.sample_image}")
            return False
        
        # Check Python environment
        try:
            import torch
            import onnx
            import onnxruntime
            import cv2
            import numpy as np
            print("✅ Required Python packages available")
        except ImportError as e:
            print(f"❌ Missing Python package: {e}")
            return False
        
        # Check if we're on Jetson (optional)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'aarch64' in cpuinfo and 'tegra' in cpuinfo.lower():
                    print("✅ Running on Jetson platform")
                else:
                    print("⚠️ Not running on Jetson platform")
        except:
            print("⚠️ Could not detect platform")
        
        print("✅ Prerequisites check completed")
        return True
    
    def convert_pth_to_onnx(self):
        """Convert PyTorch model to ONNX"""
        print("\n🔄 Converting PTH to ONNX...")
        
        # Check if ONNX already exists
        if self.onnx_path.exists():
            response = input(f"ONNX model already exists: {self.onnx_path}\nOverwrite? (y/N): ")
            if response.lower() != 'y':
                print("⏭️ Skipping ONNX conversion")
                return True
        
        # Run conversion script
        conversion_script = self.robot2_dir / "convert_pth_to_onnx.py"
        if not conversion_script.exists():
            print(f"❌ Conversion script not found: {conversion_script}")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, str(conversion_script)
            ], cwd=str(self.robot2_dir), capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ONNX conversion completed successfully")
                return True
            else:
                print(f"❌ ONNX conversion failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running conversion: {e}")
            return False
    
    def update_config(self):
        """Update configuration file for optimized ONNX model"""
        print("\n⚙️ Updating configuration...")
        
        config_content = f"""model_path: "{self.onnx_path.relative_to(self.robot2_dir)}"
img_size: [256, 256]
threshold: 0.5
providers: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
draw_overlay: true
alpha: 0.4
"""
        
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Configuration updated: {self.config_path}")
        return True
    
    def run_benchmark(self, duration=30):
        """Run comprehensive benchmark"""
        print(f"\n🧪 Running benchmark (duration: {duration}s)...")
        
        benchmark_script = self.robot2_dir / "benchmark_onnx_inference.py"
        if not benchmark_script.exists():
            print(f"❌ Benchmark script not found: {benchmark_script}")
            return False
        
        # Output file
        output_file = self.results_dir / f"benchmark_{int(time.time())}.json"
        
        try:
            result = subprocess.run([
                sys.executable, str(benchmark_script),
                "--model", str(self.onnx_path),
                "--config", str(self.config_path),
                "--image", str(self.sample_image),
                "--output", str(output_file),
                "--duration", str(duration),
                "--warmup", "5"
            ], cwd=str(self.robot2_dir), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Benchmark completed successfully")
                print(f"📊 Results saved to: {output_file}")
                return True
            else:
                print(f"❌ Benchmark failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running benchmark: {e}")
            return False
    
    def run_inference_test(self):
        """Run simple inference test"""
        print("\n🧪 Running inference test...")
        
        test_script = self.robot2_dir / "run_image.py"
        if not test_script.exists():
            print(f"❌ Test script not found: {test_script}")
            return False
        
        # Output file
        output_file = self.results_dir / f"test_output_{int(time.time())}.png"
        
        try:
            result = subprocess.run([
                sys.executable, str(test_script),
                "--config", str(self.config_path),
                "--input", str(self.sample_image),
                "--output", str(output_file)
            ], cwd=str(self.robot2_dir), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Inference test completed successfully")
                print(f"🖼️ Output saved to: {output_file}")
                return True
            else:
                print(f"❌ Inference test failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running inference test: {e}")
            return False
    
    def run_monitoring_demo(self, duration=60):
        """Run monitoring demo"""
        print(f"\n📊 Running monitoring demo (duration: {duration}s)...")
        
        monitor_script = self.robot2_dir / "realtime_monitor.py"
        if not monitor_script.exists():
            print(f"❌ Monitor script not found: {monitor_script}")
            return False
        
        # Log file
        log_file = self.results_dir / f"monitoring_{int(time.time())}.csv"
        
        try:
            print("🚀 Starting monitoring... Press Ctrl+C to stop early")
            result = subprocess.run([
                sys.executable, str(monitor_script),
                "--interval", "0.5",
                "--log", str(log_file),
                "--duration", str(duration)
            ], cwd=str(self.robot2_dir))
            
            print(f"✅ Monitoring completed")
            print(f"📊 Log saved to: {log_file}")
            return True
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            return True
        except Exception as e:
            print(f"❌ Error running monitoring: {e}")
            return False
    
    def generate_summary_report(self):
        """Generate summary report of all results"""
        print("\n📊 Generating summary report...")
        
        # Find latest benchmark results
        benchmark_files = list(self.results_dir.glob("benchmark_*.json"))
        if not benchmark_files:
            print("⚠️ No benchmark results found")
            return False
        
        latest_benchmark = max(benchmark_files, key=os.path.getctime)
        
        try:
            with open(latest_benchmark, 'r') as f:
                benchmark_data = json.load(f)
            
            # Generate summary
            summary = {
                "deployment_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "platform": "Hiwonder JetAuto (Jetson Nano)",
                    "model_path": str(self.onnx_path),
                    "model_size_mb": self.onnx_path.stat().st_size / (1024*1024) if self.onnx_path.exists() else 0
                },
                "performance_summary": {
                    "single_inference": {
                        "time_ms": benchmark_data.get("single", {}).get("inference_time_ms", 0),
                        "fps": benchmark_data.get("single", {}).get("fps", 0),
                        "cpu_usage": benchmark_data.get("single", {}).get("cpu_usage_percent", 0),
                        "memory_usage": benchmark_data.get("single", {}).get("memory_usage_percent", 0)
                    },
                    "continuous_inference": {
                        "avg_time_ms": benchmark_data.get("continuous", {}).get("avg_inference_time_ms", 0),
                        "fps": benchmark_data.get("continuous", {}).get("fps", 0),
                        "total_inferences": benchmark_data.get("continuous", {}).get("total_inferences", 0)
                    }
                },
                "system_info": benchmark_data.get("system_info", {})
            }
            
            # Save summary
            summary_file = self.results_dir / "deployment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ Summary report generated: {summary_file}")
            
            # Print key metrics
            print("\n📈 Key Performance Metrics:")
            print(f"   Single inference time: {summary['performance_summary']['single_inference']['time_ms']:.2f} ms")
            print(f"   Single inference FPS: {summary['performance_summary']['single_inference']['fps']:.2f}")
            print(f"   Continuous FPS: {summary['performance_summary']['continuous_inference']['fps']:.2f}")
            print(f"   CPU usage: {summary['performance_summary']['single_inference']['cpu_usage']:.1f}%")
            print(f"   Memory usage: {summary['performance_summary']['single_inference']['memory_usage']:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return False
    
    def deploy(self, skip_conversion=False, benchmark_duration=30, monitoring_duration=60):
        """Complete deployment pipeline"""
        print("🚀 DeepCrack Deployment for Hiwonder JetAuto")
        print("=" * 60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed")
            return False
        
        # Step 2: Convert PTH to ONNX
        if not skip_conversion:
            if not self.convert_pth_to_onnx():
                print("❌ ONNX conversion failed")
                return False
        else:
            print("⏭️ Skipping ONNX conversion")
        
        # Step 3: Update configuration
        if not self.update_config():
            print("❌ Configuration update failed")
            return False
        
        # Step 4: Run inference test
        if not self.run_inference_test():
            print("❌ Inference test failed")
            return False
        
        # Step 5: Run benchmark
        if not self.run_benchmark(benchmark_duration):
            print("❌ Benchmark failed")
            return False
        
        # Step 6: Run monitoring demo
        if not self.run_monitoring_demo(monitoring_duration):
            print("❌ Monitoring demo failed")
            return False
        
        # Step 7: Generate summary
        if not self.generate_summary_report():
            print("❌ Summary generation failed")
            return False
        
        print("\n🎉 Deployment completed successfully!")
        print(f"📁 Results directory: {self.results_dir}")
        print("\nNext steps:")
        print("1. Review benchmark results in the results directory")
        print("2. Use the optimized ONNX model for your robot applications")
        print("3. Monitor system performance during real-world usage")
        
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy DeepCrack on Hiwonder JetAuto')
    parser.add_argument('--skip-conversion', action='store_true', help='Skip PTH to ONNX conversion')
    parser.add_argument('--benchmark-duration', type=int, default=30, help='Benchmark duration in seconds')
    parser.add_argument('--monitoring-duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('--workspace', default='.', help='Workspace root directory')
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = DeepCrackDeployer(args.workspace)
    
    # Run deployment
    success = deployer.deploy(
        skip_conversion=args.skip_conversion,
        benchmark_duration=args.benchmark_duration,
        monitoring_duration=args.monitoring_duration
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
