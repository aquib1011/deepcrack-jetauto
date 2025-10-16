#!/usr/bin/env python3
"""
Real-time monitoring script for Hiwonder JetAuto during inference
Monitors CPU, GPU, memory, power, and temperature in real-time
"""

import time
import psutil
import subprocess
import threading
import queue
import json
import argparse
from datetime import datetime
from typing import Dict, List
import signal
import sys

class RealtimeMonitor:
    """Real-time system monitoring for Jetson Nano"""
    
    def __init__(self, sample_interval=0.5, log_file=None):
        self.sample_interval = sample_interval
        self.log_file = log_file
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.tegrastats_process = None
        self.stop_event = threading.Event()
        
        # Initialize logging
        if self.log_file:
            self.log_file = open(self.log_file, 'w')
            self.log_file.write('timestamp,cpu_percent,memory_percent,memory_used_mb,memory_total_mb,gpu_freq_percent,emc_freq_percent,temperature_c\n')
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print("🚀 Starting real-time monitoring...")
        self.monitoring = True
        self.stop_event.clear()
        
        # Start tegrastats monitoring
        self._start_tegrastats()
        
        # Start main monitoring loop
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("✅ Monitoring started! Press Ctrl+C to stop.")
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        print("\n🛑 Stopping monitoring...")
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self._stop_tegrastats()
        
        if self.log_file:
            self.log_file.close()
        
        print("✅ Monitoring stopped.")
    
    def _start_tegrastats(self):
        """Start tegrastats process for Jetson-specific metrics"""
        try:
            self.tegrastats_process = subprocess.Popen(
                ['tegrastats', '--interval', '500'],  # 500ms interval
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            print(f"⚠️ Could not start tegrastats: {e}")
            self.tegrastats_process = None
    
    def _stop_tegrastats(self):
        """Stop tegrastats process"""
        if self.tegrastats_process:
            self.tegrastats_process.terminate()
            self.tegrastats_process.wait(timeout=2)
            self.tegrastats_process = None
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Get system metrics
                metrics = self._collect_metrics()
                
                # Add to queue for processing
                self.metrics_queue.put(metrics)
                
                # Log to file if enabled
                if self.log_file:
                    self._log_metrics(metrics)
                
                # Display metrics
                self._display_metrics(metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(1)
    
    def _collect_metrics(self) -> Dict:
        """Collect current system metrics"""
        timestamp = datetime.now()
        
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Jetson-specific metrics from tegrastats
        jetson_metrics = self._get_tegrastats_metrics()
        
        # Temperature (if available)
        temperature = self._get_temperature()
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_total_mb': memory.total / (1024 * 1024),
            'memory_available_mb': memory.available / (1024 * 1024),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'temperature_c': temperature,
            **jetson_metrics
        }
    
    def _get_tegrastats_metrics(self) -> Dict:
        """Get Jetson-specific metrics from tegrastats"""
        if not self.tegrastats_process:
            return {}
        
        try:
            # Read latest line from tegrastats
            line = self.tegrastats_process.stdout.readline()
            if not line:
                return {}
            
            return self._parse_tegrastats_line(line.strip())
            
        except Exception as e:
            return {}
    
    def _parse_tegrastats_line(self, line: str) -> Dict:
        """Parse tegrastats output line"""
        metrics = {}
        
        try:
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
            
            # Parse temperature
            if 'TEMP' in line:
                temp_part = line.split('TEMP')[1].split('C')[0].strip()
                if '@' in temp_part:
                    temp_value = temp_part.split('@')[0].strip()
                    metrics['temperature_c'] = float(temp_value)
            
        except Exception as e:
            pass
        
        return metrics
    
    def _get_temperature(self) -> float:
        """Get system temperature"""
        try:
            # Try to read from thermal zone
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except:
            return 0.0
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to file"""
        if not self.log_file:
            return
        
        timestamp = metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        memory_used = metrics.get('memory_used_mb', 0)
        memory_total = metrics.get('memory_total_mb', 0)
        gpu_freq = metrics.get('gpu_freq_percent', 0)
        emc_freq = metrics.get('emc_freq_percent', 0)
        temp = metrics.get('temperature_c', 0)
        
        self.log_file.write(f"{timestamp},{cpu:.1f},{memory:.1f},{memory_used:.1f},{memory_total:.1f},{gpu_freq:.1f},{emc_freq:.1f},{temp:.1f}\n")
        self.log_file.flush()
    
    def _display_metrics(self, metrics: Dict):
        """Display current metrics in terminal"""
        timestamp = metrics['timestamp'].strftime('%H:%M:%S')
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        memory_used = metrics.get('memory_used_mb', 0)
        memory_total = metrics.get('memory_total_mb', 0)
        gpu_freq = metrics.get('gpu_freq_percent', 0)
        emc_freq = metrics.get('emc_freq_percent', 0)
        temp = metrics.get('temperature_c', 0)
        
        # Clear line and display metrics
        print(f"\r[{timestamp}] CPU: {cpu:5.1f}% | RAM: {memory:5.1f}% ({memory_used:6.1f}/{memory_total:6.1f}MB) | GPU: {gpu_freq:5.1f}% | EMC: {emc_freq:5.1f}% | Temp: {temp:5.1f}°C", end='', flush=True)
    
    def get_latest_metrics(self) -> Dict:
        """Get latest metrics from queue"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return {}
    
    def get_metrics_history(self, max_samples=100) -> List[Dict]:
        """Get recent metrics history"""
        history = []
        while len(history) < max_samples:
            try:
                metrics = self.metrics_queue.get_nowait()
                history.append(metrics)
            except queue.Empty:
                break
        return history

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal...")
    if 'monitor' in globals():
        monitor.stop_monitoring()
    sys.exit(0)

def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description='Real-time monitoring for Hiwonder JetAuto')
    parser.add_argument('--interval', type=float, default=0.5, help='Sampling interval in seconds')
    parser.add_argument('--log', help='Log file path (CSV format)')
    parser.add_argument('--duration', type=int, help='Monitoring duration in seconds (0 for infinite)')
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("📊 Real-time Monitoring for Hiwonder JetAuto")
    print("=" * 50)
    print(f"Sampling interval: {args.interval}s")
    if args.log:
        print(f"Logging to: {args.log}")
    if args.duration:
        print(f"Duration: {args.duration}s")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    # Initialize monitor
    global monitor
    monitor = RealtimeMonitor(sample_interval=args.interval, log_file=args.log)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run for specified duration or until interrupted
        if args.duration:
            time.sleep(args.duration)
            monitor.stop_monitoring()
        else:
            # Run indefinitely until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        monitor.stop_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
