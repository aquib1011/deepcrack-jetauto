#!/usr/bin/env python3
"""
Visualize Profiling Results
Creates plots and reports from profiling JSON output
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def load_results(json_path):
    """Load results from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        sys.exit(1)


def create_summary_plot(results, output_path='profile_summary.png'):
    """Create comprehensive summary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Edge Device Profiling Summary\n{results['device']['type']}", 
                 fontsize=16, fontweight='bold')
    
    # 1. Inference Time Distribution (simulated from statistics)
    ax1 = axes[0, 0]
    mean_ms = results['inference']['mean_ms']
    std_ms = results['inference']['std_ms']
    
    # Generate approximate distribution
    times = np.random.normal(mean_ms, std_ms, 1000)
    times = times[times > 0]  # Remove negative values
    
    ax1.hist(times, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(mean_ms, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ms:.2f}ms')
    ax1.axvline(results['inference']['p95_ms'], color='orange', linestyle='--', 
                linewidth=2, label=f'P95: {results["inference"]["p95_ms"]:.2f}ms')
    ax1.set_xlabel('Inference Time (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory Usage
    ax2 = axes[0, 1]
    memory_data = []
    labels = []
    colors = []
    
    if 'ram' in results['memory']:
        memory_data.append(results['memory']['ram']['process_rss_MB'])
        labels.append(f"RAM\n{results['memory']['ram']['process_rss_MB']:.1f} MB")
        colors.append('lightcoral')
    
    if 'gpu' in results['memory']:
        memory_data.append(results['memory']['gpu']['allocated_MB'])
        labels.append(f"GPU\n{results['memory']['gpu']['allocated_MB']:.1f} MB")
        colors.append('lightgreen')
    
    bars = ax2.bar(range(len(memory_data)), memory_data, color=colors, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Power Consumption (if available)
    ax3 = axes[1, 0]
    if results['power']:
        power_metrics = ['mean', 'max', 'min']
        power_values = [results['power']['power_W'][m] for m in power_metrics]
        
        bars = ax3.bar(power_metrics, power_values, 
                      color=['steelblue', 'tomato', 'lightblue'],
                      edgecolor='black', linewidth=2)
        ax3.set_ylabel('Power (W)', fontsize=12)
        ax3.set_title('Power Consumption', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, power_values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}W',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add energy per inference
        energy_per_inf = (results['power']['power_W']['mean'] * 
                         results['inference']['mean_ms'] / 1000)
        ax3.text(0.5, 0.95, f'Energy/Inference: {energy_per_inf*1000:.2f} mJ',
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Power data not available\n(Not running on Jetson)', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Power Consumption', fontsize=14, fontweight='bold')
    
    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    PERFORMANCE METRICS
    {'='*35}
    
    Inference Time:
      • Mean: {results['inference']['mean_ms']:.2f} ms
      • Std Dev: {results['inference']['std_ms']:.2f} ms
      • Min: {results['inference']['min_ms']:.2f} ms
      • Max: {results['inference']['max_ms']:.2f} ms
      • 95th %ile: {results['inference']['p95_ms']:.2f} ms
      • FPS: {results['inference']['fps']:.2f}
    
    Memory:
      • RAM: {results['memory']['ram']['process_rss_MB']:.1f} MB
    """
    
    if 'gpu' in results['memory']:
        summary_text += f"  • GPU: {results['memory']['gpu']['allocated_MB']:.1f} MB\n"
    
    if results['power']:
        summary_text += f"""
    Power:
      • Mean: {results['power']['power_W']['mean']:.2f} W
      • Max: {results['power']['power_W']['max']:.2f} W
        """
    
    summary_text += f"""
    Test Configuration:
      • Iterations: {results['num_iterations']}
      • Device: {results['device']['type']}
      • PyTorch: {results['device']['pytorch_version']}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {output_path}")
    
    return fig


def create_comparison_plot(results_list, labels, output_path='comparison.png'):
    """Create comparison plot for multiple profiling runs"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Profiling Comparison', fontsize=16, fontweight='bold')
    
    # 1. Inference Time Comparison
    ax1 = axes[0]
    mean_times = [r['inference']['mean_ms'] for r in results_list]
    std_times = [r['inference']['std_ms'] for r in results_list]
    
    x = np.arange(len(labels))
    bars = ax1.bar(x, mean_times, yerr=std_times, capsize=5,
                   color='skyblue', edgecolor='black', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Inference Time', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. Memory Comparison
    ax2 = axes[1]
    ram_usage = [r['memory']['ram']['process_rss_MB'] for r in results_list]
    gpu_usage = [r['memory'].get('gpu', {}).get('allocated_MB', 0) for r in results_list]
    
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, ram_usage, width, label='RAM', 
           color='lightcoral', edgecolor='black', linewidth=2)
    ax2.bar(x + width/2, gpu_usage, width, label='GPU',
           color='lightgreen', edgecolor='black', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Power Comparison (if available)
    ax3 = axes[2]
    power_available = all('power' in r and r['power'] for r in results_list)
    
    if power_available:
        mean_power = [r['power']['power_W']['mean'] for r in results_list]
        bars = ax3.bar(x, mean_power, color='steelblue', edgecolor='black', linewidth=2)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Power (W)', fontsize=12)
        ax3.set_title('Power Consumption', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Power data not available', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Power Consumption', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    
    return fig


def generate_text_report(results, output_path='profile_report.txt'):
    """Generate detailed text report"""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EDGE DEVICE PROFILING REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Device Info
        f.write("DEVICE INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Device Type: {results['device']['type']}\n")
        f.write(f"CUDA Available: {results['device']['cuda_available']}\n")
        if results['device']['cuda_device']:
            f.write(f"CUDA Device: {results['device']['cuda_device']}\n")
        f.write(f"PyTorch Version: {results['device']['pytorch_version']}\n")
        f.write(f"Test Date: {results['timestamp']}\n")
        f.write("\n")
        
        # Model Info
        f.write("MODEL INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Architecture: {results['model']['architecture']}\n")
        f.write(f"Input Size: {results['model']['input_size']}\n")
        f.write(f"Output Size: {results['model']['output_size']}\n")
        f.write(f"Model Path: {results['model']['path']}\n")
        f.write("\n")
        
        # Inference Performance
        f.write("INFERENCE PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of Iterations: {results['num_iterations']}\n")
        f.write(f"Mean Inference Time: {results['inference']['mean_ms']:.4f} ms\n")
        f.write(f"Median Inference Time: {results['inference']['median_ms']:.4f} ms\n")
        f.write(f"Standard Deviation: {results['inference']['std_ms']:.4f} ms\n")
        f.write(f"Minimum Time: {results['inference']['min_ms']:.4f} ms\n")
        f.write(f"Maximum Time: {results['inference']['max_ms']:.4f} ms\n")
        f.write(f"95th Percentile: {results['inference']['p95_ms']:.4f} ms\n")
        f.write(f"99th Percentile: {results['inference']['p99_ms']:.4f} ms\n")
        f.write(f"Throughput (FPS): {results['inference']['fps']:.2f}\n")
        f.write("\n")
        
        # Memory Usage
        f.write("MEMORY USAGE\n")
        f.write("-"*70 + "\n")
        f.write(f"Process RAM (RSS): {results['memory']['ram']['process_rss_MB']:.2f} MB\n")
        f.write(f"Process RAM (VMS): {results['memory']['ram']['process_vms_MB']:.2f} MB\n")
        f.write(f"System RAM Total: {results['memory']['ram']['system_total_MB']:.2f} MB\n")
        f.write(f"System RAM Used: {results['memory']['ram']['system_used_MB']:.2f} MB ({results['memory']['ram']['system_percent']:.1f}%)\n")
        f.write(f"System RAM Available: {results['memory']['ram']['system_available_MB']:.2f} MB\n")
        
        if 'gpu' in results['memory']:
            f.write(f"\nGPU Memory Allocated: {results['memory']['gpu']['allocated_MB']:.2f} MB\n")
            f.write(f"GPU Memory Reserved: {results['memory']['gpu']['reserved_MB']:.2f} MB\n")
            f.write(f"GPU Memory Peak: {results['memory']['gpu']['max_allocated_MB']:.2f} MB\n")
        f.write("\n")
        
        # Power Consumption
        if results['power']:
            f.write("POWER CONSUMPTION\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean Power: {results['power']['power_W']['mean']:.4f} W ({results['power']['power_mW']['mean']:.2f} mW)\n")
            f.write(f"Maximum Power: {results['power']['power_W']['max']:.4f} W ({results['power']['power_mW']['max']:.2f} mW)\n")
            f.write(f"Minimum Power: {results['power']['power_W']['min']:.4f} W ({results['power']['power_mW']['min']:.2f} mW)\n")
            f.write(f"Std Deviation: {results['power']['power_W']['std']:.4f} W\n")
            
            # Calculate energy
            energy_j = results['power']['power_W']['mean'] * results['inference']['mean_ms'] / 1000
            f.write(f"\nEnergy per Inference: {energy_j:.6f} J ({energy_j*1000:.4f} mJ)\n")
            
            # Utilization
            if 'gpu_utilization_%' in results['power']:
                f.write(f"\nGPU Utilization (Mean): {results['power']['gpu_utilization_%']['mean']:.2f}%\n")
                f.write(f"GPU Utilization (Max): {results['power']['gpu_utilization_%']['max']:.2f}%\n")
            
            if 'cpu_utilization_%' in results['power']:
                f.write(f"\nCPU Utilization (Mean): {results['power']['cpu_utilization_%']['mean']:.2f}%\n")
                f.write(f"CPU Utilization (Max): {results['power']['cpu_utilization_%']['max']:.2f}%\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Text report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize profiling results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file(s) from profiling (comma-separated for comparison)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Labels for comparison (comma-separated)')
    parser.add_argument('--output', type=str, default='profile_summary.png',
                       help='Output plot filename')
    parser.add_argument('--report', action='store_true',
                       help='Generate text report')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Parse inputs
    input_files = [f.strip() for f in args.input.split(',')]
    
    # Single file or comparison?
    if len(input_files) == 1:
        # Single profiling visualization
        results = load_results(input_files[0])
        create_summary_plot(results, args.output)
        
        if args.report:
            report_path = args.output.replace('.png', '_report.txt')
            generate_text_report(results, report_path)
    
    else:
        # Comparison mode
        results_list = [load_results(f) for f in input_files]
        
        if args.labels:
            labels = [l.strip() for l in args.labels.split(',')]
        else:
            labels = [f"Run {i+1}" for i in range(len(input_files))]
        
        if len(labels) != len(input_files):
            print(f"Error: Number of labels ({len(labels)}) doesn't match number of files ({len(input_files)})")
            sys.exit(1)
        
        create_comparison_plot(results_list, labels, args.output)
        
        if args.report:
            for i, results in enumerate(results_list):
                report_path = f"{labels[i]}_report.txt"
                generate_text_report(results, report_path)
    
    if args.show:
        plt.show()
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()

