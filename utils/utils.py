"""
Utility functions for the benchmarking project
"""

import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """Analyze and visualize benchmark results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file"""

        results_path = self.results_dir / filename

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, 'r') as f:
            return json.load(f)

    def results_to_dataframe(self, results_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""

        results_list = results_data.get("results", [])

        if not results_list:
            return pd.DataFrame()

        # Filter out failed results
        successful_results = [r for r in results_list if r.get("error") is None]

        if not successful_results:
            logger.warning("No successful results found")
            return pd.DataFrame()

        return pd.DataFrame(successful_results)

    def generate_performance_report(self, filename: str) -> str:
        """Generate a comprehensive performance report"""

        results_data = self.load_results(filename)
        df = self.results_to_dataframe(results_data)

        if df.empty:
            return "No data available for analysis"

        report = []
        report.append("# LLM Benchmarking Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Hardware information
        hardware_info = results_data.get("metadata", {}).get("hardware_info", {})
        report.append("## Hardware Information")

        cpu_info = hardware_info.get("cpu", {})
        report.append(f"- **CPU**: {cpu_info.get('brand', 'Unknown')}")
        report.append(f"- **Cores**: {cpu_info.get('count', 'Unknown')}")
        report.append(f"- **Memory**: {hardware_info.get('system', {}).get('memory_gb', 'Unknown')} GB")

        gpu_info = hardware_info.get("gpu", {})
        if gpu_info.get("available", False):
            report.append(f"- **GPU**: Available ({gpu_info.get('total_memory_gb', 0):.1f} GB)")
        else:
            report.append("- **GPU**: Not available")

        npu_info = hardware_info.get("npu", {})
        if npu_info.get("available", False):
            report.append("- **NPU**: Available")
        else:
            report.append("- **NPU**: Not available")

        report.append("")

        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- **Total Benchmarks**: {len(df)}")
        report.append(f"- **Models Tested**: {', '.join(df['model_name'].unique())}")
        report.append(f"- **Formats Tested**: {', '.join(df['model_format'].unique())}")
        report.append(f"- **Hardware Tested**: {', '.join(df['hardware'].unique())}")
        report.append(f"- **Quantizations Tested**: {', '.join(df['quantization'].unique())}")
        report.append("")

        # Performance metrics
        report.append("## Performance Metrics")
        report.append(f"- **Average Latency**: {df['latency_ms'].mean():.1f} ms")
        report.append(f"- **Average Throughput**: {df['throughput_tokens_per_sec'].mean():.1f} tokens/sec")
        report.append(f"- **Average Memory Usage**: {df['memory_used_mb'].mean():.1f} MB")
        report.append("")

        # Best performing configurations
        report.append("## Best Performing Configurations")

        # Lowest latency
        best_latency = df.loc[df['latency_ms'].idxmin()]
        report.append(f"- **Lowest Latency**: {best_latency['model_name']} ({best_latency['model_format']}, {best_latency['quantization']}) on {best_latency['hardware']}: {best_latency['latency_ms']:.1f} ms")

        # Highest throughput
        best_throughput = df.loc[df['throughput_tokens_per_sec'].idxmax()]
        report.append(f"- **Highest Throughput**: {best_throughput['model_name']} ({best_throughput['model_format']}, {best_throughput['quantization']}) on {best_throughput['hardware']}: {best_throughput['throughput_tokens_per_sec']:.1f} tokens/sec")

        # Lowest memory usage
        best_memory = df.loc[df['memory_used_mb'].idxmin()]
        report.append(f"- **Lowest Memory Usage**: {best_memory['model_name']} ({best_memory['model_format']}, {best_memory['quantization']}) on {best_memory['hardware']}: {best_memory['memory_used_mb']:.1f} MB")

        report.append("")

        # Performance by model
        report.append("## Performance by Model")
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            report.append(f"### {model}")
            report.append(f"- **Average Latency**: {model_df['latency_ms'].mean():.1f} ms")
            report.append(f"- **Average Throughput**: {model_df['throughput_tokens_per_sec'].mean():.1f} tokens/sec")
            report.append(f"- **Average Memory**: {model_df['memory_used_mb'].mean():.1f} MB")
            report.append("")

        # Performance by hardware
        report.append("## Performance by Hardware")
        for hardware in df['hardware'].unique():
            hardware_df = df[df['hardware'] == hardware]
            report.append(f"### {hardware.upper()}")
            report.append(f"- **Average Latency**: {hardware_df['latency_ms'].mean():.1f} ms")
            report.append(f"- **Average Throughput**: {hardware_df['throughput_tokens_per_sec'].mean():.1f} tokens/sec")
            report.append(f"- **Average Memory**: {hardware_df['memory_used_mb'].mean():.1f} MB")
            report.append("")

        # Performance by quantization
        report.append("## Performance by Quantization")
        for quant in df['quantization'].unique():
            quant_df = df[df['quantization'] == quant]
            report.append(f"### {quant}")
            report.append(f"- **Average Latency**: {quant_df['latency_ms'].mean():.1f} ms")
            report.append(f"- **Average Throughput**: {quant_df['throughput_tokens_per_sec'].mean():.1f} tokens/sec")
            report.append(f"- **Average Memory**: {quant_df['memory_used_mb'].mean():.1f} MB")
            report.append("")

        return "\n".join(report)

    def generate_visualizations(self, filename: str, output_dir: str = "visualizations"):
        """Generate performance visualizations"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results_data = self.load_results(filename)
        df = self.results_to_dataframe(results_data)

        if df.empty:
            logger.warning("No data available for visualization")
            return

        # Set up matplotlib
        plt.style.use('seaborn-v0_8')

        # 1. Latency comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='model_name', y='latency_ms', hue='hardware')
        plt.title('Latency by Model and Hardware')
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'latency_comparison.png', dpi=300)
        plt.close()

        # 2. Throughput comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='model_name', y='throughput_tokens_per_sec', hue='hardware')
        plt.title('Throughput by Model and Hardware')
        plt.xlabel('Model')
        plt.ylabel('Throughput (tokens/sec)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'throughput_comparison.png', dpi=300)
        plt.close()

        # 3. Memory usage comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='model_name', y='memory_used_mb', hue='quantization')
        plt.title('Memory Usage by Model and Quantization')
        plt.xlabel('Model')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'memory_comparison.png', dpi=300)
        plt.close()

        # 4. Quantization performance
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='latency_ms', y='throughput_tokens_per_sec',
                       hue='quantization', style='model_format', size='memory_used_mb')
        plt.title('Latency vs Throughput by Quantization')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (tokens/sec)')
        plt.tight_layout()
        plt.savefig(output_path / 'quantization_performance.png', dpi=300)
        plt.close()

        # 5. Hardware comparison heatmap
        pivot_latency = df.pivot_table(
            values='latency_ms',
            index='model_name',
            columns='hardware',
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_latency, annot=True, fmt='.1f', cmap='RdYlBu_r')
        plt.title('Average Latency by Model and Hardware (ms)')
        plt.tight_layout()
        plt.savefig(output_path / 'hardware_heatmap.png', dpi=300)
        plt.close()

        logger.info(f"Visualizations saved to: {output_path}")

    def compare_configurations(self, filename: str) -> pd.DataFrame:
        """Compare different model configurations"""

        results_data = self.load_results(filename)
        df = self.results_to_dataframe(results_data)

        if df.empty:
            return pd.DataFrame()

        # Group by configuration
        comparison = df.groupby(['model_name', 'model_format', 'quantization', 'hardware']).agg({
            'latency_ms': ['mean', 'std'],
            'throughput_tokens_per_sec': ['mean', 'std'],
            'memory_used_mb': ['mean', 'std'],
            'tokens_generated': 'mean'
        }).round(2)

        # Flatten column names
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]

        return comparison.reset_index()

    def export_to_csv(self, filename: str, output_file: str = "benchmark_results.csv"):
        """Export results to CSV"""

        results_data = self.load_results(filename)
        df = self.results_to_dataframe(results_data)

        if df.empty:
            logger.warning("No data available for export")
            return

        output_path = self.results_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to: {output_path}")


def setup_logging(log_level: str = "INFO", log_file: str = "logs/benchmark.log"):
    """Set up logging configuration"""

    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed"""

    required_packages = [
        "torch", "transformers", "numpy", "pandas",
        "matplotlib", "seaborn", "psutil", "cpuinfo"
    ]

    optional_packages = {
        "onnx": "ONNX format support",
        "onnxruntime": "ONNX runtime support",
        "openvino": "OpenVINO support",
        "openvino_genai": "OpenVINO GenAI support",
        "bitsandbytes": "Quantization support",
        "pynvml": "GPU monitoring"
    }

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)

    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, description))

    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"  - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    if missing_optional:
        print("⚠️  Missing optional packages:")
        for package, description in missing_optional:
            print(f"  - {package}: {description}")
        print("\nSome features may not be available.")

    print("✅ All required dependencies are installed")
    return True

def clean_cache():
    """Clean model cache and temporary files"""

    cache_dirs = [
        Path("cache"),
        Path("models"),
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "transformers"
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"✅ Cleaned cache: {cache_dir}")
            except Exception as e:
                print(f"❌ Failed to clean cache {cache_dir}: {e}")

    # Clean Python cache
    import subprocess
    try:
        subprocess.run(["find", ".", "-name", "*.pyc", "-delete"],
                      capture_output=True, text=True)
        subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"],
                      capture_output=True, text=True)
        print("✅ Cleaned Python cache")
    except Exception as e:
        print(f"❌ Failed to clean Python cache: {e}")

if __name__ == "__main__":
    # Test utilities
    import argparse

    parser = argparse.ArgumentParser(description="Utility functions for benchmarking")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--clean-cache", action="store_true", help="Clean cache")
    parser.add_argument("--analyze", type=str, help="Analyze results file")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV")

    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()

    if args.clean_cache:
        clean_cache()

    if args.analyze:
        analyzer = ResultsAnalyzer()
        report = analyzer.generate_performance_report(args.analyze)
        print(report)

        # Generate visualizations
        analyzer.generate_visualizations(args.analyze)

    if args.export_csv:
        analyzer = ResultsAnalyzer()
        analyzer.export_to_csv(args.export_csv)
