#!/usr/bin/env python3
"""
Example scripts for common benchmarking tasks
"""

from run_benchmark import main as run_benchmark_main
from utils.utils import ResultsAnalyzer, check_dependencies, clean_cache
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_single_benchmark():
    """Example: Run a single benchmark"""

    print("=== Single Benchmark Example ===")
    print("Running Llama 3.2 3B with HF format, FP16 quantization on CPU")

    # Simulate command line arguments
    original_argv = sys.argv
    sys.argv = [
        "run_benchmark.py",
        "--model", "llama3.2-3b",
        "--format", "hf",
        "--quantization", "fp16",
        "--hardware", "cpu",
        "--batch-size", "1",
        "--warmup-runs", "2",
        "--benchmark-runs", "3",
        "--max-new-tokens", "50",
        "--convert-model",
        "--custom-prompts", "Hello, how are you today?", "What is artificial intelligence?"
    ]

    try:
        run_benchmark_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def example_gpu_benchmark():
    """Example: Run GPU benchmark if available"""

    print("=== GPU Benchmark Example ===")
    print("Running Qwen 2.5 with OpenVINO format, 8-bit quantization on GPU")

    # Simulate command line arguments
    original_argv = sys.argv
    sys.argv = [
        "run_benchmark.py",
        "--model", "qwen2.5",
        "--format", "ov",
        "--quantization", "8bit",
        "--hardware", "gpu",
        "--batch-size", "2",
        "--convert-model",
        "--quantize-model",
        "--custom-prompts", "Explain quantum computing", "Write a short poem about AI"
    ]

    try:
        run_benchmark_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def example_quantization_comparison():
    """Example: Compare different quantization methods"""

    print("=== Quantization Comparison Example ===")

    model = "llama3.2-3b"
    format_type = "hf"
    hardware = "cpu"

    quantizations = ["fp16", "8bit", "4bit"]

    for quant in quantizations:
        print(f"\n--- Testing {quant} quantization ---")

        original_argv = sys.argv
        sys.argv = [
            "run_benchmark.py",
            "--model", model,
            "--format", format_type,
            "--quantization", quant,
            "--hardware", hardware,
            "--batch-size", "1",
            "--warmup-runs", "1",
            "--benchmark-runs", "2",
            "--max-new-tokens", "32",
            "--convert-model",
            "--quantize-model",
            "--custom-prompts", "Test prompt for quantization comparison"
        ]

        try:
            run_benchmark_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv


def example_hardware_detection():
    """Example: Detect available hardware"""

    print("=== Hardware Detection Example ===")

    # Simulate command line arguments
    original_argv = sys.argv
    sys.argv = [
        "run_benchmark.py",
        "--detect-hardware"
    ]

    try:
        run_benchmark_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def example_batch_benchmark():
    """Example: Run batch benchmarks"""

    print("=== Batch Benchmark Example ===")
    print("Running benchmarks for all supported combinations")

    # Simulate command line arguments
    original_argv = sys.argv
    sys.argv = [
        "run_benchmark.py",
        "--batch-benchmark",
        "--warmup-runs", "1",
        "--benchmark-runs", "2",
        "--max-new-tokens", "32",
        "--convert-model",
        "--output-file", "batch_results.json"
    ]

    try:
        run_benchmark_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def example_results_analysis():
    """Example: Analyze benchmark results"""

    print("=== Results Analysis Example ===")

    # Check if results file exists
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found. Run some benchmarks first.")
        return

    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print("No result files found. Run some benchmarks first.")
        return

    # Use the most recent results file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"Analyzing results from: {latest_file.name}")

    analyzer = ResultsAnalyzer()

    # Generate report
    report = analyzer.generate_performance_report(latest_file.name)
    print("\n" + "="*50)
    print(report)
    print("="*50)

    # Generate visualizations
    try:
        analyzer.generate_visualizations(latest_file.name)
        print("\nVisualizations generated in 'visualizations' directory")
    except Exception as e:
        print(f"Visualization generation failed: {e}")

    # Export to CSV
    try:
        analyzer.export_to_csv(latest_file.name)
        print("Results exported to CSV")
    except Exception as e:
        print(f"CSV export failed: {e}")


def example_model_conversion():
    """Example: Convert and quantize models"""

    print("=== Model Conversion Example ===")

    from models.model_manager import ModelManager
    from quantization.quantization_manager import QuantizationManager

    model_manager = ModelManager()
    quantization_manager = QuantizationManager()

    model_name = "llama3.2-3b"

    print(f"Converting {model_name} to different formats...")

    # Convert to ONNX
    try:
        print("Converting to ONNX...")
        onnx_path = model_manager.convert_to_onnx(model_name, "fp16")
        if onnx_path:
            print(f"✅ ONNX conversion successful: {onnx_path}")
        else:
            print("❌ ONNX conversion failed")
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")

    # Convert to OpenVINO
    try:
        print("Converting to OpenVINO...")
        ov_path = model_manager.convert_to_openvino(model_name, "fp16")
        if ov_path:
            print(f"✅ OpenVINO conversion successful: {ov_path}")
        else:
            print("❌ OpenVINO conversion failed")
    except Exception as e:
        print(f"❌ OpenVINO conversion failed: {e}")

    # Quantize models
    for format_type in ["hf", "onnx", "ov"]:
        for quant_type in ["8bit", "4bit"]:
            try:
                print(f"Quantizing {format_type} model to {quant_type}...")
                result = quantization_manager.quantize_model(
                    model_name, format_type, quant_type
                )
                if result:
                    print(
                        f"✅ {format_type} {quant_type} quantization successful")
                else:
                    print(f"❌ {format_type} {quant_type} quantization failed")
            except Exception as e:
                print(f"❌ {format_type} {quant_type} quantization failed: {e}")


def main():
    """Main function to run examples"""

    print("ONNX OpenVINO Benchmarking LLM Models - Examples")
    print("="*60)

    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install required dependencies before running examples.")
        return

    examples = {
        "1": ("Hardware Detection", example_hardware_detection),
        "2": ("Single Benchmark", example_single_benchmark),
        "3": ("GPU Benchmark", example_gpu_benchmark),
        "4": ("Quantization Comparison", example_quantization_comparison),
        "5": ("Model Conversion", example_model_conversion),
        "6": ("Batch Benchmark", example_batch_benchmark),
        "7": ("Results Analysis", example_results_analysis),
        "8": ("Clean Cache", clean_cache)
    }

    while True:
        print("\nAvailable Examples:")
        for key, (description, _) in examples.items():
            print(f"  {key}. {description}")
        print("  0. Exit")

        choice = input("\nEnter your choice (0-8): ").strip()

        if choice == "0":
            break
        elif choice in examples:
            description, func = examples[choice]
            print(f"\nRunning: {description}")
            print("-" * 40)

            try:
                func()
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
            except Exception as e:
                print(f"Error running example: {e}")
        else:
            print("Invalid choice. Please try again.")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
