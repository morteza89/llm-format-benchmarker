#!/usr/bin/env python3
"""
Main benchmarking script for ONNX OpenVINO LLM Models
"""

from benchmarking.benchmark_engine import BenchmarkEngine
from quantization.quantization_manager import QuantizationManager
from models.model_manager import ModelManager
from utils.hardware_detector import HardwareDetector
from configs.config import MODELS, HARDWARE, DEFAULT_ENV
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Set up logging
log_file = project_root / 'logs' / 'benchmark.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables"""
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models with different formats and hardware backends"
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to benchmark"
    )

    parser.add_argument(
        "--format",
        choices=["hf", "onnx", "ov"],
        help="Model format to use"
    )

    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "fp16"],
        default="fp16",
        help="Quantization type (default: fp16)"
    )

    parser.add_argument(
        "--hardware",
        choices=list(HARDWARE.keys()),
        help="Hardware backend to use"
    )

    # Benchmark parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)"
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )

    # Model preparation
    parser.add_argument(
        "--convert-model",
        action="store_true",
        help="Convert model to specified format if needed"
    )

    parser.add_argument(
        "--quantize-model",
        action="store_true",
        help="Quantize model if needed"
    )

    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force model conversion even if it exists"
    )

    # Custom prompts
    parser.add_argument(
        "--custom-prompts",
        nargs="+",
        help="Custom prompts to use for benchmarking"
    )

    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File containing prompts (one per line)"
    )

    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results (default: auto-generated)"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save benchmark results to file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Hardware detection
    parser.add_argument(
        "--detect-hardware",
        action="store_true",
        help="Only detect and display hardware information"
    )

    # Batch benchmarking
    parser.add_argument(
        "--batch-benchmark",
        action="store_true",
        help="Run benchmarks for all supported combinations"
    )

    return parser.parse_args()


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from file"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
    except Exception as e:
        logger.error(f"Failed to load prompts from file: {e}")
        return []

    return prompts


def detect_hardware():
    """Detect and display hardware information"""
    detector = HardwareDetector()
    detector.print_hardware_summary()

    print("\n=== Recommendations ===")

    if detector.is_hardware_supported("npu"):
        print("✓ NPU detected - Use OpenVINO format for best performance")

    if detector.is_hardware_supported("gpu"):
        if detector.gpu_info["nvidia_gpu"]:
            print("✓ NVIDIA GPU detected - All formats supported")
        elif detector.gpu_info["intel_gpu"]:
            print("✓ Intel GPU detected - HF and OpenVINO formats recommended")

    if detector.is_hardware_supported("cpu"):
        print("✓ CPU always available - All formats supported")

    # Memory recommendations
    cpu_memory = detector.get_hardware_memory("cpu")
    if cpu_memory and cpu_memory < 16:
        print("⚠ Low system memory - Consider using 4-bit or 8-bit quantization")

    gpu_memory = detector.get_hardware_memory("gpu")
    if gpu_memory and gpu_memory < 8:
        print("⚠ Low GPU memory - Consider using CPU backend or quantization")


def prepare_model(
    model_name: str,
    model_format: str,
    quantization: str,
    force_conversion: bool = False,
    convert_model: bool = False,
    quantize_model: bool = False
) -> bool:
    """Prepare model for benchmarking"""

    model_manager = ModelManager()
    quantization_manager = QuantizationManager()

    try:
        # Check if model exists
        if model_format == "hf":
            # HF models are downloaded on-demand
            logger.info(f"HF model {model_name} will be downloaded if needed")
            return True

        elif model_format == "onnx":
            model_path = project_root / "models" / \
                f"{model_name}_{quantization}.onnx"

            if not model_path.exists() or force_conversion:
                if convert_model:
                    logger.info(f"Converting {model_name} to ONNX format")
                    result = model_manager.convert_to_onnx(
                        model_name, quantization, force_conversion
                    )
                    if not result:
                        logger.error("ONNX conversion failed")
                        return False
                else:
                    logger.error(f"ONNX model not found: {model_path}")
                    logger.info("Use --convert-model to generate it")
                    return False

            # Quantize if needed
            if quantize_model and quantization != "fp16":
                logger.info(f"Quantizing ONNX model to {quantization}")
                result = quantization_manager.quantize_model(
                    model_name, "onnx", quantization
                )
                if not result:
                    logger.error("ONNX quantization failed")
                    return False

        elif model_format == "ov":
            model_path = project_root / "models" / \
                f"{model_name}_{quantization}_ov"

            if not model_path.exists() or force_conversion:
                if convert_model:
                    logger.info(f"Converting {model_name} to OpenVINO format")
                    result = model_manager.convert_to_openvino(
                        model_name, quantization, force_conversion
                    )
                    if not result:
                        logger.error("OpenVINO conversion failed")
                        return False
                else:
                    logger.error(f"OpenVINO model not found: {model_path}")
                    logger.info("Use --convert-model to generate it")
                    return False

            # Quantize if needed
            if quantize_model and quantization != "fp16":
                logger.info(f"Quantizing OpenVINO model to {quantization}")
                result = quantization_manager.quantize_model(
                    model_name, "ov", quantization
                )
                if not result:
                    logger.error("OpenVINO quantization failed")
                    return False

        return True

    except Exception as e:
        logger.error(f"Model preparation failed: {e}")
        return False


def run_single_benchmark(args) -> bool:
    """Run benchmark for single configuration"""

    # Prepare model
    if not prepare_model(
        args.model,
        args.format,
        args.quantization,
        args.force_conversion,
        args.convert_model,
        args.quantize_model
    ):
        return False

    # Load custom prompts
    custom_prompts = None
    if args.custom_prompts:
        custom_prompts = args.custom_prompts
    elif args.prompts_file:
        custom_prompts = load_prompts_from_file(args.prompts_file)
        if not custom_prompts:
            logger.error("No prompts loaded from file")
            return False

    # Initialize benchmark engine
    engine = BenchmarkEngine()

    # Update benchmark configuration
    from configs.config import BENCHMARK
    BENCHMARK.warmup_runs = args.warmup_runs
    BENCHMARK.benchmark_runs = args.benchmark_runs
    BENCHMARK.max_new_tokens = args.max_new_tokens

    # Run benchmark
    logger.info(
        f"Starting benchmark: {args.model} ({args.format}, {args.quantization}) on {args.hardware}")

    results = engine.run_benchmark(
        model_name=args.model,
        model_format=args.format,
        quantization=args.quantization,
        hardware=args.hardware,
        batch_size=args.batch_size,
        custom_prompts=custom_prompts
    )

    # Display results
    print("\n=== Benchmark Results ===")

    for i, result in enumerate(results, 1):
        print(f"\nRun {i}:")
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Prompt: {result.prompt[:50]}...")
            print(f"  Tokens Generated: {result.tokens_generated}")
            print(f"  Latency: {result.latency_ms:.1f} ms")
            print(
                f"  Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec")
            print(f"  Memory Used: {result.memory_used_mb:.1f} MB")
            if result.gpu_memory_used_mb > 0:
                print(f"  GPU Memory Used: {result.gpu_memory_used_mb:.1f} MB")

    # Summary
    summary = engine.get_summary()
    print(f"\n=== Summary ===")
    print(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
    print(f"Successful: {summary.get('successful_benchmarks', 0)}")
    print(f"Failed: {summary.get('failed_benchmarks', 0)}")

    if summary.get('successful_benchmarks', 0) > 0:
        print(
            f"Average Latency: {summary.get('average_latency_ms', 0):.1f} ms")
        print(
            f"Average Throughput: {summary.get('average_throughput', 0):.1f} tokens/sec")
        print(f"Average Memory: {summary.get('average_memory_mb', 0):.1f} MB")

    # Save results
    if args.save_results:
        output_file = engine.save_results(args.output_file)
        print(f"\nResults saved to: {output_file}")

    return summary['successful_benchmarks'] > 0


def run_batch_benchmark(args) -> bool:
    """Run benchmarks for all supported combinations"""

    detector = HardwareDetector()
    engine = BenchmarkEngine()

    # Get all combinations
    combinations = []

    for model_name in MODELS.keys():
        for hardware in HARDWARE.keys():
            if detector.is_hardware_supported(hardware):
                supported_formats = detector.get_supported_formats(hardware)

                for format_type in supported_formats:
                    for quantization in ["fp16", "8bit", "4bit"]:
                        combinations.append({
                            "model": model_name,
                            "format": format_type,
                            "quantization": quantization,
                            "hardware": hardware
                        })

    logger.info(f"Running {len(combinations)} benchmark combinations")

    successful_count = 0
    failed_count = 0

    for i, combo in enumerate(combinations, 1):
        print(f"\n=== Benchmark {i}/{len(combinations)} ===")
        print(f"Model: {combo['model']}")
        print(f"Format: {combo['format']}")
        print(f"Quantization: {combo['quantization']}")
        print(f"Hardware: {combo['hardware']}")

        try:
            # Prepare model
            if not prepare_model(
                combo['model'],
                combo['format'],
                combo['quantization'],
                args.force_conversion,
                args.convert_model,
                args.quantize_model
            ):
                print("❌ Model preparation failed")
                failed_count += 1
                continue

            # Run benchmark
            results = engine.run_benchmark(
                model_name=combo['model'],
                model_format=combo['format'],
                quantization=combo['quantization'],
                hardware=combo['hardware'],
                batch_size=args.batch_size
            )

            # Check results
            if any(r.error is None for r in results):
                print("✅ Benchmark completed successfully")
                successful_count += 1
            else:
                print("❌ Benchmark failed")
                failed_count += 1

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            failed_count += 1

    print(f"\n=== Batch Benchmark Summary ===")
    print(f"Total Combinations: {len(combinations)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")

    # Save results
    if args.save_results:
        output_file = engine.save_results(args.output_file)
        print(f"\nResults saved to: {output_file}")

    return successful_count > 0


def main():
    """Main function"""
    # Set up environment
    setup_environment()

    # Parse arguments
    args = parse_arguments()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle hardware detection
    if args.detect_hardware:
        detect_hardware()
        return

    # Handle batch benchmarking
    if args.batch_benchmark:
        success = run_batch_benchmark(args)
        sys.exit(0 if success else 1)

    # Validate required arguments for single benchmark
    if not args.model:
        print("Error: --model is required for benchmarking")
        sys.exit(1)
    if not args.format:
        print("Error: --format is required for benchmarking")
        sys.exit(1)
    if not args.hardware:
        print("Error: --hardware is required for benchmarking")
        sys.exit(1)

    # Run single benchmark
    success = run_single_benchmark(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
