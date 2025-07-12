"""
Benchmarking engine for different model formats and hardware backends
"""

import os
import time
import logging
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    from openvino_genai import LLMPipeline
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from configs.config import MODELS, HARDWARE, BENCHMARK, TEST_PROMPTS, RESULTS_DIR, MODELS_DIR
from utils.hardware_detector import HardwareDetector
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structure for benchmark results"""
    model_name: str
    model_format: str
    quantization: str
    hardware: str
    batch_size: int
    prompt: str
    tokens_generated: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_used_mb: float
    gpu_memory_used_mb: float
    timestamp: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BenchmarkEngine:
    """Main benchmarking engine"""

    def __init__(self):
        self.hardware_detector = HardwareDetector()
        self.model_manager = ModelManager()
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        model_name: str,
        model_format: str,
        quantization: str,
        hardware: str,
        batch_size: int = 1,
        custom_prompts: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for specified configuration

        Args:
            model_name: Name of the model to benchmark
            model_format: Format of the model (hf, onnx, ov)
            quantization: Quantization type (4bit, 8bit, fp16)
            hardware: Hardware backend (cpu, gpu, npu)
            batch_size: Batch size for inference
            custom_prompts: Custom prompts to use for testing

        Returns:
            List of benchmark results
        """

        # Validation
        if model_name not in MODELS:
            error_msg = f"Model {model_name} not supported. Available: {list(MODELS.keys())}"
            logger.error(error_msg)
            return [self._create_error_result(model_name, model_format, quantization, hardware, batch_size, error_msg)]

        if hardware not in HARDWARE:
            error_msg = f"Hardware {hardware} not supported. Available: {list(HARDWARE.keys())}"
            logger.error(error_msg)
            return [self._create_error_result(model_name, model_format, quantization, hardware, batch_size, error_msg)]

        # Check hardware support
        if not self.hardware_detector.is_hardware_supported(hardware):
            error_msg = f"Not supported by hardware: {hardware}"
            logger.warning(error_msg)
            return [self._create_error_result(model_name, model_format, quantization, hardware, batch_size, error_msg)]

        # Check format support for hardware
        supported_formats = self.hardware_detector.get_supported_formats(
            hardware)
        if model_format not in supported_formats:
            error_msg = f"Format {model_format} not supported by hardware {hardware}. Supported: {supported_formats}"
            logger.warning(error_msg)
            return [self._create_error_result(model_name, model_format, quantization, hardware, batch_size, error_msg)]

        # Get test prompts
        model_config = MODELS[model_name]
        if custom_prompts:
            test_prompts = custom_prompts
        else:
            test_prompts = TEST_PROMPTS.get(
                model_config.model_type, TEST_PROMPTS["text-generation"])

        # Run benchmarks
        results = []

        for prompt in test_prompts:
            try:
                result = self._run_single_benchmark(
                    model_name, model_format, quantization, hardware, batch_size, prompt
                )
                results.append(result)

                # Clear memory between runs
                self._clear_memory()

            except Exception as e:
                error_msg = f"Benchmark failed: {str(e)}"
                logger.error(error_msg)
                results.append(self._create_error_result(
                    model_name, model_format, quantization, hardware, batch_size, error_msg, prompt
                ))

        self.results.extend(results)
        return results

    def _run_single_benchmark(
        self,
        model_name: str,
        model_format: str,
        quantization: str,
        hardware: str,
        batch_size: int,
        prompt: str
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration"""

        # Record initial memory
        initial_memory = self._get_memory_usage()

        # Load model based on format
        if model_format == "hf":
            return self._benchmark_huggingface(
                model_name, quantization, hardware, batch_size, prompt, initial_memory
            )
        elif model_format == "onnx":
            return self._benchmark_onnx(
                model_name, quantization, hardware, batch_size, prompt, initial_memory
            )
        elif model_format == "ov":
            return self._benchmark_openvino(
                model_name, quantization, hardware, batch_size, prompt, initial_memory
            )
        else:
            raise ValueError(f"Unsupported format: {model_format}")

    def _benchmark_huggingface(
        self,
        model_name: str,
        quantization: str,
        hardware: str,
        batch_size: int,
        prompt: str,
        initial_memory: Dict[str, float]
    ) -> BenchmarkResult:
        """Benchmark Hugging Face model"""

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for HF benchmarking")

        # Load model
        model, tokenizer = self.model_manager.load_huggingface_model(
            model_name, quantization, device_map=self._get_device_map(hardware)
        )

        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODELS[model_name].max_length
        )

        # Move to device
        device = self._get_torch_device(hardware)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Warmup
        for _ in range(BENCHMARK.warmup_runs):
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Benchmark
        start_time = time.time()
        tokens_generated = 0

        for _ in range(BENCHMARK.benchmark_runs):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=BENCHMARK.max_new_tokens,
                    temperature=BENCHMARK.temperature,
                    top_p=BENCHMARK.top_p,
                    do_sample=BENCHMARK.do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )

                tokens_generated += outputs.shape[1] - \
                    inputs["input_ids"].shape[1]

        end_time = time.time()

        # Calculate metrics
        latency_ms = ((end_time - start_time) /
                      BENCHMARK.benchmark_runs) * 1000
        throughput = (tokens_generated /
                      BENCHMARK.benchmark_runs) / (latency_ms / 1000)

        # Memory usage
        final_memory = self._get_memory_usage()
        memory_used = final_memory["ram"] - initial_memory["ram"]
        gpu_memory_used = final_memory["gpu"] - initial_memory["gpu"]

        return BenchmarkResult(
            model_name=model_name,
            model_format="hf",
            quantization=quantization,
            hardware=hardware,
            batch_size=batch_size,
            prompt=prompt,
            tokens_generated=tokens_generated // BENCHMARK.benchmark_runs,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_used_mb=memory_used,
            gpu_memory_used_mb=gpu_memory_used,
            timestamp=datetime.now().isoformat()
        )

    def _benchmark_onnx(
        self,
        model_name: str,
        quantization: str,
        hardware: str,
        batch_size: int,
        prompt: str,
        initial_memory: Dict[str, float]
    ) -> BenchmarkResult:
        """Benchmark ONNX model"""

        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available for benchmarking")

        # Get model path
        model_path = self._get_model_path(model_name, "onnx", quantization)

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load model
        providers = self._get_onnx_providers(hardware)
        session = ort.InferenceSession(str(model_path), providers=providers)

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODELS[model_name].hf_model_id)

        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=MODELS[model_name].max_length
        )

        # Warmup
        for _ in range(BENCHMARK.warmup_runs):
            _ = session.run(None, dict(inputs))

        # Benchmark
        start_time = time.time()

        for _ in range(BENCHMARK.benchmark_runs):
            outputs = session.run(None, dict(inputs))

        end_time = time.time()

        # Calculate metrics (simplified for ONNX)
        latency_ms = ((end_time - start_time) /
                      BENCHMARK.benchmark_runs) * 1000
        tokens_generated = BENCHMARK.max_new_tokens  # Approximation
        throughput = tokens_generated / (latency_ms / 1000)

        # Memory usage
        final_memory = self._get_memory_usage()
        memory_used = final_memory["ram"] - initial_memory["ram"]
        gpu_memory_used = final_memory["gpu"] - initial_memory["gpu"]

        return BenchmarkResult(
            model_name=model_name,
            model_format="onnx",
            quantization=quantization,
            hardware=hardware,
            batch_size=batch_size,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_used_mb=memory_used,
            gpu_memory_used_mb=gpu_memory_used,
            timestamp=datetime.now().isoformat()
        )

    def _benchmark_openvino(
        self,
        model_name: str,
        quantization: str,
        hardware: str,
        batch_size: int,
        prompt: str,
        initial_memory: Dict[str, float]
    ) -> BenchmarkResult:
        """Benchmark OpenVINO model"""

        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO not available for benchmarking")

        # Get model path
        model_path = self._get_model_path(model_name, "ov", quantization)

        if not model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {model_path}")

        # Load model
        device = self._get_openvino_device(hardware)
        pipe = LLMPipeline(str(model_path), device)

        # Warmup
        for _ in range(BENCHMARK.warmup_runs):
            _ = pipe.generate(prompt, max_new_tokens=32)

        # Benchmark
        start_time = time.time()
        total_tokens = 0

        for _ in range(BENCHMARK.benchmark_runs):
            result = pipe.generate(
                prompt,
                max_new_tokens=BENCHMARK.max_new_tokens,
                temperature=BENCHMARK.temperature,
                top_p=BENCHMARK.top_p,
                do_sample=BENCHMARK.do_sample
            )

            # Count tokens (approximation)
            total_tokens += len(result.split())

        end_time = time.time()

        # Calculate metrics
        latency_ms = ((end_time - start_time) /
                      BENCHMARK.benchmark_runs) * 1000
        tokens_generated = total_tokens // BENCHMARK.benchmark_runs
        throughput = tokens_generated / (latency_ms / 1000)

        # Memory usage
        final_memory = self._get_memory_usage()
        memory_used = final_memory["ram"] - initial_memory["ram"]
        gpu_memory_used = final_memory["gpu"] - initial_memory["gpu"]

        return BenchmarkResult(
            model_name=model_name,
            model_format="ov",
            quantization=quantization,
            hardware=hardware,
            batch_size=batch_size,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_used_mb=memory_used,
            gpu_memory_used_mb=gpu_memory_used,
            timestamp=datetime.now().isoformat()
        )

    def _get_model_path(self, model_name: str, format_type: str, quantization: str) -> Path:
        """Get path to model file"""

        if format_type == "hf":
            return MODELS_DIR / f"{model_name}_{quantization}_hf"
        elif format_type == "onnx":
            return MODELS_DIR / f"{model_name}_{quantization}.onnx"
        elif format_type == "ov":
            return MODELS_DIR / f"{model_name}_{quantization}_ov"
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _get_device_map(self, hardware: str) -> str:
        """Get device map for HF models"""

        if hardware == "cpu":
            return "cpu"
        elif hardware == "gpu":
            return "auto"
        elif hardware == "npu":
            return "cpu"  # NPU not directly supported by HF
        else:
            return "auto"

    def _get_torch_device(self, hardware: str) -> str:
        """Get PyTorch device"""

        if hardware == "cpu":
            return "cpu"
        elif hardware == "gpu" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _get_onnx_providers(self, hardware: str) -> List[str]:
        """Get ONNX Runtime providers"""

        if hardware == "cpu":
            return ["CPUExecutionProvider"]
        elif hardware == "gpu":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif hardware == "npu":
            # NPU provider might not be available
            return ["CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def _get_openvino_device(self, hardware: str) -> str:
        """Get OpenVINO device"""

        if hardware == "cpu":
            return "CPU"
        elif hardware == "gpu":
            return "GPU"
        elif hardware == "npu":
            return "NPU"
        else:
            return "CPU"

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""

        # RAM usage
        ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB

        # GPU usage
        gpu_usage = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        return {"ram": ram_usage, "gpu": gpu_usage}

    def _clear_memory(self):
        """Clear memory caches"""

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_error_result(
        self,
        model_name: str,
        model_format: str,
        quantization: str,
        hardware: str,
        batch_size: int,
        error_msg: str,
        prompt: str = "N/A"
    ) -> BenchmarkResult:
        """Create error result"""

        return BenchmarkResult(
            model_name=model_name,
            model_format=model_format,
            quantization=quantization,
            hardware=hardware,
            batch_size=batch_size,
            prompt=prompt,
            tokens_generated=0,
            latency_ms=0.0,
            throughput_tokens_per_sec=0.0,
            memory_used_mb=0.0,
            gpu_memory_used_mb=0.0,
            timestamp=datetime.now().isoformat(),
            error=error_msg
        )

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        results_path = RESULTS_DIR / filename

        # Convert results to dict format
        results_dict = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hardware_info": {
                    "cpu": self.hardware_detector.cpu_info,
                    "gpu": self.hardware_detector.gpu_info,
                    "npu": self.hardware_detector.npu_info,
                    "system": self.hardware_detector.system_info
                }
            },
            "results": [result.to_dict() for result in self.results]
        }

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to: {results_path}")
        return str(results_path)

    def load_results(self, filename: str) -> List[BenchmarkResult]:
        """Load benchmark results from file"""

        results_path = RESULTS_DIR / filename

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, 'r') as f:
            data = json.load(f)

        # Convert to BenchmarkResult objects
        results = []
        for result_data in data.get("results", []):
            result = BenchmarkResult(**result_data)
            results.append(result)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results"""

        if not self.results:
            return {"error": "No results available"}

        summary = {
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len([r for r in self.results if r.error is None]),
            "failed_benchmarks": len([r for r in self.results if r.error is not None]),
            "models_tested": list(set(r.model_name for r in self.results)),
            "formats_tested": list(set(r.model_format for r in self.results)),
            "hardware_tested": list(set(r.hardware for r in self.results)),
            "quantizations_tested": list(set(r.quantization for r in self.results))
        }

        # Calculate averages for successful benchmarks
        successful_results = [r for r in self.results if r.error is None]

        if successful_results:
            summary["average_latency_ms"] = np.mean(
                [r.latency_ms for r in successful_results])
            summary["average_throughput"] = np.mean(
                [r.throughput_tokens_per_sec for r in successful_results])
            summary["average_memory_mb"] = np.mean(
                [r.memory_used_mb for r in successful_results])
            summary["average_gpu_memory_mb"] = np.mean(
                [r.gpu_memory_used_mb for r in successful_results])

        return summary


def get_benchmark_engine() -> BenchmarkEngine:
    """Get benchmark engine instance"""
    return BenchmarkEngine()


if __name__ == "__main__":
    # Test benchmarking
    engine = BenchmarkEngine()

    # Print hardware summary
    engine.hardware_detector.print_hardware_summary()

    # Test a simple benchmark
    print("\n=== Testing CPU benchmark ===")
    try:
        results = engine.run_benchmark(
            model_name="llama3.2-3b",
            model_format="hf",
            quantization="fp16",
            hardware="cpu",
            batch_size=1,
            custom_prompts=["Hello, how are you?"]
        )

        for result in results:
            if result.error:
                print(f"Error: {result.error}")
            else:
                print(f"Latency: {result.latency_ms:.1f}ms")
                print(
                    f"Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec")

    except Exception as e:
        print(f"Benchmark test failed: {e}")

    # Print summary
    summary = engine.get_summary()
    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))
