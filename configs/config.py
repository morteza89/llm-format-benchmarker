"""
Configuration settings for ONNX OpenVINO Benchmarking LLM Models
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    hf_model_id: str
    model_type: str  # "text-generation", "vision-language"
    max_length: int = 512
    context_length: int = 2048
    supported_quantizations: List[str] = field(default_factory=lambda: ["4bit", "8bit", "fp16"])
    special_tokens: Dict[str, str] = field(default_factory=dict)

@dataclass
class HardwareConfig:
    """Configuration for hardware backends"""
    name: str
    device_type: str  # "npu", "gpu", "cpu"
    supported_formats: List[str] = field(default_factory=lambda: ["hf", "onnx", "ov"])
    memory_limit_gb: Optional[float] = None
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking parameters"""
    warmup_runs: int = 3
    benchmark_runs: int = 10
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

# Model configurations
MODELS = {
    "qwen2.5": ModelConfig(
        name="qwen2.5",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        model_type="text-generation",
        max_length=1024,
        context_length=4096,
        special_tokens={"pad_token": "<|endoftext|>"}
    ),
    "llama3.2-3b": ModelConfig(
        name="llama3.2-3b",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_type="text-generation",
        max_length=1024,
        context_length=2048,
        special_tokens={"pad_token": "<|end_of_text|>"}
    ),
    "llava": ModelConfig(
        name="llava",
        hf_model_id="llava-hf/llava-1.5-7b-hf",
        model_type="vision-language",
        max_length=512,
        context_length=2048,
        special_tokens={"pad_token": "<unk>"}
    )
}

# Hardware configurations
HARDWARE = {
    "npu": HardwareConfig(
        name="npu",
        device_type="npu",
        supported_formats=["ov"],  # NPU typically only supports OpenVINO
        memory_limit_gb=8.0,
        batch_sizes=[1, 2, 4]
    ),
    "gpu": HardwareConfig(
        name="gpu",
        device_type="gpu",
        supported_formats=["hf", "onnx", "ov"],
        memory_limit_gb=16.0,
        batch_sizes=[1, 2, 4, 8]
    ),
    "cpu": HardwareConfig(
        name="cpu",
        device_type="cpu",
        supported_formats=["hf", "onnx", "ov"],
        memory_limit_gb=32.0,
        batch_sizes=[1, 2, 4, 8, 16]
    )
}

# Benchmark configuration
BENCHMARK = BenchmarkConfig()

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Quantization configurations
QUANTIZATION_CONFIGS = {
    "4bit": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    },
    "8bit": {
        "load_in_8bit": True,
        "device_map": "auto"
    },
    "fp16": {
        "torch_dtype": "float16",
        "device_map": "auto"
    }
}

# Test prompts for different model types
TEST_PROMPTS = {
    "text-generation": [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the process of photosynthesis.",
        "What are the benefits of renewable energy?"
    ],
    "vision-language": [
        "Describe what you see in this image.",
        "What is the main subject of this picture?",
        "Can you identify any objects in this image?",
        "What colors are prominent in this image?",
        "What is happening in this scene?"
    ]
}

# Default environment variables
DEFAULT_ENV = {
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": str(CACHE_DIR),
    "HF_HOME": str(CACHE_DIR / "huggingface"),
    "OPENVINO_LOG_LEVEL": "WARNING"
}
