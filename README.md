# LLM Format Benchmarker

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)

A comprehensive benchmarking framework for Large Language Models (LLMs) using different quantization techniques and hardware backends including Intel NPU, GPU, and CPU.

## About

This project provides a unified benchmarking platform for evaluating Large Language Model performance across different model formats, quantization levels, and hardware configurations. Originally designed to leverage Intel's hardware acceleration capabilities, the framework supports comprehensive testing of model conversion pipelines from HuggingFace transformers to optimized ONNX and OpenVINO formats.

The benchmarker addresses the critical need for performance validation when deploying LLMs in production environments, helping developers make informed decisions about model optimization trade-offs between accuracy, speed, and resource consumption. With built-in hardware detection and automatic fallback mechanisms, it ensures reliable benchmarking across diverse deployment scenarios.

## Supported Models

Following models have been tested and verified on Intel AI PC platform:

- **Llama 3.2 3B**: Meta's compact language model ✅ **Tested on Intel AI PC**
  - Successfully benchmarked on Intel CPU with 4.9 tokens/sec average throughput
  - Memory usage: ~6.2GB for FP16 inference
  - Model loading time: ~6 seconds (cached)
- **Microsoft DialoGPT-small**: Small conversational model ✅ **Verified**
  - Quick testing model (351MB) for development and CI
  - Loading time: ~3 minutes, Generation: <1 second
- **Qwen 2.5**: Alibaba's latest language model (in testing)
- **LLaVA**: Large Language and Vision Assistant (in testing)

## Features

### Model Formats

- **Hugging Face (HF)**: Original PyTorch format
- **ONNX**: Open Neural Network Exchange format
- **OpenVINO (OV)**: Intel's optimized format using OpenVINO GenAI

### Quantization Support

- **4-bit quantization**: Extreme compression for memory efficiency
- **8-bit quantization**: Balanced performance and memory usage
- **FP16**: Half precision for better performance

### Hardware Backends

- **Intel NPU**: Neural Processing Unit for AI acceleration (Intel AI PC)
- **Intel GPU**: Integrated graphics (Intel Iris Xe and newer)
- **Intel CPU**: Multi-core processor support (tested on Intel Core Ultra)

### Intel AI PC Platform Testing

This framework has been specifically tested and optimized for Intel AI PC platforms:

**Tested Configuration:**

- **CPU**: Intel Core Ultra 9 288V (8 cores)
- **GPU**: Intel Integrated Graphics
- **NPU**: Intel NPU (detected and supported)
- **Memory**: 31.48 GB System RAM
- **OS**: Windows 11

**Performance Results:**

- **CPU Inference**: 4.9 tokens/sec average (Llama 3.2 3B)
- **Memory Usage**: ~6.2GB for 3B parameter models
- **Model Loading**: 6-10 seconds for cached models
- **Hardware Detection**: Full NPU/GPU/CPU detection working

## Project Structure

```
├── configs/                    # Model and hardware configurations
│   ├── config.py              # Main configuration file
│   ├── cache/                 # Model cache directory
│   └── results/               # Benchmark results
├── models/                     # Model management and conversion
│   └── model_manager.py       # Model loading and conversion
├── quantization/              # Quantization scripts and utilities
│   └── quantization_manager.py # Quantization implementation
├── benchmarking/              # Benchmarking engine
│   └── benchmark_engine.py    # Main benchmarking logic
├── inference/                 # Model inference and comparison tools
│   ├── quick_comparison.py    # Quick model comparison
│   ├── model_comparison.py    # Detailed model comparison
│   └── comparison_examples.py # Usage examples
├── utils/                     # Common utilities and helpers
│   ├── hardware_detector.py   # Hardware detection
│   └── utils.py              # Utility functions
├── examples/                  # Example scripts and tutorials
│   └── run_examples.py       # Interactive examples
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── run_benchmark.py           # Main benchmarking script
```

## Installation

### Option 1: Manual Installation

```bash
pip install -r requirements.txt
```

### Option 2: Environment Setup Scripts

```bash
# Windows
install_environment.bat

# Linux/Mac
chmod +x install_environment.sh
./install_environment.sh
```

## Usage

### Quick Start

```bash
# Run a basic benchmark
python run_benchmark.py --model qwen2.5 --format hf --quantization fp16 --hardware cpu

# Detect available hardware
python run_benchmark.py --detect-hardware

# Run interactive examples
python examples/run_examples.py
```

### Advanced Usage

```bash
# Run with custom prompts
python run_benchmark.py --model llama3.2-3b --format onnx --quantization 8bit --hardware gpu --custom-prompts "Hello, how are you?" "What is AI?"

# Run with prompts from file
python run_benchmark.py --model qwen2.5 --format ov --quantization 4bit --hardware npu --prompts-file test_prompts.txt

# Convert and quantize models
python run_benchmark.py --model llama3.2-3b --format onnx --quantization 8bit --hardware cpu --convert-model --quantize-model

# Run batch benchmarks for all combinations
python run_benchmark.py --batch-benchmark --convert-model
```

### Model Comparison

```bash
# Quick comparison between quantizations
python inference/quick_comparison.py --model qwen2.5 --prompts "Hello world" "Explain quantum computing"

# Detailed model comparison
python inference/model_comparison.py --model llama3.2-3b --formats hf onnx ov --quantizations fp16 8bit
```

### Available Commands

- `--model`: Choose from qwen2.5, llama3.2-3b, llava
- `--format`: Choose from hf, onnx, ov
- `--quantization`: Choose from 4bit, 8bit, fp16
- `--hardware`: Choose from npu, gpu, cpu

## Requirements

- Python 3.8+
- PyTorch
- ONNX Runtime
- OpenVINO Toolkit
- Transformers (Hugging Face)
- Intel Extension for PyTorch (for Intel hardware optimization)

## Hardware Compatibility

The framework automatically detects hardware capabilities and gracefully handles unsupported configurations by displaying "Not supported by hardware" instead of throwing errors.

## Project Status

For detailed information about completed benchmarks, current development progress, and future roadmap, see [project_summary.md](project_summary.md).

### Recent Achievements

- ✅ **Intel AI PC Platform**: Fully tested on Intel Core Ultra with NPU/GPU/CPU detection
- ✅ **Llama 3.2 3B Performance**: 4.9 tokens/sec on Intel CPU, 6.2GB memory usage
- ✅ **Microsoft DialoGPT-small**: Verified as quick test model (351MB)
- ✅ **Environment Setup**: Python 3.11.9 virtual environment with all dependencies
- ✅ **Hardware Detection**: NPU, Intel GPU, and CPU properly detected
- ✅ **Model Benchmarking**: Successful FP16 inference on CPU backend
- ✅ **19+ benchmark runs** completed with detailed performance metrics
- ✅ **Model comparison tools** operational with quality assessment
- ✅ **Interactive examples** providing user-friendly experience
