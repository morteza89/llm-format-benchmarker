#!/usr/bin/env python3
"""
Test script for Intel GPU quantization using PyTorch native quantization
"""

import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_intel_gpu_quantization():
    print("=== Intel GPU Quantization Test ===")

    # Check available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"Available devices: {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU only'}")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model in FP32 for quantization compatibility
        print("Loading FP32 model...")
        model_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        memory_fp32 = get_memory_usage()
        print(f"FP32 model memory usage: {memory_fp32:.1f} MB")

        # Test FP32 inference
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt")

        print("\n=== FP32 Inference Test ===")
        start_time = time.time()

        with torch.no_grad():
            outputs = model_fp32.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        fp32_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"FP32 Response: {response}")
        print(f"FP32 Time: {fp32_time:.2f}s")
        print(f"FP32 Tokens/sec: {10 / fp32_time:.2f}")

        # Try dynamic quantization (8-bit)
        print("\n=== Dynamic Quantization (8-bit) Test ===")
        try:
            model_int8 = torch.quantization.quantize_dynamic(
                model_fp32,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

            memory_int8 = get_memory_usage()
            print(f"INT8 model memory usage: {memory_int8:.1f} MB")
            print(
                f"Memory reduction: {((memory_fp32 - memory_int8) / memory_fp32 * 100):.1f}%")

            # Test INT8 inference
            start_time = time.time()

            with torch.no_grad():
                outputs_int8 = model_int8.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            int8_time = time.time() - start_time
            response_int8 = tokenizer.decode(
                outputs_int8[0], skip_special_tokens=True)
            print(f"INT8 Response: {response_int8}")
            print(f"INT8 Time: {int8_time:.2f}s")
            print(f"INT8 Tokens/sec: {10 / int8_time:.2f}")
            print(f"Speed improvement: {(fp32_time / int8_time):.2f}x")

        except Exception as e:
            print(f"❌ Dynamic quantization failed: {e}")

        print("\n=== Test Results Summary ===")
        print(f"✅ FP32 model loaded successfully")
        print(f"✅ FP32 inference: {10 / fp32_time:.2f} tokens/sec")
        print(f"✅ Memory usage: {memory_fp32:.1f} MB")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")


if __name__ == "__main__":
    test_intel_gpu_quantization()
