#!/usr/bin/env python3
"""
Intel GPU Testing Script
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_intel_gpu_inference():
    print("=== Intel GPU Testing ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check MPS (Metal Performance Shaders) - Apple's GPU backend
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    # Check available devices
    print("Available devices:")
    print(f"- CPU: {torch.device('cpu')}")

    try:
        print("\nTesting model loading and inference...")
        start_time = time.time()

        # Load small model for testing
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")

        # Test inference
        gen_start = time.time()
        inputs = tokenizer("Hello", return_tensors="pt")
        print(f"Input device: {inputs.input_ids.device}")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - gen_start

        print(f"Generation completed in {gen_time:.3f} seconds")
        print(f"Result: {result}")

        # Memory info
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"System memory usage: {memory_info.percent}%")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_intel_gpu_inference()
