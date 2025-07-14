#!/usr/bin/env python3
"""
Test PyTorch native quantization for Intel GPU/NPU
"""

import torch
import torch.quantization as quant
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os


def test_pytorch_quantization():
    """Test PyTorch dynamic quantization"""

    print("=== PyTorch Native Quantization Test ===")

    # Load model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"✅ Model loaded successfully")
        print(f"Model device: {next(model.parameters()).device}")

        # Test original model
        prompt = "What is artificial intelligence?"
        inputs = tokenizer(prompt, return_tensors="pt")

        print("\n--- Testing Original Model ---")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        original_time = time.time() - start_time
        original_response = tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        print(f"Time: {original_time:.2f}s")
        print(f"Response: {original_response}")

        # Apply dynamic quantization
        print("\n--- Applying Dynamic Quantization ---")
        try:
            quantized_model = quant.quantize_dynamic(
                model.cpu(),  # Move to CPU for quantization
                {torch.nn.Linear},  # Quantize Linear layers
                dtype=torch.qint8
            )
            print("✅ Dynamic quantization successful")

            # Test quantized model
            print("\n--- Testing Quantized Model ---")
            inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs_quant = quantized_model.generate(
                    inputs_cpu["input_ids"],
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            quantized_time = time.time() - start_time
            quantized_response = tokenizer.decode(
                outputs_quant[0], skip_special_tokens=True)

            print(f"Time: {quantized_time:.2f}s")
            print(f"Response: {quantized_response}")

            # Compare results
            print("\n--- Comparison ---")
            print(f"Original time: {original_time:.2f}s")
            print(f"Quantized time: {quantized_time:.2f}s")
            print(f"Speedup: {original_time/quantized_time:.2f}x")

        except Exception as e:
            print(f"❌ Dynamic quantization failed: {e}")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")


if __name__ == "__main__":
    test_pytorch_quantization()
