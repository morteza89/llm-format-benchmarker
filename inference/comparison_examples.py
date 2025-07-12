#!/usr/bin/env python3
"""
Simple Model Comparison Example
Demonstrates how to compare original vs quantized models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_fp16_vs_8bit_comparison():
    """Example: Compare FP16 vs 8-bit quantized model"""

    print("=== FP16 vs 8-bit Quantization Comparison ===")

    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain the concept of machine learning.",
        "Write a short story about a robot.",
        "How does solar energy work?",
        "What are the benefits of exercise?"
    ]

    print("Test prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")

    print("\nThis would compare:")
    print("  Original: llama3.2-3b (HF format, FP16)")
    print("  Quantized: llama3.2-3b (HF format, 8-bit)")

    print("\nTo run this comparison:")
    print("python inference/model_comparison.py \\")
    print("  --model llama3.2-3b \\")
    print("  --original-format hf \\")
    print("  --quantized-format hf \\")
    print("  --original-quantization fp16 \\")
    print("  --quantized-quantization 8bit \\")
    print("  --hardware cpu \\")
    print("  --prompts \"What is AI?\" \"Explain ML\" \"Write a story\"")


def run_hf_vs_onnx_comparison():
    """Example: Compare HF vs ONNX format"""

    print("\n=== HF vs ONNX Format Comparison ===")

    print("This would compare:")
    print("  Original: llama3.2-3b (HF format, FP16)")
    print("  Converted: llama3.2-3b (ONNX format, FP16)")

    print("\nTo run this comparison:")
    print("python inference/model_comparison.py \\")
    print("  --model llama3.2-3b \\")
    print("  --original-format hf \\")
    print("  --quantized-format onnx \\")
    print("  --original-quantization fp16 \\")
    print("  --quantized-quantization fp16 \\")
    print("  --hardware cpu \\")
    print("  --prompts-file test_prompts.txt")


def run_hf_vs_openvino_comparison():
    """Example: Compare HF vs OpenVINO format"""

    print("\n=== HF vs OpenVINO Format Comparison ===")

    print("This would compare:")
    print("  Original: llama3.2-3b (HF format, FP16)")
    print("  Converted: llama3.2-3b (OpenVINO format, FP16)")

    print("\nTo run this comparison:")
    print("python inference/model_comparison.py \\")
    print("  --model llama3.2-3b \\")
    print("  --original-format hf \\")
    print("  --quantized-format ov \\")
    print("  --original-quantization fp16 \\")
    print("  --quantized-quantization fp16 \\")
    print("  --hardware npu \\")
    print("  --max-new-tokens 50")


def run_onnx_vs_openvino_comparison():
    """Example: Compare ONNX vs OpenVINO format with same precision"""

    print("\n=== ONNX vs OpenVINO Format Comparison ===")

    print("Option 1 - Both FP16 precision:")
    print("  Original: llama3.2-3b (ONNX format, FP16)")
    print("  Converted: llama3.2-3b (OpenVINO format, FP16)")
    print("  Command:")
    print("  python inference/model_comparison.py \\")
    print("    --model llama3.2-3b \\")
    print("    --original-format onnx \\")
    print("    --quantized-format ov \\")
    print("    --original-quantization fp16 \\")
    print("    --quantized-quantization fp16 \\")
    print("    --hardware npu \\")
    print("    --prompts \"Compare FP16 formats\" \"Test conversion quality\"")

    print("\nOption 2 - Both 8-bit precision:")
    print("  Original: llama3.2-3b (ONNX format, 8-bit)")
    print("  Converted: llama3.2-3b (OpenVINO format, 8-bit)")
    print("  Command:")
    print("  python inference/model_comparison.py \\")
    print("    --model llama3.2-3b \\")
    print("    --original-format onnx \\")
    print("    --quantized-format ov \\")
    print("    --original-quantization 8bit \\")
    print("    --quantized-quantization 8bit \\")
    print("    --hardware npu \\")
    print("    --prompts \"Compare 8-bit formats\" \"Test quantized conversion\"")

    print("\nOption 3 - Both 4-bit precision:")
    print("  Original: llama3.2-3b (ONNX format, 4-bit)")
    print("  Converted: llama3.2-3b (OpenVINO format, 4-bit)")
    print("  Command:")
    print("  python inference/model_comparison.py \\")
    print("    --model llama3.2-3b \\")
    print("    --original-format onnx \\")
    print("    --quantized-format ov \\")
    print("    --original-quantization 4bit \\")
    print("    --quantized-quantization 4bit \\")
    print("    --hardware npu \\")
    print("    --prompts \"Compare 4-bit formats\" \"Test aggressive quantization\"")

    print("\nBenefits of same-precision comparison:")
    print("  - Isolates format optimization effects (no quantization differences)")
    print("  - Tests pure conversion pipeline: HF → ONNX → OpenVINO")
    print("  - Measures OpenVINO optimization gains vs cross-platform ONNX")
    print("  - Validates quality preservation across format conversions")

    print("\nIntel Hardware Optimization:")
    print("  - ONNX models automatically use Intel-optimized execution providers")
    print("  - CPU: Uses OpenVINOExecutionProvider when available")
    print("  - GPU: Uses OpenVINOExecutionProvider or DmlExecutionProvider")
    print("  - NPU: Uses OpenVINOExecutionProvider for best performance")
    print("  - Automatic fallback to CPUExecutionProvider if specialized providers unavailable")


def create_test_prompts_file():
    """Create a sample prompts file"""

    prompts = [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the process of photosynthesis.",
        "What are the benefits of renewable energy?",
        "How does artificial intelligence work?",
        "What is the future of space exploration?",
        "Explain the concept of blockchain technology.",
        "What are the ethical implications of AI?",
        "How can we address climate change?"
    ]

    prompts_file = project_root / "test_prompts.txt"

    with open(prompts_file, 'w', encoding='utf-8') as f:
        f.write("# Test prompts for model comparison\n")
        f.write("# One prompt per line\n")
        f.write("# Lines starting with # are ignored\n\n")

        for prompt in prompts:
            f.write(f"{prompt}\n")

    print(f"\nCreated test prompts file: {prompts_file}")
    print(f"Contains {len(prompts)} test prompts")


def main():
    """Main function to demonstrate usage"""

    print("Model Comparison Tool - Usage Examples")
    print("=" * 50)

    # Show different comparison scenarios
    run_fp16_vs_8bit_comparison()
    run_hf_vs_onnx_comparison()
    run_hf_vs_openvino_comparison()
    run_onnx_vs_openvino_comparison()

    # Create sample prompts file
    create_test_prompts_file()

    print("\n" + "=" * 50)
    print("Additional Options:")
    print("  --verbose              Enable detailed logging")
    print("  --output-file results.json    Save results to specific file")
    print("  --max-new-tokens 200   Generate longer outputs")
    print("  --hardware gpu         Run on GPU instead of CPU")

    print("\nExpected Output:")
    print("  - Similarity scores between original and quantized outputs")
    print("  - Performance comparison (latency)")
    print("  - Quality assessment (PASS/CAUTION/FAIL)")
    print("  - Detailed JSON results file")

    print("\nInterpretation:")
    print("  - Similarity ≥ 0.8: High quality preservation")
    print("  - Similarity 0.6-0.8: Moderate quality degradation")
    print("  - Similarity < 0.6: Significant quality loss")


if __name__ == "__main__":
    main()
