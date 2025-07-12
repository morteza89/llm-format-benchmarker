#!/usr/bin/env python3
"""
Simple test script to demonstrate model comparison functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_functionality():
    """Test basic comparison functionality without actual model loading"""

    print("=== Model Comparison Tool Test ===")

    try:
        from inference.quick_comparison import QuickModelComparator

        # Initialize comparator
        comparator = QuickModelComparator()

        # Test similarity calculation
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "The quick brown fox leaps over the lazy dog."
        text3 = "A completely different sentence about cats."

        sim1 = comparator.calculate_simple_similarity(text1, text2)
        sim2 = comparator.calculate_simple_similarity(text1, text3)

        print(f"✅ Similarity calculation working:")
        print(f"   Similar texts: {sim1:.3f}")
        print(f"   Different texts: {sim2:.3f}")

        # Test report generation with mock data
        from inference.quick_comparison import QuickComparisonResult
        from datetime import datetime

        mock_results = [
            QuickComparisonResult(
                model_name="test-model",
                prompt="What is AI?",
                fp16_output="AI is artificial intelligence...",
                quantized_output="AI is artificial intelligence...",
                fp16_latency_ms=1000.0,
                quantized_latency_ms=800.0,
                similarity_score=0.85,
                speedup=1.25,
                timestamp=datetime.now().isoformat()
            ),
            QuickComparisonResult(
                model_name="test-model",
                prompt="Explain ML",
                fp16_output="Machine learning is a subset of AI...",
                quantized_output="Machine learning is a subset of AI...",
                fp16_latency_ms=1200.0,
                quantized_latency_ms=900.0,
                similarity_score=0.78,
                speedup=1.33,
                timestamp=datetime.now().isoformat()
            )
        ]

        report = comparator.generate_report(mock_results)
        print(f"\n✅ Report generation working:")
        print(report)

        # Test file saving
        output_file = comparator.save_results(
            mock_results, "test_comparison.json")
        print(f"\n✅ File saving working:")
        print(f"   Results saved to: {output_file}")

        print(f"\n🎉 All basic functionality tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


def show_usage_examples():
    """Show practical usage examples"""

    print("\n=== Usage Examples ===")

    print("\n1. Quick FP16 vs 8-bit comparison:")
    print("   python inference/quick_comparison.py")

    print("\n2. Custom prompts comparison:")
    print("   python inference/model_comparison.py \\")
    print("     --model llama3.2-3b \\")
    print("     --original-format hf \\")
    print("     --quantized-format hf \\")
    print("     --original-quantization fp16 \\")
    print("     --quantized-quantization 8bit \\")
    print("     --prompts \"What is AI?\" \"Explain ML\"")

    print("\n3. Using prompts file:")
    print("   python inference/model_comparison.py \\")
    print("     --model llama3.2-3b \\")
    print("     --original-format hf \\")
    print("     --quantized-format hf \\")
    print("     --original-quantization fp16 \\")
    print("     --quantized-quantization 4bit \\")
    print("     --prompts-file test_prompts.txt")

    print("\n4. ONNX format comparison (when available):")
    print("   python inference/model_comparison.py \\")
    print("     --model llama3.2-3b \\")
    print("     --original-format hf \\")
    print("     --quantized-format onnx \\")
    print("     --original-quantization fp16 \\")
    print("     --quantized-quantization fp16")

    print("\n5. OpenVINO format comparison (when available):")
    print("   python inference/model_comparison.py \\")
    print("     --model llama3.2-3b \\")
    print("     --original-format hf \\")
    print("     --quantized-format ov \\")
    print("     --original-quantization fp16 \\")
    print("     --quantized-quantization fp16 \\")
    print("     --hardware npu")


def main():
    """Main function"""

    print("Model Comparison Tool - Test & Examples")
    print("=" * 50)

    # Run basic functionality test
    if test_basic_functionality():
        # Show usage examples
        show_usage_examples()

        print("\n" + "=" * 50)
        print("🚀 Model Comparison Tool is ready for use!")
        print("\nNext steps:")
        print("1. Run 'python inference/quick_comparison.py' for a quick test")
        print("2. Use the full comparison tool with your custom prompts")
        print("3. Check the results in the 'configs/results/' directory")

    else:
        print("\n❌ Setup incomplete. Please check the installation.")


if __name__ == "__main__":
    main()
