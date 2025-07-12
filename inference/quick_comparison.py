#!/usr/bin/env python3
"""
Practical Model Comparison Tool
Compares HF model outputs with different quantizations
"""

from configs.config import MODELS, RESULTS_DIR
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import sys
from pathlib import Path
import logging
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuickComparisonResult:
    """Simple comparison result structure"""
    model_name: str
    prompt: str
    fp16_output: str
    quantized_output: str
    fp16_latency_ms: float
    quantized_latency_ms: float
    similarity_score: float
    speedup: float
    timestamp: str
    error: Optional[str] = None


class QuickModelComparator:
    """Simple model comparator for HF models"""

    def __init__(self):
        self.results = []

    def calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Simple word-level comparison
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def generate_text(self, model, tokenizer, prompt: str, max_new_tokens: int = 100) -> tuple[str, float]:
        """Generate text and measure latency"""
        try:
            start_time = time.time()

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )

            # Decode
            generated_text = tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            # Remove original prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            return generated_text, latency_ms

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return "", 0.0

    def compare_fp16_vs_quantized(self, model_name: str, prompts: List[str],
                                  quantization: str = "8bit", max_new_tokens: int = 100) -> List[QuickComparisonResult]:
        """Compare FP16 vs quantized model"""

        results = []

        if model_name not in MODELS:
            logger.error(f"Model {model_name} not found in config")
            return results

        model_config = MODELS[model_name]
        hf_model_id = model_config.hf_model_id

        logger.info(f"Loading models for {model_name}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load FP16 model
            logger.info("Loading FP16 model...")
            fp16_model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # Load quantized model
            logger.info(f"Loading {quantization} quantized model...")
            if quantization == "8bit":
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                logger.error(f"Unsupported quantization: {quantization}")
                return results

            # Run comparisons
            for i, prompt in enumerate(prompts, 1):
                logger.info(
                    f"Comparing prompt {i}/{len(prompts)}: {prompt[:50]}...")

                try:
                    # Generate with FP16 model
                    fp16_output, fp16_latency = self.generate_text(
                        fp16_model, tokenizer, prompt, max_new_tokens
                    )

                    # Generate with quantized model
                    quantized_output, quantized_latency = self.generate_text(
                        quantized_model, tokenizer, prompt, max_new_tokens
                    )

                    # Calculate similarity
                    similarity = self.calculate_simple_similarity(
                        fp16_output, quantized_output)

                    # Calculate speedup
                    speedup = fp16_latency / quantized_latency if quantized_latency > 0 else 0.0

                    # Create result
                    result = QuickComparisonResult(
                        model_name=model_name,
                        prompt=prompt,
                        fp16_output=fp16_output,
                        quantized_output=quantized_output,
                        fp16_latency_ms=fp16_latency,
                        quantized_latency_ms=quantized_latency,
                        similarity_score=similarity,
                        speedup=speedup,
                        timestamp=datetime.now().isoformat()
                    )

                    results.append(result)

                    # Log result
                    logger.info(f"  Similarity: {similarity:.3f}")
                    logger.info(f"  Speedup: {speedup:.2f}x")
                    logger.info(f"  FP16 latency: {fp16_latency:.1f}ms")
                    logger.info(
                        f"  {quantization} latency: {quantized_latency:.1f}ms")

                except Exception as e:
                    logger.error(f"Comparison failed for prompt: {e}")

                    result = QuickComparisonResult(
                        model_name=model_name,
                        prompt=prompt,
                        fp16_output="",
                        quantized_output="",
                        fp16_latency_ms=0.0,
                        quantized_latency_ms=0.0,
                        similarity_score=0.0,
                        speedup=0.0,
                        timestamp=datetime.now().isoformat(),
                        error=str(e)
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return results

        return results

    def generate_report(self, results: List[QuickComparisonResult]) -> str:
        """Generate comparison report"""
        if not results:
            return "No comparison results available."

        successful_results = [r for r in results if r.error is None]

        if not successful_results:
            return "All comparisons failed."

        # Calculate statistics
        similarities = [r.similarity_score for r in successful_results]
        speedups = [r.speedup for r in successful_results if r.speedup > 0]

        avg_similarity = sum(similarities) / len(similarities)
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0

        # Generate report
        report = f"""
=== Model Comparison Report ===
Model: {results[0].model_name}
Total Comparisons: {len(results)}
Successful: {len(successful_results)}
Failed: {len(results) - len(successful_results)}

=== Quality Analysis ===
Average Similarity: {avg_similarity:.3f}
High Quality (>0.7): {sum(1 for s in similarities if s > 0.7)}/{len(similarities)} ({sum(1 for s in similarities if s > 0.7)/len(similarities)*100:.1f}%)
Medium Quality (0.5-0.7): {sum(1 for s in similarities if 0.5 <= s <= 0.7)}/{len(similarities)} ({sum(1 for s in similarities if 0.5 <= s <= 0.7)/len(similarities)*100:.1f}%)
Low Quality (<0.5): {sum(1 for s in similarities if s < 0.5)}/{len(similarities)} ({sum(1 for s in similarities if s < 0.5)/len(similarities)*100:.1f}%)

=== Performance Analysis ===
Average Speedup: {avg_speedup:.2f}x

=== Sample Outputs ===
"""

        # Show first few comparisons
        for i, result in enumerate(successful_results[:3], 1):
            report += f"\nExample {i}:\n"
            report += f"Prompt: {result.prompt}\n"
            report += f"FP16 Output: {result.fp16_output[:100]}...\n"
            report += f"Quantized Output: {result.quantized_output[:100]}...\n"
            report += f"Similarity: {result.similarity_score:.3f}\n"

        # Quality assessment
        if avg_similarity >= 0.7:
            report += "\n✅ PASS: Quantized model maintains good quality"
        elif avg_similarity >= 0.5:
            report += "\n⚠️  CAUTION: Quantized model has moderate quality degradation"
        else:
            report += "\n❌ FAIL: Quantized model has significant quality degradation"

        return report

    def save_results(self, results: List[QuickComparisonResult], output_file: Optional[str] = None) -> str:
        """Save results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_comparison_{timestamp}.json"

        output_path = RESULTS_DIR / output_file

        # Convert to dict
        results_dict = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_comparisons": len(results),
                "successful_comparisons": len([r for r in results if r.error is None])
            },
            "results": [asdict(result) for result in results]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        return str(output_path)


def main():
    """Main function for quick testing"""

    print("=== Quick Model Comparison Tool ===")

    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How does solar energy work?",
        "Write a short story about a robot."
    ]

    print("Test prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")

    # Run comparison
    comparator = QuickModelComparator()

    print("\nRunning FP16 vs 8-bit comparison...")
    print("This may take a few minutes...")

    try:
        results = comparator.compare_fp16_vs_quantized(
            model_name="llama3.2-3b",
            prompts=test_prompts,
            quantization="8bit",
            max_new_tokens=50
        )

        # Generate report
        report = comparator.generate_report(results)
        print(report)

        # Save results
        output_file = comparator.save_results(results)
        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Comparison failed: {e}")
        print("This might be due to quantization library compatibility issues.")
        print("The comparison tool is ready for use when quantization is working.")


if __name__ == "__main__":
    main()
