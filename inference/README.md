# Model Inference and Comparison Tools

This folder contains tools for comparing model outputs between different formats and quantizations to ensure quality is maintained.

## Files

### `quick_comparison.py`

- **Purpose**: Compare FP16 vs quantized HF models
- **Features**:
  - Side-by-side output comparison
  - Similarity scoring
  - Performance analysis
  - Quality assessment (PASS/CAUTION/FAIL)

### `model_comparison.py`

- **Purpose**: Full comparison tool (supports HF/ONNX/OpenVINO)
- **Features**:
  - Multi-format comparison
  - Advanced similarity metrics
  - Detailed reporting
  - JSON output

### `comparison_examples.py`

- **Purpose**: Usage examples and demonstrations
- **Features**:
  - Example command lines
  - Sample prompts file generation
  - Best practices

## Quick Usage

### Compare FP16 vs 8-bit quantization:

```bash
python inference/quick_comparison.py
```

### Compare original vs converted models:

```bash
python inference/model_comparison.py \
  --model llama3.2-3b \
  --original-format hf \
  --quantized-format onnx \
  --original-quantization fp16 \
  --quantized-quantization fp16 \
  --prompts "What is AI?" "Explain machine learning"
```

### Use custom prompts file:

```bash
python inference/model_comparison.py \
  --model llama3.2-3b \
  --original-format hf \
  --quantized-format hf \
  --original-quantization fp16 \
  --quantized-quantization 8bit \
  --prompts-file test_prompts.txt
```

## Output Interpretation

### Similarity Scores

- **≥ 0.8**: High quality preservation (excellent)
- **0.6-0.8**: Moderate quality preservation (good)
- **0.4-0.6**: Noticeable quality degradation (caution)
- **< 0.4**: Significant quality loss (poor)

### Quality Assessment

- **✅ PASS**: Quantized model maintains acceptable quality
- **⚠️ CAUTION**: Some quality degradation, review specific outputs
- **❌ FAIL**: Significant quality loss, quantization not recommended

## Sample Output

```
=== Model Comparison Report ===
Model: llama3.2-3b
Total Comparisons: 5
Successful: 5
Failed: 0

=== Quality Analysis ===
Average Similarity: 0.847
High Quality (>0.7): 5/5 (100.0%)
Medium Quality (0.5-0.7): 0/5 (0.0%)
Low Quality (<0.5): 0/5 (0.0%)

=== Performance Analysis ===
Average Speedup: 1.23x

✅ PASS: Quantized model maintains good quality
```

## Advanced Usage

### Custom Test Prompts

Create a `test_prompts.txt` file:

```
# Test prompts for model comparison
What is artificial intelligence?
Explain quantum computing
Write a short story about robots
How does photosynthesis work?
```

### Batch Comparison

Compare multiple quantization levels:

```bash
for quant in 4bit 8bit; do
  python inference/model_comparison.py \
    --model llama3.2-3b \
    --original-quantization fp16 \
    --quantized-quantization $quant \
    --output-file comparison_fp16_vs_${quant}.json
done
```

## Requirements

- Working HF model setup
- Sufficient memory for loading multiple models
- Optional: ONNX/OpenVINO models for format comparison

## Troubleshooting

1. **Memory Issues**: Reduce `max_new_tokens` or use smaller models
2. **Quantization Errors**: Check if bitsandbytes is properly installed
3. **ONNX/OpenVINO**: Ensure models are properly converted first

## Notes

- The quick comparison tool focuses on HF models since they're currently working
- Full comparison tool supports multiple formats but requires proper model conversion
- Results are saved in the `configs/results/` directory
- Use verbose mode (`--verbose`) for detailed logging
