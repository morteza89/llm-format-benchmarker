"""
Quantization utilities for different model formats
"""

import os
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    from openvino.tools import mo
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from configs.config import MODELS, MODELS_DIR, CACHE_DIR

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Manages quantization for different model formats"""

    def __init__(self):
        self.quantization_methods = {
            "hf": self._quantize_huggingface,
            "onnx": self._quantize_onnx,
            "ov": self._quantize_openvino
        }

    def quantize_model(
        self,
        model_name: str,
        format_type: str,
        quantization_type: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Quantize a model in specified format

        Args:
            model_name: Name of the model to quantize
            format_type: Format of the model (hf, onnx, ov)
            quantization_type: Type of quantization (4bit, 8bit, fp16)
            input_path: Path to input model (optional)
            output_path: Path to save quantized model (optional)

        Returns:
            Path to quantized model or None if failed
        """

        if model_name not in MODELS:
            logger.error(f"Model {model_name} not supported")
            return None

        if format_type not in self.quantization_methods:
            logger.error(f"Format {format_type} not supported")
            return None

        logger.info(
            f"Quantizing {model_name} ({format_type}) to {quantization_type}")

        try:
            return self.quantization_methods[format_type](
                model_name, quantization_type, input_path, output_path
            )
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return None

    def _quantize_huggingface(
        self,
        model_name: str,
        quantization_type: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Quantize Hugging Face model"""

        model_config = MODELS[model_name]

        if output_path is None:
            output_path = MODELS_DIR / f"{model_name}_{quantization_type}_hf"

        output_path = Path(output_path)

        if output_path.exists():
            logger.info(f"Quantized model already exists: {output_path}")
            return str(output_path)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.hf_model_id,
                cache_dir=str(CACHE_DIR),
                trust_remote_code=True
            )

            # Configure quantization
            model_kwargs = {
                "cache_dir": str(CACHE_DIR),
                "trust_remote_code": True
            }

            if quantization_type == "4bit":
                # Use the newer API for BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                # Add device mapping for better memory management
                model_kwargs["device_map"] = "auto"

            elif quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model_kwargs["quantization_config"] = quantization_config
                # Add device mapping for better memory management
                model_kwargs["device_map"] = "auto"

            elif quantization_type == "fp16":
                model_kwargs["torch_dtype"] = torch.float16

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_model_id,
                **model_kwargs
            )

            # Handle special tokens
            if model_config.special_tokens:
                for token_type, token_value in model_config.special_tokens.items():
                    if not hasattr(tokenizer, token_type) or getattr(tokenizer, token_type) is None:
                        setattr(tokenizer, token_type, token_value)
                        tokenizer.add_special_tokens({token_type: token_value})

                # Resize embeddings if needed
                if len(tokenizer) > model.get_input_embeddings().num_embeddings:
                    model.resize_token_embeddings(len(tokenizer))

            # Save quantized model
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            logger.info(f"Successfully quantized HF model: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to quantize HF model: {e}")
            return None

    def _quantize_onnx(
        self,
        model_name: str,
        quantization_type: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Quantize ONNX model"""

        if not ONNX_AVAILABLE:
            logger.error("ONNX not available for quantization")
            return None

        if input_path is None:
            input_path = MODELS_DIR / f"{model_name}_fp16.onnx"

        if output_path is None:
            output_path = MODELS_DIR / f"{model_name}_{quantization_type}.onnx"

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input ONNX model not found: {input_path}")
            return None

        if output_path.exists():
            logger.info(f"Quantized ONNX model already exists: {output_path}")
            return str(output_path)

        try:
            if quantization_type == "8bit":
                # Dynamic quantization to INT8
                quantize_dynamic(
                    str(input_path),
                    str(output_path),
                    weight_type=QuantType.QUInt8
                )
            elif quantization_type == "4bit":
                # Note: 4-bit quantization might not be directly supported in ONNX
                # This is a placeholder for future implementation
                logger.warning(
                    "4-bit quantization not fully supported for ONNX")
                # For now, fall back to 8-bit
                quantize_dynamic(
                    str(input_path),
                    str(output_path),
                    weight_type=QuantType.QUInt8
                )
            else:
                # For fp16, just copy the model
                import shutil
                shutil.copy2(str(input_path), str(output_path))

            logger.info(f"Successfully quantized ONNX model: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to quantize ONNX model: {e}")
            return None

    def _quantize_openvino(
        self,
        model_name: str,
        quantization_type: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Quantize OpenVINO model"""

        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO not available for quantization")
            return None

        model_config = MODELS[model_name]

        if output_path is None:
            output_path = MODELS_DIR / f"{model_name}_{quantization_type}_ov"

        output_path = Path(output_path)

        if output_path.exists():
            logger.info(
                f"Quantized OpenVINO model already exists: {output_path}")
            return str(output_path)

        try:
            # Use optimum-intel for quantization
            from optimum.intel import OVModelForCausalLM, OVConfig

            # Configure quantization
            ov_config = OVConfig()

            if quantization_type == "8bit":
                ov_config.compression = {"algorithm": "quantization"}
            elif quantization_type == "4bit":
                ov_config.compression = {
                    "algorithm": "quantization", "preset": "mixed"}

            # Load and quantize model
            ov_model = OVModelForCausalLM.from_pretrained(
                model_config.hf_model_id,
                export=True,
                ov_config=ov_config,
                cache_dir=str(CACHE_DIR)
            )

            # Save quantized model
            ov_model.save_pretrained(str(output_path))

            logger.info(
                f"Successfully quantized OpenVINO model: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to quantize OpenVINO model: {e}")
            return None

    def get_quantization_info(self, model_path: str, format_type: str) -> Dict[str, Any]:
        """Get information about quantized model"""

        model_path = Path(model_path)

        if not model_path.exists():
            return {"error": "Model not found"}

        info = {
            "path": str(model_path),
            "format": format_type,
            "size_mb": self._get_model_size(model_path),
            "exists": True
        }

        # Add format-specific information
        if format_type == "hf":
            info.update(self._get_hf_model_info(model_path))
        elif format_type == "onnx":
            info.update(self._get_onnx_model_info(model_path))
        elif format_type == "ov":
            info.update(self._get_ov_model_info(model_path))

        return info

    def _get_model_size(self, model_path: Path) -> float:
        """Get model size in MB"""

        if model_path.is_file():
            return model_path.stat().st_size / (1024 * 1024)
        elif model_path.is_dir():
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        else:
            return 0.0

    def _get_hf_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get Hugging Face model information"""

        config_path = model_path / "config.json"

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {
                "model_type": config.get("model_type", "unknown"),
                "vocab_size": config.get("vocab_size", 0),
                "hidden_size": config.get("hidden_size", 0),
                "num_layers": config.get("num_hidden_layers", 0)
            }
        else:
            return {"error": "Config not found"}

    def _get_onnx_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get ONNX model information"""

        if not ONNX_AVAILABLE:
            return {"error": "ONNX not available"}

        try:
            model = onnx.load(str(model_path))
            return {
                "opset_version": model.opset_import[0].version,
                "ir_version": model.ir_version,
                "num_inputs": len(model.graph.input),
                "num_outputs": len(model.graph.output)
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_ov_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get OpenVINO model information"""

        if not OPENVINO_AVAILABLE:
            return {"error": "OpenVINO not available"}

        try:
            xml_path = model_path / "openvino_model.xml"
            if xml_path.exists():
                core = ov.Core()
                model = core.read_model(str(xml_path))
                return {
                    "num_inputs": len(model.inputs),
                    "num_outputs": len(model.outputs),
                    "input_shapes": [str(input.shape) for input in model.inputs],
                    "output_shapes": [str(output.shape) for output in model.outputs]
                }
            else:
                return {"error": "OpenVINO XML not found"}
        except Exception as e:
            return {"error": str(e)}

    def compare_quantization_methods(self, model_name: str) -> Dict[str, Any]:
        """Compare different quantization methods for a model"""

        results = {}

        for format_type in ["hf", "onnx", "ov"]:
            results[format_type] = {}

            for quant_type in ["fp16", "8bit", "4bit"]:
                model_path = MODELS_DIR / \
                    f"{model_name}_{quant_type}_{format_type if format_type != 'hf' else 'hf'}"

                if format_type == "onnx":
                    model_path = MODELS_DIR / f"{model_name}_{quant_type}.onnx"

                info = self.get_quantization_info(str(model_path), format_type)
                results[format_type][quant_type] = info

        return results


def get_quantization_manager() -> QuantizationManager:
    """Get quantization manager instance"""
    return QuantizationManager()


if __name__ == "__main__":
    # Test quantization
    manager = QuantizationManager()

    # Test quantization info
    for model_name in ["qwen2.5", "llama3.2-3b", "llava"]:
        print(f"\n=== {model_name} ===")
        comparison = manager.compare_quantization_methods(model_name)

        for format_type, quant_results in comparison.items():
            print(f"\n{format_type.upper()}:")
            for quant_type, info in quant_results.items():
                if "error" not in info:
                    print(f"  {quant_type}: {info.get('size_mb', 0):.1f} MB")
                else:
                    print(f"  {quant_type}: Not available")
