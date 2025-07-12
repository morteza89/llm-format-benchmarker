"""
Model management utilities for loading and converting models
"""

import os
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    BitsAndBytesConfig, LlavaForConditionalGeneration
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    from openvino_genai import LLMPipeline
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from configs.config import MODELS, QUANTIZATION_CONFIGS, CACHE_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, conversion, and quantization"""

    def __init__(self):
        self.models_cache = {}
        self.tokenizers_cache = {}

    def load_huggingface_model(
        self,
        model_name: str,
        quantization: str = "fp16",
        device_map: str = "auto"
    ) -> tuple[Any, Any]:
        """Load a Hugging Face model with specified quantization"""

        if model_name not in MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Available: {list(MODELS.keys())}")

        model_config = MODELS[model_name]
        cache_key = f"{model_name}_{quantization}_{device_map}"

        # Check cache first
        if cache_key in self.models_cache:
            return self.models_cache[cache_key], self.tokenizers_cache[cache_key]

        logger.info(f"Loading {model_name} with {quantization} quantization")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.hf_model_id,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True
        )

        # Set up quantization config
        model_kwargs = {"cache_dir": str(CACHE_DIR), "trust_remote_code": True}

        if quantization in QUANTIZATION_CONFIGS:
            quant_config = QUANTIZATION_CONFIGS[quantization]

            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    **quant_config)
            elif quantization == "8bit":
                model_kwargs.update(quant_config)
            elif quantization == "fp16":
                model_kwargs.update(quant_config)

        # Handle special tokens
        if model_config.special_tokens:
            for token_type, token_value in model_config.special_tokens.items():
                if not hasattr(tokenizer, token_type) or getattr(tokenizer, token_type) is None:
                    setattr(tokenizer, token_type, token_value)
                    tokenizer.add_special_tokens({token_type: token_value})

        # Load model based on type
        try:
            if model_config.model_type == "vision-language":
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_config.hf_model_id,
                    **model_kwargs
                )
                # Also load processor for vision-language models
                processor = AutoProcessor.from_pretrained(
                    model_config.hf_model_id,
                    cache_dir=str(CACHE_DIR)
                )
                self.tokenizers_cache[cache_key] = (tokenizer, processor)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.hf_model_id,
                    **model_kwargs
                )
                self.tokenizers_cache[cache_key] = tokenizer

            # Resize token embeddings if needed
            if len(tokenizer) > model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(tokenizer))

            self.models_cache[cache_key] = model
            logger.info(f"Successfully loaded {model_name}")

            return model, self.tokenizers_cache[cache_key]

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def convert_to_onnx(
        self,
        model_name: str,
        quantization: str = "fp16",
        force_convert: bool = False
    ) -> Optional[str]:
        """Convert Hugging Face model to ONNX format"""

        if not ONNX_AVAILABLE:
            logger.error(
                "ONNX not available. Please install onnx and onnxruntime.")
            return None

        model_config = MODELS[model_name]
        onnx_path = MODELS_DIR / f"{model_name}_{quantization}.onnx"

        if onnx_path.exists() and not force_convert:
            logger.info(f"ONNX model already exists: {onnx_path}")
            return str(onnx_path)

        try:
            # Use optimum for better ONNX conversion support
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer

            logger.info(f"Converting {model_name} to ONNX using optimum...")

            # Load model with optimum
            model = ORTModelForCausalLM.from_pretrained(
                model_config.hf_model_id,
                export=True,
                cache_dir=str(CACHE_DIR),
                trust_remote_code=True
            )

            # Save the converted model
            model.save_pretrained(
                str(onnx_path.parent / f"{model_name}_{quantization}_onnx"))

            # Copy the main ONNX file to the expected location
            onnx_dir = onnx_path.parent / f"{model_name}_{quantization}_onnx"
            if (onnx_dir / "model.onnx").exists():
                import shutil
                shutil.copy(str(onnx_dir / "model.onnx"), str(onnx_path))
                logger.info(
                    f"Successfully converted {model_name} to ONNX: {onnx_path}")
                return str(onnx_path)
            else:
                logger.error("ONNX model.onnx not found after conversion")
                return None

        except Exception as e:
            logger.error(f"Failed to convert {model_name} to ONNX: {e}")

            # Fallback to manual torch.onnx.export (may not work for all models)
            try:
                logger.info("Trying fallback torch.onnx.export method...")

                # Load the HF model
                model, tokenizer = self.load_huggingface_model(
                    model_name, quantization)

                # Create dummy input
                dummy_input = tokenizer(
                    "This is a test sentence for ONNX conversion.",
                    return_tensors="pt",
                    max_length=min(model_config.max_length, 512),
                    padding="max_length",
                    truncation=True
                )

                # Convert to ONNX with simpler approach
                torch.onnx.export(
                    model,
                    tuple(dummy_input.values()),
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'output': {0: 'batch_size'}
                    }
                )

                logger.info(
                    f"Successfully converted {model_name} to ONNX: {onnx_path}")
                return str(onnx_path)

            except Exception as fallback_e:
                logger.error(
                    f"Fallback ONNX conversion also failed: {fallback_e}")
                return None

    def convert_to_openvino(
        self,
        model_name: str,
        quantization: str = "fp16",
        force_convert: bool = False
    ) -> Optional[str]:
        """Convert model to OpenVINO format"""

        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO not available. Please install openvino.")
            return None

        model_config = MODELS[model_name]
        ov_path = MODELS_DIR / f"{model_name}_{quantization}_ov"

        if ov_path.exists() and not force_convert:
            logger.info(f"OpenVINO model already exists: {ov_path}")
            return str(ov_path)

        try:
            # Try to convert from HF model directly
            from optimum.intel import OVModelForCausalLM
            import io
            from contextlib import redirect_stderr

            # Create a buffer to capture stderr and filter out warnings
            stderr_buffer = io.StringIO()

            # Load and convert with better error handling
            with redirect_stderr(stderr_buffer):
                ov_model = OVModelForCausalLM.from_pretrained(
                    model_config.hf_model_id,
                    export=True,
                    cache_dir=str(CACHE_DIR),
                    trust_remote_code=True
                )

            # Save the converted model
            ov_model.save_pretrained(str(ov_path))

            logger.info(
                f"Successfully converted {model_name} to OpenVINO: {ov_path}")
            return str(ov_path)

        except Exception as e:
            # Better error logging with more context
            error_msg = str(e)
            logger.error(
                f"Failed to convert {model_name} to OpenVINO: {error_msg}")

            # Try to provide more helpful error information
            if "invalid literal for int()" in error_msg and "WARNING" in error_msg:
                logger.warning(
                    "OpenVINO conversion failed due to warning message parsing. This is likely a version compatibility issue.")
                logger.info(
                    "Try updating optimum[intel] or use a different model format.")

            return None

    def load_onnx_model(self, onnx_path: str, providers: list = None, hardware: str = "cpu") -> Optional[Any]:
        """Load ONNX model for inference with Intel hardware optimization"""

        if not ONNX_AVAILABLE:
            logger.error("ONNX not available.")
            return None

        if providers is None:
            # Get recommended providers based on hardware type
            providers = self.get_recommended_onnx_providers(hardware)

        try:
            # Configure OpenVINO provider options if needed
            provider_options = {}
            if 'OpenVINOExecutionProvider' in providers:
                provider_options['OpenVINOExecutionProvider'] = {
                    'device_type': hardware.upper(),
                    'enable_opencl_throttling': False,
                    'enable_dynamic_shapes': True
                }

            # Try to create session with preferred providers
            session = ort.InferenceSession(
                onnx_path, providers=providers, provider_options=provider_options)

            # Log which providers are actually being used
            available_providers = session.get_providers()
            logger.info(f"Successfully loaded ONNX model: {onnx_path}")
            logger.info(f"Active execution providers: {available_providers}")

            return session
        except Exception as e:
            logger.error(
                f"Failed to load ONNX model with providers {providers}: {e}")

            # Fallback to CPU provider if OpenVINO provider fails
            try:
                logger.info("Attempting fallback to CPUExecutionProvider...")
                fallback_providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(
                    onnx_path, providers=fallback_providers)
                logger.info(
                    f"Successfully loaded ONNX model with fallback providers: {fallback_providers}")
                return session
            except Exception as fallback_error:
                logger.error(
                    f"Failed to load ONNX model with fallback providers: {fallback_error}")
                return None

    def convert_onnx_model_to_openvino(self, onnx_path: str, ov_path: str, hardware: str = "CPU") -> Optional[str]:
        """Convert ONNX model to OpenVINO format"""

        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO not available. Please install openvino.")
            return None

        try:
            # Use OpenVINO GenAI for conversion
            from openvino_genai import convert

            # Convert ONNX model to OpenVINO format
            convert(onnx_model=onnx_path,
                    openvino_model=ov_path, device=hardware)

            logger.info(
                f"Successfully converted ONNX model to OpenVINO: {ov_path}")
            return str(ov_path)
        except Exception as e:
            logger.error(f"Failed to convert ONNX model to OpenVINO: {e}")
            return None

    def load_openvino_model(self, ov_path: str, device: str = "CPU") -> Optional[Any]:
        """Load OpenVINO model for inference"""

        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO not available.")
            return None

        try:
            # Use OpenVINO GenAI for LLM inference
            pipe = LLMPipeline(str(ov_path), device)
            logger.info(f"Successfully loaded OpenVINO model: {ov_path}")
            return pipe
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            return None

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration information"""

        if model_name not in MODELS:
            return {}

        model_config = MODELS[model_name]
        return {
            "name": model_config.name,
            "hf_model_id": model_config.hf_model_id,
            "model_type": model_config.model_type,
            "max_length": model_config.max_length,
            "context_length": model_config.context_length,
            "supported_quantizations": model_config.supported_quantizations
        }

    def cleanup_cache(self):
        """Clear model cache to free memory"""
        self.models_cache.clear()
        self.tokenizers_cache.clear()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")

    def get_available_onnx_providers(self) -> List[str]:
        """Get list of available ONNX Runtime execution providers"""
        if not ONNX_AVAILABLE:
            return []

        try:
            # Try different ways to get available providers
            if hasattr(ort, 'get_available_providers'):
                available_providers = ort.get_available_providers()
            else:
                # Fallback: use common providers and test them
                available_providers = ['CPUExecutionProvider']

                # Test if OpenVINO provider is available
                try:
                    providers_to_test = [
                        'OpenVINOExecutionProvider', 'DmlExecutionProvider', 'CUDAExecutionProvider']
                    for provider in providers_to_test:
                        try:
                            # This will raise an exception if the provider is not available
                            ort.InferenceSession(providers=[provider])
                            available_providers.append(provider)
                        except Exception:
                            continue
                except Exception:
                    pass

            logger.info(
                f"Available ONNX Runtime providers: {available_providers}")
            return available_providers

        except Exception as e:
            logger.error(f"Failed to get available providers: {e}")
            return ['CPUExecutionProvider']  # Fallback to CPU

    def get_recommended_onnx_providers(self, hardware: str = "cpu") -> List[str]:
        """Get recommended ONNX Runtime providers for specific hardware"""
        available_providers = self.get_available_onnx_providers()

        # Define provider preferences by hardware
        if hardware.lower() == "npu":
            preferred = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        elif hardware.lower() == "gpu":
            preferred = ['OpenVINOExecutionProvider', 'DmlExecutionProvider',
                         'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:  # CPU
            preferred = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']

        # Return only available providers in preferred order
        recommended = [p for p in preferred if p in available_providers]

        # Always ensure CPU is available as fallback
        if 'CPUExecutionProvider' not in recommended:
            recommended.append('CPUExecutionProvider')

        logger.info(f"Recommended providers for {hardware}: {recommended}")
        return recommended


def get_model_manager() -> ModelManager:
    """Get model manager instance"""
    return ModelManager()


if __name__ == "__main__":
    # Test model loading
    manager = ModelManager()

    # Test model info
    for model_name in MODELS.keys():
        info = manager.get_model_info(model_name)
        print(f"\n{model_name}: {info}")

    # Test loading a small model (if available)
    try:
        model, tokenizer = manager.load_huggingface_model(
            "llama3.2-3b", "fp16")
        print(f"\nSuccessfully loaded model: {type(model)}")
        print(f"Tokenizer: {type(tokenizer)}")
    except Exception as e:
        print(f"Model loading test failed: {e}")
