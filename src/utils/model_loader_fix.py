#!/usr/bin/env python3
"""
Fixed model loader for handling 7B model quantization errors
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB"""
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        return 0.0


def get_optimal_model_config(model_name: str, gpu_memory_gb: float) -> Dict[str, Any]:
    """
    Get optimal configuration based on model size and available GPU memory
    
    Args:
        model_name: HuggingFace model name
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with quantization_config and device_map
    """
    model_name_lower = model_name.lower()
    
    # 32B models - always use 4-bit with auto device map
    if "32b" in model_name_lower:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            "device_map": "auto"
        }
    
    # 22B models - 4-bit with auto device map
    elif "22b" in model_name_lower:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            "device_map": "auto"
        }
    
    # 13B models
    elif "13b" in model_name_lower:
        if gpu_memory_gb >= 32:
            # Can fit with 8-bit on single GPU
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": {"": 0}
            }
        else:
            # Need 4-bit quantization
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                "device_map": "auto"
            }
    
    # 7B models - special handling to avoid CPU offloading error
    elif "7b" in model_name_lower:
        if gpu_memory_gb >= 16:
            # Use 8-bit for better quality
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": {"": 0}  # Force single GPU
            }
        elif gpu_memory_gb >= 8:
            # Use 4-bit to fit on single GPU
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                "device_map": {"": 0}  # Force single GPU
            }
        else:
            # Very limited memory - enable CPU offloading
            logger.warning(f"Limited GPU memory ({gpu_memory_gb:.1f}GB) for 7B model. Enabling CPU offloading.")
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": "auto"
            }
    
    # 3B models
    elif "3b" in model_name_lower:
        if gpu_memory_gb >= 8:
            # No quantization needed
            return {
                "quantization_config": None,
                "device_map": {"": 0}
            }
        else:
            # Use 8-bit quantization
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": {"": 0}
            }
    
    # Default for unknown/small models
    else:
        return {
            "quantization_config": None,
            "device_map": "auto" if torch.cuda.is_available() else None
        }


def load_model_with_optimal_settings(
    model_name: str,
    cache_dir: Optional[str] = None,
    use_auth_token: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load model with optimal settings based on model size and GPU memory
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Optional cache directory
        use_auth_token: Optional HuggingFace auth token
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Get GPU memory
    gpu_memory = get_gpu_memory_gb()
    logger.info(f"Available GPU memory: {gpu_memory:.1f}GB")
    
    # Get optimal configuration
    config = get_optimal_model_config(model_name, gpu_memory)
    logger.info(f"Using configuration: {config}")
    
    # Load tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": True
    }
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir
    if use_auth_token:
        tokenizer_kwargs["token"] = use_auth_token
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading arguments
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": config["device_map"]
    }
    
    if config["quantization_config"]:
        model_kwargs["quantization_config"] = config["quantization_config"]
    
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if use_auth_token:
        model_kwargs["token"] = use_auth_token
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        
        # Fallback: Try with more aggressive quantization
        if "7b" in model_name.lower() and gpu_memory < 16:
            logger.info("Retrying with 4-bit quantization...")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = {"": 0}
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            logger.info("Model loaded with 4-bit quantization")
            return model, tokenizer
        else:
            raise


def prepare_model_for_training(model, use_gradient_checkpointing: bool = True):
    """
    Prepare model for training with memory optimizations
    
    Args:
        model: The loaded model
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Prepared model
    """
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Enable mixed precision if using CUDA
    if torch.cuda.is_available():
        model = model.to(torch.float16)
    
    return model


# Example usage
if __name__ == "__main__":
    # Test with 7B model
    model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    
    try:
        model, tokenizer = load_model_with_optimal_settings(model_name)
        print(f"Successfully loaded {model_name}")
        
        # Get model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count:,}")
        
        # Test generation
        inputs = tokenizer("こんにちは", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()