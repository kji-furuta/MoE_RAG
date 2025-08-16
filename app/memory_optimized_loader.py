#!/usr/bin/env python3
"""
メモリ最適化されたモデルローダー
ファインチューニングしたモデルの読み込み時のメモリ不足問題を解決
"""

import os
import torch
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

# メモリ管理の環境変数を設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def clear_gpu_memory():
    """GPUメモリをクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")

def get_gpu_memory_info():
    """GPUメモリ情報を取得"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free
        }
    return None

def get_optimal_quantization_config(model_name: str, available_memory_gb: float = None):
    """モデルサイズと利用可能メモリに基づいて最適な量子化設定を決定"""
    
    model_name_lower = model_name.lower()
    
    # 利用可能メモリが指定されていない場合は自動検出
    if available_memory_gb is None and torch.cuda.is_available():
        memory_info = get_gpu_memory_info()
        available_memory_gb = memory_info["free_gb"] if memory_info else 0
    
    # 32B/22Bモデルの場合 - 最も積極的な量子化
    if any(size in model_name_lower for size in ['22b', '32b']):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        ), "auto"
    
    # 7B/8Bモデルの場合 - メモリに応じて調整
    elif any(size in model_name_lower for size in ['7b', '8b']):
        if available_memory_gb >= 16:
            # 十分なメモリがある場合は8bit量子化
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            ), {"": 0}
        else:
            # メモリが不足している場合は4bit量子化
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ), {"": 0}
    
    # 3B/4Bモデルの場合 - 軽量な量子化
    elif any(size in model_name_lower for size in ['3b', '4b']):
        if available_memory_gb >= 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            ), {"": 0}
        else:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ), {"": 0}
    
    # その他のモデル - デフォルト設定
    else:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ), "auto"

def load_base_model_with_optimization(
    model_name: str,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_cpu: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """メモリ最適化されたベースモデルの読み込み"""
    
    logger.info(f"Loading base model: {model_name}")
    
    # GPUメモリをクリア
    clear_gpu_memory()
    
    # GPUメモリ情報をログ
    memory_info = get_gpu_memory_info()
    if memory_info:
        logger.info(f"GPU memory before loading: {memory_info}")
    
    # トークナイザーの読み込み
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
    
    # CPU強制使用の場合
    if force_cpu or not torch.cuda.is_available():
        logger.info("Loading model on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            token=use_auth_token
        )
        return model, tokenizer
    
    # 最適な量子化設定を取得
    quantization_config, device_map = get_optimal_quantization_config(model_name)
    
    # モデルの読み込み
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }
    
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if use_auth_token:
        model_kwargs["token"] = use_auth_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # メモリ使用量をログ
    memory_info = get_gpu_memory_info()
    if memory_info:
        logger.info(f"GPU memory after loading: {memory_info}")
    
    return model, tokenizer

def load_finetuned_model_with_optimization(
    base_model_name: str,
    model_path: str,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_cpu: bool = False
) -> Tuple[PeftModel, AutoTokenizer]:
    """メモリ最適化されたファインチューニングモデルの読み込み"""
    
    logger.info(f"Loading finetuned model: {base_model_name} + {model_path}")
    
    # GPUメモリをクリア
    clear_gpu_memory()
    
    # ベースモデルを読み込み
    base_model, tokenizer = load_base_model_with_optimization(
        base_model_name, use_auth_token, cache_dir, force_cpu
    )
    
    # LoRAアダプターを読み込み
    logger.info(f"Loading LoRA adapter from: {model_path}")
    
    # メモリ使用量をチェック
    memory_info = get_gpu_memory_info()
    if memory_info and memory_info["free_gb"] < 2:
        logger.warning(f"Low GPU memory available: {memory_info['free_gb']:.2f}GB")
        # より積極的なメモリクリア
        clear_gpu_memory()
    
    # LoRAアダプターを読み込み
    model = PeftModel.from_pretrained(base_model, str(model_path))
    
    # デバイス配置の最適化
    if torch.cuda.is_available() and not force_cpu:
        # モデル全体を単一GPUに移動（デバイス間の問題を回避）
        model = model.to("cuda:0")
        logger.info("Model moved to cuda:0")
    elif force_cpu:
        # CPUに移動
        model = model.to("cpu")
        logger.info("Model moved to CPU")
    
    # 最終的なメモリ使用量をログ
    memory_info = get_gpu_memory_info()
    if memory_info:
        logger.info(f"Final GPU memory usage: {memory_info}")
    
    return model, tokenizer

def safe_model_generation(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    force_cpu: bool = False
) -> str:
    """安全なテキスト生成（メモリ管理付き）"""
    
    try:
        # 入力のトークン化
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # モデルのデバイスを確認
        model_device = next(model.parameters()).device
        logger.info(f"Model device: {model_device}")
        
        # 入力テンソルをモデルと同じデバイスに転送
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        logger.info(f"Input tensors moved to {model_device}")
        
        # 生成設定
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # テキスト生成
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        # デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 入力プロンプトを除去
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # メモリをクリア
        clear_gpu_memory()
        raise

def get_model_memory_requirements(model_name: str) -> dict:
    """モデルのメモリ要件を推定"""
    
    model_name_lower = model_name.lower()
    
    # モデルサイズの推定
    if any(size in model_name_lower for size in ['32b', '32b']):
        return {
            "estimated_size_gb": 32,
            "min_gpu_memory_gb": 24,
            "recommended_quantization": "4bit",
            "can_run_on_cpu": False
        }
    elif any(size in model_name_lower for size in ['22b', '22b']):
        return {
            "estimated_size_gb": 22,
            "min_gpu_memory_gb": 16,
            "recommended_quantization": "4bit",
            "can_run_on_cpu": False
        }
    elif any(size in model_name_lower for size in ['7b', '8b']):
        return {
            "estimated_size_gb": 8,
            "min_gpu_memory_gb": 8,
            "recommended_quantization": "8bit",
            "can_run_on_cpu": True
        }
    elif any(size in model_name_lower for size in ['3b', '4b']):
        return {
            "estimated_size_gb": 4,
            "min_gpu_memory_gb": 4,
            "recommended_quantization": "8bit",
            "can_run_on_cpu": True
        }
    else:
        return {
            "estimated_size_gb": 8,
            "min_gpu_memory_gb": 8,
            "recommended_quantization": "4bit",
            "can_run_on_cpu": True
        } 