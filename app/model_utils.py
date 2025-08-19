#!/usr/bin/env python3
"""
モデル関連のユーティリティ関数
Model loading, quantization, and configuration utilities
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)


def get_auth_token() -> Optional[str]:
    """
    Hugging Face認証トークンを取得
    
    Returns:
        認証トークンまたはNone
    """
    return os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')


def requires_authentication(model_name: str) -> bool:
    """
    モデルが認証を必要とするかチェック
    
    Args:
        model_name: モデル名
        
    Returns:
        認証が必要な場合True
    """
    auth_required_models = ['meta-llama', 'Meta-Llama', 'Qwen', 'DeepSeek']
    return any(auth_model in model_name for auth_model in auth_required_models)


def get_model_size_category(model_name: str) -> str:
    """
    モデル名からサイズカテゴリを判定
    
    Args:
        model_name: モデル名
        
    Returns:
        サイズカテゴリ ('small', 'medium', 'large', 'xlarge')
    """
    model_name_lower = model_name.lower()
    
    # Extra large models (32B+)
    if any(size in model_name_lower for size in ['70b', '32b', '22b']):
        return 'xlarge'
    
    # Large models (13B-20B)
    if any(size in model_name_lower for size in ['17b', '13b', '10b']):
        return 'large'
    
    # Medium models (7B-8B)
    if any(size in model_name_lower for size in ['8b', '7b']):
        return 'medium'
    
    # Small models (<7B)
    if any(size in model_name_lower for size in ['3b', '3.6b', '1.8b', '1.3b']):
        return 'small'
    
    # Default to medium if size not specified
    return 'medium'


def create_quantization_config(
    model_name: str,
    training_method: str = "lora",
    force_4bit: bool = False
) -> Optional[BitsAndBytesConfig]:
    """
    モデルとトレーニング方法に基づいて量子化設定を作成
    
    Args:
        model_name: モデル名
        training_method: トレーニング方法 ('lora', 'qlora', 'full')
        force_4bit: 強制的に4bit量子化を使用
        
    Returns:
        量子化設定またはNone（フルファインチューニングの場合）
    """
    # フルファインチューニングの場合は量子化なし
    if training_method == "full" and not force_4bit:
        return None
    
    model_size = get_model_size_category(model_name)
    
    # DeepSeek-R1-Distill-Qwen-32Bの特別処理（強化されたメモリ最適化）
    if "DeepSeek-R1-Distill-Qwen-32B" in model_name:
        if training_method == "qlora":
            # QLoRAの場合は最大限のメモリ最適化
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,  # メモリ削減のため
                bnb_4bit_quant_storage=torch.uint8  # ストレージを最小化
            )
        elif training_method == "lora":
            # 通常のLoRAでも4bit量子化を使用
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None
    
    # 小〜中規模モデルの最適化
    if model_size in ['small', 'medium'] and not force_4bit:
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # GPU メモリが十分な場合は8bit量子化
        if gpu_memory >= 16:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
    
    # デフォルトまたは大規模モデルは4bit量子化
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def get_device_map(model_name: str, model_size: Optional[str] = None) -> Any:
    """
    モデルサイズに基づいてデバイスマップを決定
    
    Args:
        model_name: モデル名
        model_size: モデルサイズカテゴリ（指定されない場合は自動判定）
        
    Returns:
        デバイスマップ設定
    """
    if model_size is None:
        model_size = get_model_size_category(model_name)
    
    # 小〜中規模モデルは単一GPUに配置（CPUオフロード回避）
    if model_size in ['small', 'medium']:
        if torch.cuda.is_available():
            return {"": 0}
        return None
    
    # 大規模モデルは自動配置（CPUオフロード有効）
    if model_size in ['large', 'xlarge']:
        if torch.cuda.is_available():
            # DeepSeek-R1-Distill-Qwen-32Bの特別処理
            if "DeepSeek-R1-Distill-Qwen-32B" in model_name:
                # メモリ効率を最大化するためのカスタムデバイスマップ
                return {
                    "model.embed_tokens": 0,
                    "model.layers": 0,
                    "model.norm": 0,
                    "lm_head": "cpu"  # lm_headをCPUにオフロード
                }
            return "auto"
        return None
    
    # デフォルト
    return "auto" if torch.cuda.is_available() else None


def load_tokenizer(
    model_name: str,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True
) -> PreTrainedTokenizer:
    """
    トークナイザーを読み込み、pad_tokenを設定
    
    Args:
        model_name: モデル名
        cache_dir: キャッシュディレクトリ
        trust_remote_code: リモートコードを信頼するか
        
    Returns:
        設定済みのトークナイザー
    """
    auth_token = get_auth_token() if requires_authentication(model_name) else None
    
    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": auth_token
    }
    
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = str(cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # pad_tokenが設定されていない場合はeos_tokenを使用
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    training_method: str = "lora",
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    モデルとトークナイザーを統一的に読み込む
    
    Args:
        model_name: モデル名  
        training_method: トレーニング方法
        cache_dir: キャッシュディレクトリ
        trust_remote_code: リモートコードを信頼するか
        low_cpu_mem_usage: CPU メモリ使用量を削減するか
        
    Returns:
        (model, tokenizer) のタプル
    """
    # DeepSeek モデルの環境設定
    if "DeepSeek-R1-Distill-Qwen-32B" in model_name and cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "0"
    
    # トークナイザーの読み込み
    tokenizer = load_tokenizer(model_name, cache_dir, trust_remote_code)
    
    # 量子化設定の作成
    quantization_config = create_quantization_config(model_name, training_method)
    
    # デバイスマップの決定
    device_map = get_device_map(model_name)
    
    # モデル読み込みパラメータの構築
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage
    }
    
    # 認証トークンの追加
    if requires_authentication(model_name):
        model_kwargs["token"] = get_auth_token()
    
    # キャッシュディレクトリの追加
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
    
    # 量子化設定の追加
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # デバイスマップの追加
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    # モデルの読み込み
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise
    
    return model, tokenizer


def handle_model_loading_error(
    error: Exception,
    model_name: str,
    task_id: Optional[str] = None
) -> str:
    """
    モデル読み込みエラーを処理し、適切なエラーメッセージを返す
    
    Args:
        error: 発生したエラー
        model_name: モデル名
        task_id: タスクID（ログ用）
        
    Returns:
        ユーザー向けエラーメッセージ
    """
    error_str = str(error)
    
    # CUDA OOM エラー
    if "CUDA out of memory" in error_str:
        if "DeepSeek-R1-Distill-Qwen-32B" in model_name:
            return (
                "メモリ不足エラー: DeepSeek-R1-Distill-Qwen-32B-Japaneseは非常に大きなモデルです。\n"
                "推奨: QLoRA (4bit) または LoRA を使用してください。\n"
                "フルファインチューニングには140GB以上のGPUメモリが必要です。"
            )
        return "GPUメモリ不足エラー: より小さいバッチサイズまたは量子化を使用してください。"
    
    # 認証エラー
    if "401 Client Error: Unauthorized" in error_str:
        if any(auth_model in model_name for auth_model in ['meta-llama', 'Meta-Llama']):
            return (
                "認証エラー: 無効なHugging Faceトークンまたは権限不足です。\n"
                "1. https://huggingface.co/settings/tokens で有効なトークンを作成\n"
                "2. Meta Llamaモデルのライセンス同意が必要\n"
                "3. 環境変数HUGGINGFACE_TOKENまたはHF_TOKENを正しく設定"
            )
        return "認証エラー: Hugging Faceトークンが必要です。環境変数HUGGINGFACE_TOKENまたはHF_TOKENを設定してください。"
    
    # モデル識別子エラー
    if "not a valid model identifier" in error_str:
        if any(auth_model in model_name for auth_model in ['meta-llama', 'Meta-Llama']):
            return "認証エラー: Meta Llamaモデルを使用するにはHugging Face認証トークンが必要です。"
        return f"モデル識別子エラー: {model_name} が見つかりません。モデル名を確認してください。"
    
    # デフォルトエラー
    if task_id:
        logger.error(f"Task {task_id}: Model loading error for {model_name}: {error_str}")
    
    return f"モデル読み込みエラー: {error_str}"


def get_training_config_path() -> Path:
    """
    トレーニング設定ファイルのパスを取得
    
    Returns:
        設定ファイルのパス
    """
    project_root = Path(os.getcwd())
    return project_root / "config" / "training_config.yaml"


def load_training_config(training_method: str) -> Dict[str, Any]:
    """
    トレーニング設定を読み込む
    
    Args:
        training_method: トレーニング方法
        
    Returns:
        トレーニング設定の辞書
    """
    config_path = get_training_config_path()
    training_config = {}
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                preset_key = f"{training_method}_finetuning"
                if preset_key in config_data.get('training_presets', {}):
                    training_config = config_data['training_presets'][preset_key]
        except Exception as e:
            logger.warning(f"Failed to load training config: {e}")
    
    return training_config


def get_output_directory(method_name: str, timestamp: Optional[str] = None) -> Path:
    """
    出力ディレクトリを作成して返す
    
    Args:
        method_name: メソッド名
        timestamp: タイムスタンプ（指定されない場合は自動生成）
        
    Returns:
        出力ディレクトリのパス
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    project_root = Path(os.getcwd())
    output_base = project_root / "outputs"
    
    # outputsディレクトリが存在しない場合は作成
    if not output_base.exists():
        try:
            output_base.mkdir(parents=True, exist_ok=True)
            os.chmod(output_base, 0o777)
        except Exception as e:
            logger.warning(f"Failed to create outputs directory with permissions: {e}")
    
    output_dir = output_base / f"{method_name.lower()}_{timestamp}"
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(output_dir, 0o777)
    except PermissionError:
        # 権限エラーの場合、親ディレクトリの権限を再設定してリトライ
        try:
            os.chmod(output_base, 0o777)
            output_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(output_dir, 0o777)
        except Exception as e:
            logger.error(f"Failed to create output directory with retry: {e}")
            raise
    
    return output_dir