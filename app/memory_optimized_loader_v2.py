#!/usr/bin/env python3
"""
改善版メモリ最適化モデルローダー
統一メモリ管理システムと連携したモデル読み込み
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 統一メモリ管理システムをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.memory_manager import get_memory_manager, QuantizationType

logger = logging.getLogger(__name__)


class OptimizedModelLoader:
    """最適化されたモデルローダー"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: デバッグモード（本番環境では必ずFalse）
        """
        self.memory_manager = get_memory_manager(debug_mode=debug_mode)
        self.loaded_models = {}  # モデルキャッシュ
        
    def load_base_model(
        self,
        model_name: str,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_cpu: bool = False,
        target_batch_size: int = 1,
        for_training: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """メモリ最適化されたベースモデルの読み込み
        
        Args:
            model_name: モデル名またはパス
            use_auth_token: Hugging Face トークン
            cache_dir: キャッシュディレクトリ
            force_cpu: CPU強制使用フラグ
            target_batch_size: 目標バッチサイズ（メモリ計算用）
            for_training: トレーニング用フラグ（メモリ要求が異なる）
        
        Returns:
            モデルとトークナイザーのタプル
        """
        logger.info(f"Loading base model: {model_name}")
        
        # キャッシュチェック
        cache_key = f"{model_name}_{for_training}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        # メモリクリア（積極的）
        self.memory_manager.clear_gpu_memory(aggressive=True)
        
        # メモリ状況をログ
        memory_status = self.memory_manager.monitor_memory_usage()
        logger.info(f"Memory status before loading: {memory_status}")
        
        # トークナイザーの読み込み
        tokenizer = self._load_tokenizer(model_name, use_auth_token, cache_dir)
        
        # CPU強制使用またはGPU不使用の場合
        if force_cpu or not torch.cuda.is_available():
            logger.info("Loading model on CPU")
            model = self._load_cpu_model(model_name, use_auth_token, cache_dir)
        else:
            # 最適な量子化設定を取得
            quantization_config = self.memory_manager.get_optimal_quantization(
                model_name=model_name,
                target_batch_size=target_batch_size,
                for_training=for_training
            )
            
            logger.info(f"Using quantization: {quantization_config.type.value}")
            
            # モデルを読み込み
            model = self._load_quantized_model(
                model_name,
                quantization_config,
                use_auth_token,
                cache_dir
            )
        
        # メモリ状況を再度ログ
        memory_status = self.memory_manager.monitor_memory_usage()
        logger.info(f"Memory status after loading: {memory_status}")
        
        # レコメンデーションがある場合は警告
        if memory_status.get("recommendations"):
            for rec in memory_status["recommendations"]:
                logger.warning(f"Memory recommendation: {rec}")
        
        # キャッシュに保存（トレーニング用でない場合のみ）
        if not for_training:
            self.loaded_models[cache_key] = (model, tokenizer)
        
        return model, tokenizer
    
    def _load_tokenizer(
        self,
        model_name: str,
        use_auth_token: Optional[str],
        cache_dir: Optional[str]
    ) -> AutoTokenizer:
        """トークナイザーを読み込み"""
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": True  # 高速トークナイザーを優先
        }
        
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        if use_auth_token:
            tokenizer_kwargs["token"] = use_auth_token
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            
            # パディングトークンの設定
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            # フォールバック：低速トークナイザー
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
    
    def _load_cpu_model(
        self,
        model_name: str,
        use_auth_token: Optional[str],
        cache_dir: Optional[str]
    ) -> AutoModelForCausalLM:
        """CPUモデルを読み込み"""
        model_kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": {"": "cpu"}
        }
        
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if use_auth_token:
            model_kwargs["token"] = use_auth_token
        
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    def _load_quantized_model(
        self,
        model_name: str,
        quantization_config,
        use_auth_token: Optional[str],
        cache_dir: Optional[str]
    ) -> AutoModelForCausalLM:
        """量子化モデルを読み込み"""
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # 量子化設定を適用
        if quantization_config.type != QuantizationType.NONE:
            bnb_config = quantization_config.to_bnb_config()
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
        
        # デバイスマップとdtypeの設定
        model_kwargs["device_map"] = quantization_config.device_map or "auto"
        
        # 量子化なしの場合のみdtypeを設定（量子化時は自動設定される）
        if quantization_config.type == QuantizationType.NONE:
            model_kwargs["torch_dtype"] = quantization_config.compute_dtype
        
        # 最大メモリ制限
        if quantization_config.max_memory:
            model_kwargs["max_memory"] = quantization_config.max_memory
        
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if use_auth_token:
            model_kwargs["token"] = use_auth_token
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            return model
            
        except OutOfMemoryError as e:
            logger.error(f"OOM Error: {e}")
            logger.info("Retrying with more aggressive quantization...")
            
            # より積極的な量子化で再試行
            self.memory_manager.clear_gpu_memory(aggressive=True)
            
            # 4bit量子化 + CPU オフロード
            fallback_config = self.memory_manager._determine_quantization_config(
                required_memory=100,  # 大きな値を設定して強制的に最高圧縮
                available_memory=1,
                model_size=self.memory_manager.get_model_size(model_name),
                for_training=False
            )
            
            model_kwargs["quantization_config"] = fallback_config.to_bnb_config()
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = fallback_config.max_memory
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            logger.warning("Loaded model with fallback quantization settings")
            return model
    
    def load_finetuned_model(
        self,
        base_model_name: str,
        adapter_path: str,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_cpu: bool = False,
        for_training: bool = False
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """ファインチューニングモデル（LoRA/PEFT）を読み込み
        
        Args:
            base_model_name: ベースモデル名
            adapter_path: アダプターのパス
            use_auth_token: Hugging Face トークン
            cache_dir: キャッシュディレクトリ
            force_cpu: CPU強制使用フラグ
            for_training: トレーニング用フラグ
        
        Returns:
            PEFTモデルとトークナイザーのタプル
        """
        logger.info(f"Loading finetuned model: {base_model_name} + {adapter_path}")
        
        # ベースモデルを読み込み（トレーニング用フラグなし）
        base_model, tokenizer = self.load_base_model(
            base_model_name,
            use_auth_token,
            cache_dir,
            force_cpu,
            for_training=False  # アダプター読み込み時はベースモデルなので
        )
        
        # PEFTアダプターを読み込み
        logger.info(f"Loading PEFT adapter from: {adapter_path}")
        
        try:
            # アダプター設定を読み込み
            peft_config = PeftConfig.from_pretrained(adapter_path)
            
            # PEFTモデルを作成
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=base_model.dtype,
                device_map=base_model.hf_device_map if hasattr(base_model, 'hf_device_map') else None
            )
            
            # トレーニング用の設定
            if for_training:
                model.train()
            else:
                model.eval()
                # 推論用にマージを検討
                if hasattr(model, 'merge_and_unload'):
                    logger.info("Merging adapter weights for inference")
                    model = model.merge_and_unload()
            
            logger.info("PEFT model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading PEFT adapter: {e}")
            logger.warning("Returning base model without adapter")
            return base_model, tokenizer
    
    def load_full_finetuned_model(
        self,
        model_path: str,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_cpu: bool = False,
        for_training: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """フルファインチューニングモデルを読み込み
        
        Args:
            model_path: モデルのパス（ローカルまたはHugging Face）
            use_auth_token: Hugging Face トークン
            cache_dir: キャッシュディレクトリ
            force_cpu: CPU強制使用フラグ
            for_training: トレーニング用フラグ
        
        Returns:
            モデルとトークナイザーのタプル
        """
        # 通常のベースモデル読み込みと同じ処理
        return self.load_base_model(
            model_path,
            use_auth_token,
            cache_dir,
            force_cpu,
            for_training=for_training
        )
    
    def estimate_memory_requirements(
        self,
        model_name: str,
        batch_size: int = 1,
        sequence_length: int = 2048,
        for_training: bool = False
    ) -> Dict[str, Any]:
        """メモリ要件を推定
        
        Args:
            model_name: モデル名
            batch_size: バッチサイズ
            sequence_length: シーケンス長
            for_training: トレーニング用フラグ
        
        Returns:
            メモリ要件の詳細
        """
        if for_training:
            return self.memory_manager.get_training_memory_requirements(
                model_name,
                batch_size,
                sequence_length
            )
        else:
            # 推論時のメモリ要件（簡易版）
            model_size = self.memory_manager.get_model_size(model_name)
            base_memory = {
                "tiny": 1, "small": 3, "medium": 8,
                "large": 16, "xlarge": 32, "xxlarge": 64
            }.get(model_size.value, 16)
            
            activation_memory = (batch_size * sequence_length * 4096 * 2) / 1024**3
            
            return {
                "model": base_memory,
                "activations": activation_memory,
                "total": base_memory + activation_memory,
                "recommended_quantization": self.memory_manager._get_quantization_recommendation(
                    base_memory + activation_memory
                )
            }
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        # キャッシュされたモデルを削除
        for key in list(self.loaded_models.keys()):
            del self.loaded_models[key]
        
        # メモリクリア
        self.memory_manager.clear_gpu_memory(aggressive=True)
        
        logger.info("Model loader cleanup completed")


# 便利な関数
def create_model_loader(debug_mode: bool = False) -> OptimizedModelLoader:
    """モデルローダーを作成"""
    # 環境変数からデバッグモードを読み取り
    if not debug_mode:
        debug_mode = os.environ.get("MOE_RAG_DEBUG", "0") == "1"
    
    if debug_mode:
        logger.warning("Debug mode enabled - performance may be impacted")
    
    return OptimizedModelLoader(debug_mode=debug_mode)


# バックワード互換性のための関数
def load_base_model_with_optimization(
    model_name: str,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_cpu: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """既存コードとの互換性維持用"""
    loader = create_model_loader()
    return loader.load_base_model(
        model_name,
        use_auth_token,
        cache_dir,
        force_cpu
    )


def load_finetuned_model_with_optimization(
    base_model_name: str,
    model_path: str,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_cpu: bool = False
) -> Tuple[PeftModel, AutoTokenizer]:
    """既存コードとの互換性維持用"""
    loader = create_model_loader()
    return loader.load_finetuned_model(
        base_model_name,
        model_path,
        use_auth_token,
        cache_dir,
        force_cpu
    )
