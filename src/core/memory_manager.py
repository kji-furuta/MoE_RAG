"""
統一メモリ管理システム
GPUメモリの監視、最適化、量子化設定の一元管理
"""

import os
import gc
import psutil
import torch
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)

# プロダクション環境用の最適化設定
PRODUCTION_ENV_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
    "CUDA_LAUNCH_BLOCKING": "0",  # 本番環境では必ず0（並列実行を有効化）
    "PYTORCH_CUDA_ALLOC_CONF_MAX_SPLIT_SIZE_MB": "512",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}

# デバッグ環境用の設定
DEBUG_ENV_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
    "CUDA_LAUNCH_BLOCKING": "1",  # デバッグ時のみ1（同期実行）
    "PYTORCH_CUDA_ALLOC_CONF_MAX_SPLIT_SIZE_MB": "256",
}


class ModelSize(Enum):
    """モデルサイズの分類"""
    TINY = "tiny"      # < 1B
    SMALL = "small"    # 1B-3B
    MEDIUM = "medium"  # 3B-7B
    LARGE = "large"    # 7B-15B
    XLARGE = "xlarge"  # 15B-30B
    XXLARGE = "xxlarge"  # > 30B


class QuantizationType(Enum):
    """量子化タイプ"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4bit Normal Float
    FP4 = "fp4"  # 4bit Floating Point


@dataclass
class GPUMemoryInfo:
    """GPU メモリ情報"""
    device_id: int
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization_percent: float
    
    @property
    def available_gb(self) -> float:
        """実際に使用可能なメモリ（GB）"""
        return self.free_gb - 0.5  # 安全マージン0.5GB


@dataclass
class QuantizationConfig:
    """量子化設定"""
    type: QuantizationType
    compute_dtype: torch.dtype
    use_double_quant: bool = False
    use_cpu_offload: bool = False
    device_map: Optional[str] = None
    max_memory: Optional[Dict[int, str]] = None
    
    def to_bnb_config(self):
        """BitsAndBytesConfig形式に変換"""
        from transformers import BitsAndBytesConfig
        
        if self.type == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self.compute_dtype,
                llm_int8_enable_fp32_cpu_offload=self.use_cpu_offload
            )
        elif self.type in [QuantizationType.INT4, QuantizationType.NF4]:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_type="nf4" if self.type == QuantizationType.NF4 else "fp4"
            )
        return None


class MemoryManager:
    """統一メモリ管理クラス"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: デバッグモードフラグ（本番環境では必ずFalse）
        """
        self.debug_mode = debug_mode
        self._setup_environment()
        self._config_cache = {}
        
        # 設定ファイルのパス
        self.config_file = Path.home() / ".moe_rag" / "memory_config.json"
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 設定を読み込み
        self.load_config()
    
    def _setup_environment(self):
        """環境変数の設定"""
        env_vars = DEBUG_ENV_VARS if self.debug_mode else PRODUCTION_ENV_VARS
        
        for key, value in env_vars.items():
            current_value = os.environ.get(key)
            if current_value != value:
                logger.info(f"設定環境変数 {key}={value} (以前: {current_value})")
                os.environ[key] = value
        
        # CUDA_LAUNCH_BLOCKING の警告
        if not self.debug_mode and os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            warnings.warn(
                "CUDA_LAUNCH_BLOCKING=1 が本番環境で設定されています。"
                "パフォーマンスが低下する可能性があります。",
                RuntimeWarning
            )
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Optional[GPUMemoryInfo]:
        """GPU メモリ情報を取得"""
        if not torch.cuda.is_available():
            return None
        
        try:
            # 基本情報
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            free = total - reserved
            
            # 使用率
            utilization = (allocated / total * 100) if total > 0 else 0
            
            return GPUMemoryInfo(
                device_id=device_id,
                total_gb=round(total, 2),
                allocated_gb=round(allocated, 2),
                reserved_gb=round(reserved, 2),
                free_gb=round(free, 2),
                utilization_percent=round(utilization, 2)
            )
        except Exception as e:
            logger.error(f"GPU メモリ情報取得エラー: {e}")
            return None
    
    def clear_gpu_memory(self, aggressive: bool = False):
        """GPU メモリをクリア
        
        Args:
            aggressive: より積極的なメモリクリア
        """
        if torch.cuda.is_available():
            # 基本的なメモリクリア
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if aggressive:
                # Pythonのガベージコレクション
                gc.collect()
                
                # 追加のメモリクリア試行
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # メモリ情報をログ
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                logger.info(f"GPU メモリクリア後: {memory_info.free_gb:.2f}GB 利用可能")
    
    def get_model_size(self, model_name: str) -> ModelSize:
        """モデル名からモデルサイズを推定"""
        model_name_lower = model_name.lower()
        
        # パターンマッチングによるサイズ判定
        size_patterns = {
            ModelSize.TINY: ["125m", "350m", "560m"],
            ModelSize.SMALL: ["1b", "1.3b", "1.5b", "2b", "2.7b"],
            ModelSize.MEDIUM: ["3b", "4b", "6b", "6.7b", "7b"],
            ModelSize.LARGE: ["8b", "13b", "14b", "15b"],
            ModelSize.XLARGE: ["20b", "22b", "25b", "30b"],
            ModelSize.XXLARGE: ["32b", "40b", "65b", "70b", "175b"]
        }
        
        for size, patterns in size_patterns.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    return size
        
        # デフォルトはMEDIUM
        logger.warning(f"モデルサイズを特定できません: {model_name}. MEDIUMとして扱います。")
        return ModelSize.MEDIUM
    
    def get_optimal_quantization(
        self,
        model_name: str,
        available_memory_gb: Optional[float] = None,
        target_batch_size: int = 1,
        for_training: bool = False
    ) -> QuantizationConfig:
        """最適な量子化設定を決定
        
        Args:
            model_name: モデル名
            available_memory_gb: 利用可能メモリ（None の場合は自動検出）
            target_batch_size: 目標バッチサイズ
            for_training: トレーニング用かどうか（メモリ要求が異なる）
        
        Returns:
            最適な量子化設定
        """
        # キャッシュチェック
        cache_key = f"{model_name}_{available_memory_gb}_{target_batch_size}_{for_training}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # メモリ情報取得
        if available_memory_gb is None:
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                available_memory_gb = memory_info.available_gb
            else:
                # GPU が利用できない場合はCPU処理
                return QuantizationConfig(
                    type=QuantizationType.INT8,
                    compute_dtype=torch.float32,
                    use_cpu_offload=True,
                    device_map="cpu"
                )
        
        # モデルサイズを取得
        model_size = self.get_model_size(model_name)
        
        # トレーニング時は追加メモリが必要（勾配、オプティマイザ状態）
        memory_multiplier = 3.0 if for_training else 1.0
        
        # バッチサイズによる追加メモリ要求
        batch_memory_factor = 1 + (target_batch_size - 1) * 0.3
        
        # 必要メモリの推定（GB）
        estimated_memory_requirements = {
            ModelSize.TINY: 1 * memory_multiplier * batch_memory_factor,
            ModelSize.SMALL: 3 * memory_multiplier * batch_memory_factor,
            ModelSize.MEDIUM: 8 * memory_multiplier * batch_memory_factor,
            ModelSize.LARGE: 16 * memory_multiplier * batch_memory_factor,
            ModelSize.XLARGE: 32 * memory_multiplier * batch_memory_factor,
            ModelSize.XXLARGE: 64 * memory_multiplier * batch_memory_factor,
        }
        
        required_memory = estimated_memory_requirements.get(model_size, 16)
        
        # 量子化設定の決定
        config = self._determine_quantization_config(
            required_memory,
            available_memory_gb,
            model_size,
            for_training
        )
        
        # キャッシュに保存
        self._config_cache[cache_key] = config
        
        return config
    
    def _determine_quantization_config(
        self,
        required_memory: float,
        available_memory: float,
        model_size: ModelSize,
        for_training: bool
    ) -> QuantizationConfig:
        """量子化設定を決定する内部メソッド"""
        
        # メモリ比率
        memory_ratio = available_memory / required_memory if required_memory > 0 else 0
        
        # 十分なメモリがある場合（比率 > 1.2）
        if memory_ratio > 1.2:
            return QuantizationConfig(
                type=QuantizationType.NONE,
                compute_dtype=torch.float16,
                device_map="auto"
            )
        
        # 適度なメモリ（0.7 < 比率 <= 1.2）
        elif memory_ratio > 0.7:
            return QuantizationConfig(
                type=QuantizationType.INT8,
                compute_dtype=torch.float16,
                device_map="auto"
            )
        
        # メモリ不足（0.4 < 比率 <= 0.7）
        elif memory_ratio > 0.4:
            # トレーニング時はダブル量子化を避ける（精度低下を防ぐ）
            return QuantizationConfig(
                type=QuantizationType.NF4,
                compute_dtype=torch.float16,
                use_double_quant=not for_training,
                device_map="auto"
            )
        
        # 深刻なメモリ不足（比率 <= 0.4）
        else:
            # CPU オフロードを有効化
            max_memory = self._calculate_max_memory(available_memory)
            
            return QuantizationConfig(
                type=QuantizationType.NF4,
                compute_dtype=torch.float16,
                use_double_quant=True,
                use_cpu_offload=True,
                device_map="auto",
                max_memory=max_memory
            )
    
    def _calculate_max_memory(self, available_gpu_memory: float) -> Dict[int, str]:
        """デバイスごとの最大メモリを計算"""
        # GPUメモリを安全に使用（90%まで）
        gpu_memory = f"{int(available_gpu_memory * 0.9 * 1024)}MB"
        
        # CPUメモリ（システムメモリの30%まで）
        cpu_memory = psutil.virtual_memory().total
        cpu_memory_gb = cpu_memory / 1024**3
        cpu_memory_str = f"{int(cpu_memory_gb * 0.3 * 1024)}MB"
        
        return {
            0: gpu_memory,  # GPU 0
            "cpu": cpu_memory_str
        }
    
    def monitor_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用状況を監視"""
        result = {
            "gpu": None,
            "cpu": None,
            "recommendations": []
        }
        
        # GPU メモリ
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            result["gpu"] = {
                "total_gb": gpu_info.total_gb,
                "used_gb": gpu_info.allocated_gb,
                "free_gb": gpu_info.free_gb,
                "utilization_percent": gpu_info.utilization_percent
            }
            
            # 警告とレコメンデーション
            if gpu_info.utilization_percent > 90:
                result["recommendations"].append(
                    "GPU メモリ使用率が高い：量子化の使用を検討してください"
                )
            elif gpu_info.free_gb < 2:
                result["recommendations"].append(
                    "GPU メモリ残量が少ない：メモリクリアを実行してください"
                )
        
        # CPU メモリ
        cpu_memory = psutil.virtual_memory()
        result["cpu"] = {
            "total_gb": round(cpu_memory.total / 1024**3, 2),
            "used_gb": round(cpu_memory.used / 1024**3, 2),
            "free_gb": round(cpu_memory.available / 1024**3, 2),
            "percent": cpu_memory.percent
        }
        
        if cpu_memory.percent > 90:
            result["recommendations"].append(
                "システムメモリ使用率が高い：不要なプロセスを終了してください"
            )
        
        return result
    
    def save_config(self):
        """現在の設定を保存"""
        config = {
            "debug_mode": self.debug_mode,
            "env_vars": dict(os.environ),
            "timestamp": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"設定を保存しました: {self.config_file}")
    
    def load_config(self):
        """設定を読み込み"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"設定を読み込みました: {self.config_file}")
                return config
            except Exception as e:
                logger.warning(f"設定読み込みエラー: {e}")
        return None
    
    def get_training_memory_requirements(
        self,
        model_name: str,
        batch_size: int = 1,
        sequence_length: int = 2048,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """トレーニング時のメモリ要件を推定
        
        Returns:
            必要メモリの推定値（GB）
        """
        model_size = self.get_model_size(model_name)
        
        # 基本モデルサイズ（GB）
        base_sizes = {
            ModelSize.TINY: 0.5,
            ModelSize.SMALL: 2,
            ModelSize.MEDIUM: 6,
            ModelSize.LARGE: 14,
            ModelSize.XLARGE: 28,
            ModelSize.XXLARGE: 64
        }
        
        base_memory = base_sizes.get(model_size, 14)
        
        # 各コンポーネントのメモリ要求
        model_memory = base_memory
        gradient_memory = base_memory  # 勾配用
        optimizer_memory = base_memory * 2  # Adam の場合（momentum + variance）
        activation_memory = (batch_size * sequence_length * 4096 * 4) / 1024**3  # 推定
        
        # 効果的なバッチサイズ
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        total_memory = (
            model_memory +
            gradient_memory +
            optimizer_memory +
            activation_memory * effective_batch_size
        )
        
        return {
            "model": model_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "activations": activation_memory * effective_batch_size,
            "total": total_memory,
            "recommended_quantization": self._get_quantization_recommendation(total_memory)
        }
    
    def _get_quantization_recommendation(self, required_memory: float) -> str:
        """必要メモリに基づいて量子化の推奨を返す"""
        gpu_info = self.get_gpu_memory_info()
        if not gpu_info:
            return "GPU が利用できません。CPU処理を使用してください。"
        
        available = gpu_info.available_gb
        
        if available >= required_memory:
            return "量子化不要（十分なメモリ）"
        elif available >= required_memory * 0.5:
            return "INT8 量子化を推奨"
        elif available >= required_memory * 0.25:
            return "INT4/NF4 量子化を推奨"
        else:
            return "INT4 + CPU オフロードを推奨"


# シングルトンインスタンス
_memory_manager = None

def get_memory_manager(debug_mode: bool = False) -> MemoryManager:
    """メモリマネージャーのシングルトンインスタンスを取得"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(debug_mode=debug_mode)
    return _memory_manager
