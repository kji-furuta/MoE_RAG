"""
Docker環境用メモリ管理パッチ
初期化の無限ループを防ぎ、Docker環境での動作を最適化
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Docker環境検出
def is_docker_environment() -> bool:
    """Docker環境かどうかを検出"""
    # Docker環境の一般的な識別子
    docker_indicators = [
        Path("/.dockerenv").exists(),
        os.environ.get("DOCKER_CONTAINER") == "true",
        "/docker" in Path("/proc/self/cgroup").read_text() if Path("/proc/self/cgroup").exists() else False,
        os.environ.get("HOSTNAME", "").startswith("ai-ft")
    ]
    return any(docker_indicators)


class DockerSafeMemoryManager:
    """Docker環境用の安全なメモリマネージャー"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 重複初期化を防ぐ
        if DockerSafeMemoryManager._initialized:
            return
        
        DockerSafeMemoryManager._initialized = True
        self.is_docker = is_docker_environment()
        self._setup_docker_safe_environment()
        logger.info(f"DockerSafeMemoryManager initialized (Docker: {self.is_docker})")
    
    def _setup_docker_safe_environment(self):
        """Docker環境用の安全な環境変数設定"""
        if self.is_docker:
            # Docker環境では控えめな設定
            safe_env_vars = {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                # CUDA_LAUNCH_BLOCKINGは設定しない（Docker内では問題を起こす可能性）
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                # メモリマネージャーの初期化フラグ
                "MEMORY_MANAGER_INITIALIZED": "1"
            }
        else:
            # 通常環境
            safe_env_vars = {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
                "CUDA_LAUNCH_BLOCKING": "0",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "MEMORY_MANAGER_INITIALIZED": "1"
            }
        
        # 既に設定されている場合はスキップ
        if os.environ.get("MEMORY_MANAGER_INITIALIZED") == "1":
            logger.debug("Environment already initialized, skipping")
            return
        
        for key, value in safe_env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.debug(f"Set {key}={value}")
    
    def clear_gpu_memory(self, aggressive: bool = False):
        """GPUメモリをクリア（Docker安全版）"""
        if not torch.cuda.is_available():
            return
        
        try:
            torch.cuda.empty_cache()
            if aggressive and not self.is_docker:
                # Docker環境では積極的なGCを避ける
                import gc
                gc.collect()
        except Exception as e:
            logger.warning(f"Memory clear failed: {e}")
    
    def get_optimal_quantization_config(
        self,
        model_name: str,
        available_memory_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Docker環境用の簡略化された量子化設定"""
        model_size = self._estimate_model_size(model_name)
        
        # Docker環境では保守的な設定
        if self.is_docker:
            if model_size > 30:  # 30GB以上のモデル
                return {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "device_map": "auto"
                }
            elif model_size > 14:  # 14GB以上のモデル
                return {
                    "load_in_8bit": True,
                    "bnb_8bit_compute_dtype": torch.float16,
                    "device_map": "auto"
                }
        
        # デフォルト設定
        return {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
    
    def _estimate_model_size(self, model_name: str) -> float:
        """モデルサイズの簡易推定"""
        model_name_lower = model_name.lower()
        
        size_map = {
            "70b": 140, "65b": 130, "32b": 64, "30b": 60,
            "22b": 44, "20b": 40, "14b": 28, "13b": 26,
            "8b": 16, "7b": 14, "3b": 6, "1b": 2
        }
        
        for pattern, size in size_map.items():
            if pattern in model_name_lower:
                return size
        return 14  # デフォルト


# グローバルインスタンス（シングルトン）
_docker_safe_manager = None

def get_docker_safe_memory_manager() -> DockerSafeMemoryManager:
    """Docker安全なメモリマネージャーを取得"""
    global _docker_safe_manager
    if _docker_safe_manager is None:
        _docker_safe_manager = DockerSafeMemoryManager()
    return _docker_safe_manager


# 既存のメモリ管理システムをオーバーライド
def patch_memory_manager_for_docker():
    """既存のメモリ管理をDocker用にパッチ"""
    try:
        # 既存のメモリマネージャーをインポート
        from src.core import memory_manager
        
        # Docker環境の場合はパッチを適用
        if is_docker_environment():
            logger.info("Applying Docker memory manager patch")
            
            # get_memory_manager関数を置き換え
            original_get_memory_manager = memory_manager.get_memory_manager
            
            def docker_safe_get_memory_manager(debug_mode: bool = False):
                """Docker安全版のメモリマネージャー取得"""
                # 初期化ループを防ぐ
                if os.environ.get("MEMORY_MANAGER_INITIALIZED") == "1":
                    return get_docker_safe_memory_manager()
                
                # 初回のみオリジナルを呼び出し
                os.environ["MEMORY_MANAGER_INITIALIZED"] = "1"
                return get_docker_safe_memory_manager()
            
            memory_manager.get_memory_manager = docker_safe_get_memory_manager
            logger.info("Docker memory manager patch applied successfully")
        
    except ImportError:
        logger.warning("Memory manager module not found, using Docker safe defaults")
    except Exception as e:
        logger.error(f"Failed to patch memory manager: {e}")


# 起動時に自動パッチ
if is_docker_environment():
    patch_memory_manager_for_docker()
