"""
統一量子化設定マネージャー
すべてのモデル読み込み処理で使用される量子化設定を一元管理
"""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class UnifiedQuantizationConfig:
    """統一量子化設定クラス"""
    
    # 基本設定
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # 4bit量子化の詳細設定
    bnb_4bit_compute_dtype: Optional[torch.dtype] = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    
    # 8bit量子化の詳細設定
    bnb_8bit_compute_dtype: Optional[torch.dtype] = torch.float16
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[list] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    
    # デバイス設定
    device_map: Optional[str] = "auto"
    max_memory: Optional[Dict] = None
    offload_folder: Optional[str] = None
    offload_state_dict: bool = False
    
    # メモリ最適化設定
    low_cpu_mem_usage: bool = True
    torch_dtype: Optional[torch.dtype] = torch.float16
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        config_dict = {}
        
        # BitsAndBytes設定
        if self.load_in_4bit or self.load_in_8bit:
            config_dict["quantization_config"] = {}
            
            if self.load_in_4bit:
                config_dict["quantization_config"]["load_in_4bit"] = True
                config_dict["quantization_config"]["bnb_4bit_compute_dtype"] = str(self.bnb_4bit_compute_dtype).split('.')[-1]
                config_dict["quantization_config"]["bnb_4bit_quant_type"] = self.bnb_4bit_quant_type
                config_dict["quantization_config"]["bnb_4bit_use_double_quant"] = self.bnb_4bit_use_double_quant
            
            elif self.load_in_8bit:
                config_dict["quantization_config"]["load_in_8bit"] = True
                config_dict["quantization_config"]["bnb_8bit_compute_dtype"] = str(self.bnb_8bit_compute_dtype).split('.')[-1]
                config_dict["quantization_config"]["llm_int8_threshold"] = self.llm_int8_threshold
                config_dict["quantization_config"]["llm_int8_enable_fp32_cpu_offload"] = self.llm_int8_enable_fp32_cpu_offload
                
                if self.llm_int8_skip_modules:
                    config_dict["quantization_config"]["llm_int8_skip_modules"] = self.llm_int8_skip_modules
        
        # デバイス設定
        config_dict["device_map"] = self.device_map
        if self.max_memory:
            config_dict["max_memory"] = self.max_memory
        
        # メモリ最適化設定
        config_dict["low_cpu_mem_usage"] = self.low_cpu_mem_usage
        config_dict["torch_dtype"] = str(self.torch_dtype).split('.')[-1]
        
        # オフロード設定
        if self.offload_folder:
            config_dict["offload_folder"] = self.offload_folder
            config_dict["offload_state_dict"] = self.offload_state_dict
        
        return config_dict
    
    def to_transformers_config(self):
        """Transformers BitsAndBytesConfig形式に変換"""
        try:
            from transformers import BitsAndBytesConfig
            
            if self.load_in_4bit:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant
                )
            elif self.load_in_8bit:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.bnb_8bit_compute_dtype,
                    llm_int8_threshold=self.llm_int8_threshold,
                    llm_int8_skip_modules=self.llm_int8_skip_modules,
                    llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
                    llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight
                )
            return None
            
        except ImportError:
            logger.warning("BitsAndBytesConfig not available")
            return None
    
    @classmethod
    def from_model_name_and_gpu_memory(
        cls,
        model_name: str,
        available_gpu_memory_gb: float,
        for_training: bool = False
    ) -> "UnifiedQuantizationConfig":
        """モデル名とGPUメモリから最適な設定を生成"""
        
        # モデルサイズの推定
        model_size_gb = cls._estimate_model_size(model_name)
        
        # トレーニング時は3倍のメモリが必要（モデル、勾配、オプティマイザ）
        required_memory = model_size_gb * 3 if for_training else model_size_gb
        
        # メモリ比率による設定選択
        memory_ratio = available_gpu_memory_gb / required_memory
        
        if memory_ratio >= 1.5:
            # 十分なメモリ：量子化なし
            return cls(
                load_in_4bit=False,
                load_in_8bit=False,
                torch_dtype=torch.float16 if not for_training else torch.float32,
                device_map="auto"
            )
        
        elif memory_ratio >= 0.8:
            # 中程度：8bit量子化
            return cls(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                device_map="auto"
            )
        
        elif memory_ratio >= 0.4:
            # 不足：4bit量子化
            return cls(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False if for_training else True,
                device_map="auto"
            )
        
        else:
            # 深刻な不足：4bit + CPU オフロード
            max_memory = cls._calculate_memory_allocation(available_gpu_memory_gb)
            
            return cls(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="/tmp/offload",
                offload_state_dict=True
            )
    
    @staticmethod
    def _estimate_model_size(model_name: str) -> float:
        """モデル名からサイズを推定（GB）"""
        model_name_lower = model_name.lower()
        
        # パターンマッチング
        size_map = {
            "70b": 140, "65b": 130, "40b": 80, "32b": 64, "30b": 60,
            "22b": 44, "20b": 40, "15b": 30, "14b": 28, "13b": 26,
            "8b": 16, "7b": 14, "6b": 12, "4b": 8, "3b": 6,
            "2b": 4, "1b": 2, "1.5b": 3, "1.3b": 2.6,
            "560m": 1.1, "350m": 0.7, "125m": 0.25
        }
        
        for pattern, size_gb in size_map.items():
            if pattern in model_name_lower:
                return size_gb
        
        # デフォルトは7Bモデル相当
        return 14
    
    @staticmethod
    def _calculate_memory_allocation(available_gpu_gb: float) -> Dict:
        """メモリ割り当てを計算"""
        # GPUメモリの90%を使用
        gpu_memory = int(available_gpu_gb * 0.9 * 1024)  # MB単位
        
        # CPUメモリは物理メモリの30%まで
        try:
            import psutil
            cpu_memory_gb = psutil.virtual_memory().total / 1024**3
            cpu_memory = int(cpu_memory_gb * 0.3 * 1024)  # MB単位
        except:
            cpu_memory = 32768  # デフォルト32GB
        
        return {
            0: f"{gpu_memory}MB",
            "cpu": f"{cpu_memory}MB"
        }
    
    def save_to_file(self, filepath: Path):
        """設定をファイルに保存"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Quantization config saved to: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> "UnifiedQuantizationConfig":
        """ファイルから設定を読み込み"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # 量子化設定の解析
        quantization_config = config_dict.get("quantization_config", {})
        
        # データ型の変換
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        
        kwargs = {
            "load_in_4bit": quantization_config.get("load_in_4bit", False),
            "load_in_8bit": quantization_config.get("load_in_8bit", False),
            "device_map": config_dict.get("device_map", "auto"),
            "low_cpu_mem_usage": config_dict.get("low_cpu_mem_usage", True),
        }
        
        # dtypeの変換
        if "torch_dtype" in config_dict:
            kwargs["torch_dtype"] = dtype_map.get(config_dict["torch_dtype"], torch.float16)
        
        # 4bit設定
        if kwargs["load_in_4bit"]:
            kwargs["bnb_4bit_compute_dtype"] = dtype_map.get(
                quantization_config.get("bnb_4bit_compute_dtype", "float16"),
                torch.float16
            )
            kwargs["bnb_4bit_quant_type"] = quantization_config.get("bnb_4bit_quant_type", "nf4")
            kwargs["bnb_4bit_use_double_quant"] = quantization_config.get("bnb_4bit_use_double_quant", False)
        
        # 8bit設定
        if kwargs["load_in_8bit"]:
            kwargs["bnb_8bit_compute_dtype"] = dtype_map.get(
                quantization_config.get("bnb_8bit_compute_dtype", "float16"),
                torch.float16
            )
            kwargs["llm_int8_threshold"] = quantization_config.get("llm_int8_threshold", 6.0)
            kwargs["llm_int8_enable_fp32_cpu_offload"] = quantization_config.get(
                "llm_int8_enable_fp32_cpu_offload", False
            )
        
        # その他の設定
        if "max_memory" in config_dict:
            kwargs["max_memory"] = config_dict["max_memory"]
        if "offload_folder" in config_dict:
            kwargs["offload_folder"] = config_dict["offload_folder"]
            kwargs["offload_state_dict"] = config_dict.get("offload_state_dict", False)
        
        return cls(**kwargs)


class QuantizationConfigManager:
    """量子化設定の管理クラス"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: 設定ファイルを保存するディレクトリ
        """
        self.config_dir = config_dir or Path.home() / ".moe_rag" / "quantization_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # プリセット設定
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, UnifiedQuantizationConfig]:
        """プリセット設定を読み込み"""
        return {
            "cpu": UnifiedQuantizationConfig(
                load_in_4bit=False,
                load_in_8bit=False,
                torch_dtype=torch.float32,
                device_map="cpu"
            ),
            "fp16": UnifiedQuantizationConfig(
                load_in_4bit=False,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto"
            ),
            "int8": UnifiedQuantizationConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                device_map="auto"
            ),
            "int4": UnifiedQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                device_map="auto"
            ),
            "int4_double": UnifiedQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                device_map="auto"
            ),
            "int4_offload": UnifiedQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
                device_map="auto",
                offload_folder="/tmp/offload",
                offload_state_dict=True
            )
        }
    
    def get_preset(self, preset_name: str) -> Optional[UnifiedQuantizationConfig]:
        """プリセット設定を取得"""
        return self.presets.get(preset_name)
    
    def get_config_for_model(
        self,
        model_name: str,
        available_gpu_memory_gb: Optional[float] = None,
        for_training: bool = False,
        preset: Optional[str] = None
    ) -> UnifiedQuantizationConfig:
        """モデル用の設定を取得
        
        Args:
            model_name: モデル名
            available_gpu_memory_gb: 利用可能GPUメモリ（GB）
            for_training: トレーニング用フラグ
            preset: プリセット名（指定時は他の条件を無視）
        
        Returns:
            量子化設定
        """
        # プリセット指定がある場合
        if preset:
            config = self.get_preset(preset)
            if config:
                return config
            logger.warning(f"Preset '{preset}' not found, using auto detection")
        
        # GPUメモリの自動検出
        if available_gpu_memory_gb is None and torch.cuda.is_available():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                available_gpu_memory_gb = total_memory - reserved - 1  # 1GB安全マージン
            except:
                available_gpu_memory_gb = 8  # デフォルト値
        
        # 自動設定生成
        return UnifiedQuantizationConfig.from_model_name_and_gpu_memory(
            model_name,
            available_gpu_memory_gb or 8,
            for_training
        )
    
    def save_config(self, config: UnifiedQuantizationConfig, name: str):
        """設定を保存"""
        filepath = self.config_dir / f"{name}.json"
        config.save_to_file(filepath)
    
    def load_config(self, name: str) -> UnifiedQuantizationConfig:
        """設定を読み込み"""
        filepath = self.config_dir / f"{name}.json"
        return UnifiedQuantizationConfig.load_from_file(filepath)
    
    def list_saved_configs(self) -> list:
        """保存された設定の一覧"""
        configs = []
        for filepath in self.config_dir.glob("*.json"):
            configs.append(filepath.stem)
        return configs


# グローバルインスタンス
_config_manager = None

def get_quantization_config_manager() -> QuantizationConfigManager:
    """量子化設定マネージャーのシングルトンインスタンスを取得"""
    global _config_manager
    if _config_manager is None:
        _config_manager = QuantizationConfigManager()
    return _config_manager
