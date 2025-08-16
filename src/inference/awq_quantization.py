"""
AWQ (Activation-aware Weight Quantization) Implementation
4bit量子化でメモリ使用量を50%削減
"""

import torch
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AWQQuantizer:
    """AWQ量子化器"""
    
    def __init__(
        self,
        model_path: str,
        quantization_bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
    ):
        self.model_path = model_path
        self.quantization_bits = quantization_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.awq_available = False
        
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
            self.AutoAWQForCausalLM = AutoAWQForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.awq_available = True
            logger.info("AWQ library imported")
        except ImportError:
            logger.warning("AWQ not installed. Install with: pip install autoawq")
    
    def quantize_model(
        self,
        output_path: str,
        calibration_data: Optional[List[str]] = None,
    ) -> str:
        """モデル量子化"""
        if not self.awq_available:
            raise RuntimeError("AWQ not installed")
            
        logger.info(f"Quantizing {self.model_path} to {self.quantization_bits} bits")
        
        # モデルロード
        model = self.AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
        )
        tokenizer = self.AutoTokenizer.from_pretrained(self.model_path)
        
        # デフォルトキャリブレーションデータ
        if calibration_data is None:
            calibration_data = self.get_default_calibration_data()
        
        # 量子化設定
        quant_config = {
            "zero_point": self.zero_point,
            "q_group_size": self.group_size,
            "w_bit": self.quantization_bits,
            "version": "GEMM",
        }
        
        # 量子化実行
        logger.info("Quantizing... This may take 30-60 minutes")
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)
        
        # 保存
        logger.info(f"Saving to {output_path}")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)
        
        # 設定保存
        config_path = Path(output_path) / "quantization_config.json"
        with open(config_path, 'w') as f:
            json.dump(quant_config, f, indent=2)
        
        self._estimate_memory_usage(output_path)
        
        return output_path
    
    def get_default_calibration_data(self) -> List[str]:
        """土木・建設分野のキャリブレーションデータ"""
        return [
            "道路設計における最小曲線半径の計算方法について説明してください。",
            "コンクリートの圧縮強度試験の手順を教えてください。",
            "土木工学における地盤改良工法の種類と特徴を述べてください。",
            "橋梁の耐震設計における重要な考慮事項は何ですか？",
            "建設プロジェクトの工程管理手法について説明してください。",
            "トンネル掘削におけるNATM工法の基本原理を教えてください。",
            "アスファルト舗装の品質管理項目を列挙してください。",
            "鉄筋コンクリート構造の設計手法について述べてください。",
        ]
    
    def load_quantized_model(self, model_path: str):
        """量子化モデルのロード"""
        if not self.awq_available:
            raise RuntimeError("AWQ not installed")
            
        model = self.AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            device_map="auto",
        )
        tokenizer = self.AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    
    def _estimate_memory_usage(self, model_path: str):
        """メモリ使用量推定"""
        import os
        
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.bin', '.safetensors')):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        size_gb = total_size / (1024**3)
        
        logger.info(f"Quantized model size: {size_gb:.2f} GB")
        logger.info(f"Original 32B model: ~64 GB")
        logger.info(f"Memory reduction: {(1 - size_gb/64)*100:.1f}%")
        logger.info(f"Fits in A5000×2: {'Yes' if size_gb < 45 else 'No'}")


class AWQInferenceOptimizer:
    """AWQ最適化推論"""
    
    def __init__(self, quantized_model_path: str, use_vllm: bool = True):
        self.quantized_model_path = quantized_model_path
        self.use_vllm = use_vllm
        
        if use_vllm:
            from .vllm_integration import vLLMConfig, VLLMInferenceEngine
            
            config = vLLMConfig(
                model_name_or_path=quantized_model_path,
                tensor_parallel_size=2,
                quantization="awq",
                dtype="float16",
            )
            
            self.engine = VLLMInferenceEngine(config)
            logger.info("AWQ + vLLM initialized")
    
    def get_expected_improvements(self) -> Dict[str, str]:
        """期待される改善"""
        return {
            "memory": "50% reduction",
            "speed": "1.2-1.5x faster",
            "accuracy": "97-98% retained",
            "batch_size": "4x larger",
        }
