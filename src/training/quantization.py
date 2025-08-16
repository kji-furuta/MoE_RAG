import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import bitsandbytes as bnb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QuantizationOptimizer:
    """モデルの量子化と最適化ツール"""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[torch.device] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
    
    def quantize_to_8bit(
        self,
        output_dir: str,
        compute_dtype: torch.dtype = torch.float16,
        llm_int8_threshold: float = 6.0,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False
    ) -> AutoModelForCausalLM:
        """モデルを8bit量子化する"""
        logger.info(f"Quantizing {self.model_name_or_path} to 8-bit...")
        
        # 8bit量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=compute_dtype,
            llm_int8_threshold=llm_int8_threshold,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
        )
        
        # モデルのロード
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # 量子化統計の計算
        self._compute_quantization_stats()
        
        # 量子化モデルの保存
        self._save_quantized_model(output_dir, "8bit")
        
        logger.info("8-bit quantization completed!")
        return self.model
    
    def quantize_to_4bit(
        self,
        output_dir: str,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
        use_double_quant: bool = True
    ) -> AutoModelForCausalLM:
        """モデルを4bit量子化する"""
        logger.info(f"Quantizing {self.model_name_or_path} to 4-bit...")
        
        # 4bit量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant
        )
        
        # モデルのロード
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # 量子化統計の計算
        self._compute_quantization_stats()
        
        # 量子化モデルの保存
        self._save_quantized_model(output_dir, "4bit")
        
        logger.info("4-bit quantization completed!")
        return self.model
    
    def dynamic_quantization(
        self,
        model: nn.Module,
        output_dir: str,
        quantize_layers: Optional[List[str]] = None
    ) -> nn.Module:
        """動的量子化を適用する (INT8)"""
        logger.info("Applying dynamic quantization...")
        
        if quantize_layers is None:
            # デフォルトでLinear層を量子化
            quantize_layers = [nn.Linear]
        
        # 動的量子化を適用
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={layer: torch.quantization.default_dynamic_qconfig for layer in quantize_layers},
            dtype=torch.qint8
        )
        
        # モデルサイズの比較
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")
        
        # モデルの保存
        torch.save(quantized_model.state_dict(), os.path.join(output_dir, "dynamic_quantized_model.pt"))
        
        return quantized_model
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        backend: str = "fbgemm"  # or "qnnpack" for mobile
    ) -> torch.jit.ScriptModule:
        """推論用にモデルを最適化する (量子化 + JIT)"""
        logger.info("Optimizing model for inference...")
        
        # 推論モードに設定
        model.eval()
        
        # 量子化設定の準備
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)
        
        # キャリブレーションデータの提供
        with torch.no_grad():
            model(example_inputs)
        
        # 量子化の実行
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        # JITトレース
        traced_model = torch.jit.trace(quantized_model, example_inputs)
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        return optimized_model
    
    def benchmark_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_inputs: List[torch.Tensor],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """量子化モデルのベンチマーク"""
        import time
        
        logger.info("Benchmarking quantization...")
        
        results = {
            "original": {"times": [], "memory": 0},
            "quantized": {"times": [], "memory": 0}
        }
        
        # オリジナルモデルのベンチマーク
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for _ in tqdm(range(num_runs), desc="Original model"):
            start_time = time.time()
            with torch.no_grad():
                for inp in test_inputs:
                    _ = original_model(inp)
            torch.cuda.synchronize()
            results["original"]["times"].append(time.time() - start_time)
        
        results["original"]["memory"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # 量子化モデルのベンチマーク
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for _ in tqdm(range(num_runs), desc="Quantized model"):
            start_time = time.time()
            with torch.no_grad():
                for inp in test_inputs:
                    _ = quantized_model(inp)
            torch.cuda.synchronize()
            results["quantized"]["times"].append(time.time() - start_time)
        
        results["quantized"]["memory"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # 結果の計算
        for model_type in ["original", "quantized"]:
            times = results[model_type]["times"]
            results[model_type]["mean_time"] = np.mean(times)
            results[model_type]["std_time"] = np.std(times)
        
        # 結果の表示
        speedup = results["original"]["mean_time"] / results["quantized"]["mean_time"]
        memory_reduction = results["original"]["memory"] / results["quantized"]["memory"]
        
        logger.info(f"\nBenchmark Results:")
        logger.info(f"Original - Mean time: {results['original']['mean_time']:.4f}s, Memory: {results['original']['memory']:.2f}GB")
        logger.info(f"Quantized - Mean time: {results['quantized']['mean_time']:.4f}s, Memory: {results['quantized']['memory']:.2f}GB")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Memory reduction: {memory_reduction:.2f}x")
        
        return results
    
    def _compute_quantization_stats(self):
        """量子化統計を計算する"""
        if self.model is None:
            return
        
        total_params = 0
        quantized_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # 量子化されたパラメータをカウント
            if hasattr(param, "dtype") and param.dtype in [torch.int8, torch.uint8]:
                quantized_params += param.numel()
        
        quantization_ratio = quantized_params / total_params if total_params > 0 else 0
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Quantized parameters: {quantized_params:,}")
        logger.info(f"Quantization ratio: {quantization_ratio:.2%}")
    
    def _save_quantized_model(self, output_dir: str, quantization_type: str):
        """量子化モデルを保存する"""
        os.makedirs(output_dir, exist_ok=True)
        
        # モデルとトークナイザーの保存
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 量子化情報の保存
        quantization_info = {
            "quantization_type": quantization_type,
            "model_name": self.model_name_or_path,
            "device": str(self.device),
            "timestamp": str(torch.cuda.Event(enable_timing=True).record())
        }
        
        import json
        with open(os.path.join(output_dir, "quantization_info.json"), "w") as f:
            json.dump(quantization_info, f, indent=2)
        
        logger.info(f"Quantized model saved to {output_dir}")
    
    def _get_model_size(self, model: nn.Module) -> float:
        """モデルサイズを取得する (MB)"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def load_quantized_model(
        model_path: str,
        device: Optional[torch.device] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """量子化モデルをロードする"""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 量子化情報の読み込み
        import json
        info_path = os.path.join(model_path, "quantization_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                quantization_info = json.load(f)
            logger.info(f"Loading {quantization_info['quantization_type']} quantized model")
        
        # モデルとトークナイザーのロード
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        return model, tokenizer


class QuantizationAwareTraining:
    """量子化認識トレーニング (QAT)"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    def prepare_qat(self) -> nn.Module:
        """QAT用にモデルを準備する"""
        self.model.train()
        
        # QAT設定
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Fuse modules
        torch.quantization.fuse_modules(self.model, [['conv', 'bn', 'relu']], inplace=True)
        
        # Prepare for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        return self.model
    
    def convert_to_quantized(self) -> nn.Module:
        """QAT後にモデルを量子化する"""
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model