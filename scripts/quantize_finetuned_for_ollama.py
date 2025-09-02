#!/usr/bin/env python3
"""
RAGシステムで設定されたファインチューニング済みモデル（LoRAアダプタ）を
量子化してOllamaで使用可能にするスクリプト

問題点の整理:
1. 設定されているモデル: outputs/lora_20250829_170202（存在しない）
2. 実際に存在するモデル:
   - lora_20250830_223432 (DeepSeek-R1-Distill-Qwen-32B)
   - lora_20250831_122140 (Qwen2.5-32B-Instruct)
3. ベースモデル設定: cyberagent/calm3-22b-chat（不一致）

解決方法:
1. LoRAアダプタとベースモデルをマージ
2. マージしたモデルを量子化
3. Ollamaで使用可能な形式に変換
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class FinetunedModelQuantizer:
    """ファインチューニング済みモデル（LoRA）の量子化とOllama統合"""
    
    def __init__(self, 
                 lora_path: str,
                 output_dir: str = "./outputs/quantized_finetuned",
                 quantization_level: str = "Q4_K_M"):
        """
        Args:
            lora_path: LoRAアダプタのパス
            output_dir: 量子化モデルの出力先
            quantization_level: 量子化レベル
        """
        self.lora_path = Path(lora_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quantization_level = quantization_level
        
        # LoRA設定を読み込み
        self.lora_config = self._load_lora_config()
        self.base_model_name = self.lora_config.get("base_model_name_or_path")
        
    def _load_lora_config(self) -> Dict[str, Any]:
        """LoRAアダプタの設定を読み込み"""
        config_path = self.lora_path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"LoRA config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def analyze_model_compatibility(self) -> Tuple[bool, str]:
        """モデルの互換性を分析"""
        logger.info("=== モデル互換性分析 ===")
        
        # ベースモデルのサイズを確認
        model_size_map = {
            "cyberagent/calm3-22b-chat": 22,
            "Qwen/Qwen2.5-32B-Instruct": 32,
            "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese": 32
        }
        
        base_size = model_size_map.get(self.base_model_name, 0)
        logger.info(f"Base model: {self.base_model_name}")
        logger.info(f"Model size: {base_size}B parameters")
        
        # メモリ要件を計算
        memory_requirements = {
            "FP16": base_size * 2,  # GB
            "INT8": base_size,       # GB
            "INT4": base_size / 2    # GB
        }
        
        logger.info("メモリ要件:")
        for dtype, mem in memory_requirements.items():
            logger.info(f"  {dtype}: ~{mem}GB")
            
        # 量子化推奨レベルを決定
        if base_size >= 30:
            recommendation = "Q4_K_M"
            logger.info("推奨: Q4_K_M (4-bit量子化) - 大規模モデルのため")
        elif base_size >= 13:
            recommendation = "Q5_K_M"
            logger.info("推奨: Q5_K_M (5-bit量子化) - 中規模モデル")
        else:
            recommendation = "Q8_0"
            logger.info("推奨: Q8_0 (8-bit量子化) - 小規模モデル")
            
        return True, recommendation
        
    def merge_lora_with_base(self) -> Path:
        """LoRAアダプタをベースモデルとマージ（マルチGPU対応）"""
        logger.info("=== LoRAアダプタのマージ（マルチGPU） ===")
        
        merged_path = self.output_dir / "merged_model"
        merged_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # マルチGPU設定を確認
            num_gpus = torch.cuda.device_count()
            logger.info(f"Number of GPUs available: {num_gpus}")
            
            total_gpu_memory = 0
            gpu_memory_map = {}
            for i in range(num_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_memory:.2f} GB")
                total_gpu_memory += gpu_memory
                # 各GPUに20GBを割り当て（バッファを残す）
                gpu_memory_map[i] = "20GiB"
            
            logger.info(f"Total GPU memory: {total_gpu_memory:.2f} GB")
            
            # メモリマップを設定（マルチGPU用）
            max_memory = gpu_memory_map.copy()
            max_memory["cpu"] = "100GiB"  # CPUメモリも使用
            
            # 8-bit量子化でロード（メモリ節約、CPUオフロード有効）
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True  # CPUオフロードを有効化
            )
            
            logger.info(f"Loading base model with multi-GPU: {self.base_model_name}")
            logger.info(f"Memory allocation: {max_memory}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="balanced_low_0",  # マルチGPU用のバランス配置
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory=max_memory,  # マルチGPUメモリ制限
                offload_folder="/tmp/offload"  # オフロードフォルダ
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            # LoRAアダプタをロード
            logger.info(f"Loading LoRA adapter from: {self.lora_path}")
            model = PeftModel.from_pretrained(
                base_model,
                str(self.lora_path),
                torch_dtype=torch.float16
            )
            
            # マージ実行
            logger.info("Merging LoRA with base model...")
            merged_model = model.merge_and_unload()
            
            # 保存
            logger.info(f"Saving merged model to: {merged_path}")
            merged_model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            
            return merged_path
            
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")
            raise
            
    def quantize_merged_model(self, merged_path: Path) -> Path:
        """マージしたモデルを量子化"""
        logger.info("=== モデルの量子化 ===")
        
        quantized_path = self.output_dir / f"quantized_{self.quantization_level}"
        quantized_path.mkdir(parents=True, exist_ok=True)
        
        # 量子化方法の選択
        if self.quantization_level == "Q4_K_M":
            return self._quantize_4bit(merged_path, quantized_path)
        elif self.quantization_level == "Q8_0":
            return self._quantize_8bit(merged_path, quantized_path)
        else:
            raise ValueError(f"Unsupported quantization: {self.quantization_level}")
            
    def _quantize_4bit(self, input_path: Path, output_path: Path) -> Path:
        """4-bit量子化（マルチGPU対応）"""
        logger.info("Applying 4-bit quantization with multi-GPU support...")
        
        # マルチGPU設定
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs for quantization: {num_gpus}")
        
        max_memory = {}
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i} memory: {gpu_memory:.2f} GB")
            max_memory[i] = "20GiB"  # 各GPUに20GB割り当て
        max_memory["cpu"] = "100GiB"
        
        # CPU offloadを有効にした量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # CPUオフロードを有効化
        )
        
        try:
            logger.info("Loading model for quantization with multi-GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                input_path,
                quantization_config=bnb_config,
                device_map="balanced",  # マルチGPUでバランス配置
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory=max_memory,
                offload_folder="/tmp/offload",
                offload_state_dict=True
            )
        except Exception as e:
            logger.warning(f"Failed with balanced device_map: {e}")
            logger.info("Retrying with sequential device_map...")
            
            # sequential配置で再試行
            model = AutoModelForCausalLM.from_pretrained(
                input_path,
                quantization_config=bnb_config,
                device_map="sequential",  # 順次配置
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory=max_memory,
                offload_folder="/tmp/offload"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            input_path,
            trust_remote_code=True
        )
        
        # 保存
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        return output_path
        
    def _quantize_8bit(self, input_path: Path, output_path: Path) -> Path:
        """8-bit量子化（マルチGPU対応）"""
        logger.info("Applying 8-bit quantization with multi-GPU support...")
        
        # マルチGPU設定
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs for quantization: {num_gpus}")
        
        max_memory = {}
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i} memory: {gpu_memory:.2f} GB")
            max_memory[i] = "20GiB"  # 各GPUに20GB割り当て
        max_memory["cpu"] = "100GiB"
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # CPUオフロードを有効化
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            quantization_config=bnb_config,
            device_map="balanced",  # マルチGPUでバランス配置
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory=max_memory,
            offload_folder="/tmp/offload"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            input_path,
            trust_remote_code=True
        )
        
        # 保存
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        return output_path
        
    def convert_to_gguf(self, quantized_path: Path) -> Path:
        """GGUF形式に変換（Ollama用）"""
        logger.info("=== GGUF形式への変換 ===")
        
        gguf_path = self.output_dir / "gguf"
        gguf_path.mkdir(parents=True, exist_ok=True)
        
        # 変換スクリプトを生成
        convert_script = self.output_dir / "convert_to_gguf.sh"
        script_content = f"""#!/bin/bash
# GGUF変換スクリプト

# llama.cppのセットアップ
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    cd ..
fi

# 変換実行
python llama.cpp/convert.py {quantized_path} \\
    --outfile {gguf_path}/model.gguf \\
    --outtype {self.quantization_level.lower()}

echo "GGUF conversion complete!"
"""
        
        convert_script.write_text(script_content)
        convert_script.chmod(0o755)
        
        logger.info(f"GGUF conversion script created: {convert_script}")
        logger.info("Run the script manually to convert to GGUF format")
        
        return gguf_path
        
    def create_ollama_modelfile(self, model_name: str) -> Path:
        """Ollama用のModelfileを作成"""
        modelfile_content = f"""# Finetuned Model for RAG System
FROM ./gguf/model.gguf

# Model configuration for road design Q&A
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"

# System prompt
SYSTEM "あなたは道路設計の専門家です。日本の道路構造令と設計基準に基づいて、正確で詳細な技術的回答を提供してください。"

# Model information
# Base: {self.base_model_name}
# LoRA: {self.lora_path.name}
# Quantization: {self.quantization_level}
"""
        
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        logger.info(f"Created Modelfile at {modelfile_path}")
        
        return modelfile_path
        
    def update_rag_config(self, model_name: str):
        """RAGシステムの設定を更新"""
        logger.info("=== RAG設定の更新方法 ===")
        
        config_update = f"""
# src/rag/config/rag_config.yamlを以下のように更新:

llm:
  use_finetuned: false  # Ollamaを使用するため
  use_ollama_fallback: true
  ollama_model: {model_name}
  ollama_host: http://localhost:11434
  
# または環境変数で設定:
export OLLAMA_MODEL={model_name}
export OLLAMA_HOST=http://localhost:11434

# Ollamaでモデルを作成:
cd {self.output_dir}
ollama create {model_name} -f Modelfile

# テスト実行:
ollama run {model_name} "設計速度100km/hの最小曲線半径は？"
"""
        
        logger.info(config_update)
        
        # 設定ファイルを生成
        config_file = self.output_dir / "rag_config_update.txt"
        config_file.write_text(config_update)
        
    def run(self) -> bool:
        """量子化プロセス全体を実行"""
        try:
            # 1. 互換性チェック
            compatible, recommendation = self.analyze_model_compatibility()
            if not compatible:
                logger.error("Model not compatible for quantization")
                return False
                
            if recommendation != self.quantization_level:
                logger.warning(f"Recommended: {recommendation}, but using: {self.quantization_level}")
                
            # 2. LoRAとベースモデルをマージ
            merged_path = self.merge_lora_with_base()
            
            # 3. 量子化
            quantized_path = self.quantize_merged_model(merged_path)
            
            # 4. GGUF変換準備
            gguf_path = self.convert_to_gguf(quantized_path)
            
            # 5. Ollama設定作成
            model_name = f"rag-finetuned-{self.lora_path.name}-{self.quantization_level.lower()}"
            self.create_ollama_modelfile(model_name)
            
            # 6. RAG設定更新方法を表示
            self.update_rag_config(model_name)
            
            logger.success("=== 量子化完了 ===")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Next steps:")
            logger.info(f"1. Run the GGUF conversion script: {self.output_dir}/convert_to_gguf.sh")
            logger.info(f"2. Create Ollama model: ollama create {model_name} -f {self.output_dir}/Modelfile")
            logger.info(f"3. Update RAG config as shown above")
            
            return True
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize finetuned model (LoRA) for Ollama"
    )
    parser.add_argument(
        "--lora-path",
        default="./outputs/lora_20250831_122140",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/quantized_finetuned",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--quantization",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        default="Q4_K_M",
        help="Quantization level"
    )
    
    args = parser.parse_args()
    
    # LoRAパスの存在確認
    if not Path(args.lora_path).exists():
        logger.error(f"LoRA path not found: {args.lora_path}")
        
        # 利用可能なLoRAを表示
        outputs_dir = Path("./outputs")
        lora_dirs = [d for d in outputs_dir.glob("lora_*") if d.is_dir()]
        if lora_dirs:
            logger.info("Available LoRA adapters:")
            for d in lora_dirs:
                config_file = d / "adapter_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                        base = config.get("base_model_name_or_path", "Unknown")
                        logger.info(f"  - {d.name}: {base}")
        sys.exit(1)
        
    quantizer = FinetunedModelQuantizer(
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        quantization_level=args.quantization
    )
    
    success = quantizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()