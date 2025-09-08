#!/usr/bin/env python3
"""
llama.cppを使用した直接量子化スクリプト
メモリ効率的な量子化のための代替アプローチ
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class LlamaCppQuantizer:
    """llama.cppを使用した量子化"""
    
    def __init__(self, 
                 lora_path: str,
                 output_dir: str = "./outputs/quantized_llamacpp",
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
        with open(self.lora_path / "adapter_config.json", 'r') as f:
            self.lora_config = json.load(f)
        self.base_model_name = self.lora_config.get("base_model_name_or_path")
        
    def setup_llamacpp(self) -> bool:
        """llama.cppのセットアップ"""
        logger.info("Setting up llama.cpp...")
        
        llamacpp_dir = self.output_dir / "llama.cpp"
        
        if not llamacpp_dir.exists():
            try:
                # llama.cppをクローン
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp",
                    str(llamacpp_dir)
                ], check=True)
                
                # ビルド
                subprocess.run([
                    "make", "-C", str(llamacpp_dir), 
                    "LLAMA_CUDA=1",  # CUDA有効化
                    "-j8"
                ], check=True)
                
                logger.info("llama.cpp setup complete")
                return True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to setup llama.cpp: {e}")
                return False
        else:
            logger.info("llama.cpp already exists")
            return True
    
    def export_to_safetensors(self) -> Path:
        """HuggingFaceモデルをsafetensors形式でエクスポート"""
        logger.info("Exporting model to safetensors format...")
        
        export_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# メモリ効率的なロード
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.base_model_name}",
    torch_dtype=torch.float16,
    device_map="cpu",  # CPUで処理
    low_cpu_mem_usage=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "{self.lora_path}",
    torch_dtype=torch.float16
)

print("Merging LoRA...")
merged = model.merge_and_unload()

print("Saving to safetensors...")
output_path = "{self.output_dir}/merged_safetensors"
merged.save_pretrained(
    output_path,
    safe_serialization=True,  # safetensors形式で保存
    max_shard_size="2GB"
)

tokenizer = AutoTokenizer.from_pretrained("{self.base_model_name}")
tokenizer.save_pretrained(output_path)

print("Export complete!")
"""
        
        script_path = self.output_dir / "export_model.py"
        script_path.write_text(export_script)
        
        try:
            subprocess.run([
                sys.executable, str(script_path)
            ], check=True)
            
            return self.output_dir / "merged_safetensors"
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def convert_to_gguf(self, model_path: Path) -> Path:
        """safetensorsをGGUF形式に変換"""
        logger.info("Converting to GGUF format...")
        
        llamacpp_dir = self.output_dir / "llama.cpp"
        convert_script = llamacpp_dir / "convert.py"
        
        gguf_path = self.output_dir / "model.gguf"
        
        try:
            # 変換実行
            subprocess.run([
                sys.executable, str(convert_script),
                str(model_path),
                "--outfile", str(gguf_path),
                "--outtype", "f16",  # まずFP16で変換
            ], check=True)
            
            logger.info(f"GGUF conversion complete: {gguf_path}")
            return gguf_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert to GGUF: {e}")
            raise
    
    def quantize_gguf(self, gguf_path: Path) -> Path:
        """GGUFモデルを量子化"""
        logger.info(f"Quantizing to {self.quantization_level}...")
        
        llamacpp_dir = self.output_dir / "llama.cpp"
        quantize_exe = llamacpp_dir / "quantize"
        
        quantized_path = self.output_dir / f"model_{self.quantization_level.lower()}.gguf"
        
        try:
            # 量子化実行
            subprocess.run([
                str(quantize_exe),
                str(gguf_path),
                str(quantized_path),
                self.quantization_level
            ], check=True)
            
            logger.info(f"Quantization complete: {quantized_path}")
            
            # 元のFP16モデルは削除（容量節約）
            if gguf_path.exists():
                gguf_path.unlink()
            
            return quantized_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to quantize: {e}")
            raise
    
    def create_ollama_modelfile(self, gguf_path: Path) -> Path:
        """Ollama用のModelfileを作成"""
        modelfile_content = f"""# Quantized LoRA Model
FROM {gguf_path}

# Configuration
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"

# System prompt
SYSTEM "あなたは道路設計の専門家です。日本の道路構造令と設計基準に基づいて、正確で詳細な技術的回答を提供してください。"

# Model info
# Base: {self.base_model_name}
# LoRA: {self.lora_path.name}
# Quantization: {self.quantization_level}
"""
        
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        
        return modelfile_path
    
    def register_with_ollama(self, modelfile_path: Path, model_name: str) -> bool:
        """Ollamaにモデルを登録"""
        logger.info(f"Registering with Ollama as '{model_name}'...")
        
        try:
            # Ollamaでモデルを作成
            subprocess.run([
                "ollama", "create", 
                model_name, 
                "-f", str(modelfile_path)
            ], check=True)
            
            logger.info(f"Model registered: {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to register with Ollama: {e}")
            return False
    
    def run(self, model_name: str = None) -> bool:
        """量子化プロセス全体を実行"""
        try:
            # 1. llama.cppのセットアップ
            if not self.setup_llamacpp():
                return False
            
            # 2. safetensors形式でエクスポート
            logger.info("=== Step 1: Export to safetensors ===")
            safetensors_path = self.export_to_safetensors()
            
            # 3. GGUF形式に変換
            logger.info("=== Step 2: Convert to GGUF ===")
            gguf_path = self.convert_to_gguf(safetensors_path)
            
            # 4. 量子化
            logger.info("=== Step 3: Quantize ===")
            quantized_path = self.quantize_gguf(gguf_path)
            
            # 5. Ollama設定
            logger.info("=== Step 4: Create Ollama config ===")
            modelfile_path = self.create_ollama_modelfile(quantized_path)
            
            # 6. Ollamaに登録
            if model_name:
                self.register_with_ollama(modelfile_path, model_name)
            
            logger.success("=== Quantization Complete ===")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Quantized model: {quantized_path}")
            logger.info(f"Model size: {quantized_path.stat().st_size / (1024**3):.2f} GB")
            
            if model_name:
                logger.info(f"Test with: ollama run {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize LoRA model using llama.cpp"
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/quantized_llamacpp",
        help="Output directory"
    )
    parser.add_argument(
        "--quantization",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        default="Q4_K_M",
        help="Quantization level"
    )
    parser.add_argument(
        "--model-name",
        help="Name for Ollama model"
    )
    
    args = parser.parse_args()
    
    quantizer = LlamaCppQuantizer(
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        quantization_level=args.quantization
    )
    
    success = quantizer.run(args.model_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()