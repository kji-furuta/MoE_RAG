#!/usr/bin/env python3
"""
CALM3-22Bモデルを量子化してOllamaで使用可能にするスクリプト

実装手順:
1. HuggingFaceモデルをGGUF形式に変換
2. 量子化レベルを選択（Q4_K_M推奨）
3. Ollama用のModelfileを作成
4. Ollamaにモデルを登録
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class ModelQuantizer:
    """モデル量子化とOllama統合クラス"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "./outputs/quantized",
                 quantization_level: str = "Q4_K_M"):
        """
        Args:
            model_path: 元のモデルパス（HuggingFace形式）
            output_dir: 量子化モデルの出力先
            quantization_level: 量子化レベル (Q4_K_M, Q5_K_M, Q8_0)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quantization_level = quantization_level
        
        # 量子化設定
        self.quant_configs = {
            "Q4_K_M": {
                "bits": 4,
                "method": "awq",  # または "gptq"
                "memory_usage": "~12GB",
                "quality": "Good balance"
            },
            "Q5_K_M": {
                "bits": 5,
                "method": "bnb",  # bitsandbytes
                "memory_usage": "~15GB",
                "quality": "Better quality"
            },
            "Q8_0": {
                "bits": 8,
                "method": "bnb",
                "memory_usage": "~22GB",
                "quality": "Near original"
            }
        }
        
    def check_dependencies(self) -> bool:
        """必要な依存関係をチェック"""
        required_packages = ["transformers", "torch", "bitsandbytes"]
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
                
        if missing:
            logger.error(f"Missing packages: {missing}")
            return False
            
        # Ollamaがインストールされているか確認
        try:
            result = subprocess.run(["ollama", "version"], 
                                  capture_output=True, text=True)
            logger.info(f"Ollama version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.warning("Ollama not found. Please install Ollama first.")
            logger.info("Install: curl -fsSL https://ollama.com/install.sh | sh")
            return False
            
        return True
        
    def quantize_with_bitsandbytes(self) -> Path:
        """BitsandBytesを使用した量子化"""
        logger.info(f"Quantizing model with bitsandbytes: {self.quantization_level}")
        
        config = self.quant_configs[self.quantization_level]
        bits = config["bits"]
        
        # 量子化設定
        from transformers import BitsAndBytesConfig
        
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported bit configuration: {bits}")
            
        # モデルロード（量子化付き）
        logger.info(f"Loading model from {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 量子化モデルを保存
        output_path = self.output_dir / f"calm3-22b-{self.quantization_level.lower()}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving quantized model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        return output_path
        
    def create_ollama_modelfile(self, model_path: Path) -> Path:
        """Ollama用のModelfileを作成"""
        modelfile_content = f"""# CALM3-22B Quantized Model for Ollama
FROM {model_path}

# Model configuration
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"

# System prompt for Japanese road design
SYSTEM "あなたは道路設計の専門家です。技術基準に基づいて正確な回答を提供してください。"

# Template for chat
TEMPLATE """{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}User: {{{{ .Prompt }}}}
Assistant: """
"""
        
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        logger.info(f"Created Modelfile at {modelfile_path}")
        
        return modelfile_path
        
    def register_with_ollama(self, modelfile_path: Path, model_name: str = "calm3-22b-quantized"):
        """Ollamaにモデルを登録"""
        logger.info(f"Registering model with Ollama as '{model_name}'")
        
        try:
            # Ollamaでモデルを作成
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Model registered: {result.stdout}")
            
            # テスト実行
            logger.info("Testing model...")
            test_cmd = ["ollama", "run", model_name, "設計速度100km/hの最小曲線半径は？"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True)
            logger.info(f"Test response: {test_result.stdout[:200]}...")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to register model: {e.stderr}")
            return False
            
    def create_gguf_conversion_script(self) -> Path:
        """GGUF変換用スクリプトを作成（代替方法）"""
        script_content = """#!/bin/bash
# GGUF形式への変換スクリプト

MODEL_PATH=$1
OUTPUT_PATH=$2
QUANT_LEVEL=${3:-Q4_K_M}

echo "Converting model to GGUF format..."
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo "Quantization: $QUANT_LEVEL"

# llama.cppのインストール確認
if ! command -v convert.py &> /dev/null; then
    echo "Installing llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    cd ..
fi

# 変換実行
python llama.cpp/convert.py \\
    --outfile "$OUTPUT_PATH/model.gguf" \\
    --outtype "$QUANT_LEVEL" \\
    "$MODEL_PATH"

echo "Conversion complete!"
"""
        
        script_path = self.output_dir / "convert_to_gguf.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        return script_path
        
    def run(self) -> bool:
        """量子化とOllama登録を実行"""
        if not self.check_dependencies():
            return False
            
        try:
            # 1. モデルを量子化
            quantized_path = self.quantize_with_bitsandbytes()
            
            # 2. Modelfileを作成
            modelfile_path = self.create_ollama_modelfile(quantized_path)
            
            # 3. Ollamaに登録
            success = self.register_with_ollama(
                modelfile_path, 
                f"calm3-22b-{self.quantization_level.lower()}"
            )
            
            if success:
                logger.success("Model successfully quantized and registered with Ollama!")
                logger.info(f"Usage: ollama run calm3-22b-{self.quantization_level.lower()}")
                
                # RAG設定更新用の情報を出力
                print("\n=== RAG設定更新 ===")
                print("src/rag/config/rag_config.yamlを以下のように更新してください:")
                print(f"""
llm:
  use_ollama_fallback: true
  ollama_model: calm3-22b-{self.quantization_level.lower()}
  # または環境変数で設定
  # export OLLAMA_MODEL=calm3-22b-{self.quantization_level.lower()}
""")
            
            return success
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Quantize CALM3-22B model for Ollama")
    parser.add_argument(
        "--model-path",
        default="cyberagent/calm3-22b-chat",
        help="Path to the model (HuggingFace ID or local path)"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/quantized",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--quantization",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        default="Q4_K_M",
        help="Quantization level"
    )
    parser.add_argument(
        "--lora-path",
        help="Path to LoRA adapter to merge before quantization"
    )
    
    args = parser.parse_args()
    
    # LoRAアダプタがある場合の処理
    if args.lora_path:
        logger.info(f"Merging LoRA adapter from {args.lora_path}")
        # TODO: LoRAマージ処理を実装
        
    quantizer = ModelQuantizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization_level=args.quantization
    )
    
    success = quantizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()