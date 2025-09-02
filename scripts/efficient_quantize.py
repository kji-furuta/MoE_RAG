#!/usr/bin/env python3
"""
効率的な量子化スクリプト - メモリ最適化版
マルチGPU対応で段階的に処理
"""

import os
import sys
import json
import gc
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# メモリ最適化設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class EfficientQuantizer:
    """効率的な量子化処理"""
    
    def __init__(self, 
                 lora_path: str,
                 output_dir: str = "./outputs/efficient_quantized",
                 quantization_level: str = "Q4_K_M"):
        self.lora_path = Path(lora_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quantization_level = quantization_level
        
        # LoRA設定を読み込み
        with open(self.lora_path / "adapter_config.json", 'r') as f:
            self.lora_config = json.load(f)
        self.base_model_name = self.lora_config.get("base_model_name_or_path")
        
    def clear_gpu_memory(self):
        """GPUメモリをクリア"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_gpu_info(self):
        """GPU情報を取得"""
        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            info.append({
                'id': i,
                'name': props.name,
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'free': total - reserved
            })
        return info
        
    def merge_lora_efficient(self) -> Path:
        """効率的なLoRAマージ（段階的処理）"""
        logger.info("=== Efficient LoRA Merge (Multi-GPU) ===")
        
        # GPUメモリをクリア
        self.clear_gpu_memory()
        
        # GPU情報を表示
        gpu_info = self.get_gpu_info()
        for gpu in gpu_info:
            logger.info(f"GPU {gpu['id']}: {gpu['name']} - Free: {gpu['free']:.2f} GB")
        
        merged_path = self.output_dir / "merged_model"
        merged_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: ベースモデルをCPUにロード（量子化なし）
            logger.info("Loading base model to CPU (no quantization)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",  # まずCPUに
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Step 2: LoRAアダプタをロード
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(
                base_model,
                str(self.lora_path),
                torch_dtype=torch.float16,
                device_map="cpu"  # CPUで処理
            )
            
            # Step 3: CPUでマージ
            logger.info("Merging on CPU...")
            merged_model = model.merge_and_unload()
            
            # Step 4: 保存（シャード化）
            logger.info("Saving merged model...")
            merged_model.save_pretrained(
                merged_path,
                max_shard_size="1GB",  # 小さいシャードサイズ
                safe_serialization=True
            )
            
            # トークナイザーも保存
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(merged_path)
            
            # メモリクリア
            del merged_model, model, base_model
            self.clear_gpu_memory()
            
            logger.success(f"Merged model saved to: {merged_path}")
            return merged_path
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            raise
            
    def quantize_with_multi_gpu(self, model_path: Path) -> Path:
        """マルチGPUで量子化"""
        logger.info("=== Multi-GPU Quantization ===")
        
        # GPUメモリをクリア
        self.clear_gpu_memory()
        
        # 量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # マルチGPU用のメモリマップ
        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = "22GiB"  # 各GPUに22GB（余裕を持たせる）
        max_memory["cpu"] = "200GiB"  # CPUメモリも活用
        
        logger.info(f"Memory allocation: {max_memory}")
        
        try:
            # 量子化してロード
            logger.info("Loading and quantizing model...")
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="balanced",  # バランス配置
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory=max_memory,
                offload_folder="/tmp/offload",
                low_cpu_mem_usage=True
            )
            
            # GPU使用状況を確認
            gpu_info = self.get_gpu_info()
            for gpu in gpu_info:
                logger.info(f"GPU {gpu['id']} - Used: {gpu['allocated']:.2f} GB / {gpu['total']:.2f} GB")
            
            # 量子化モデルを保存
            quantized_path = self.output_dir / "quantized_4bit"
            quantized_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving quantized model to {quantized_path}...")
            quantized_model.save_pretrained(
                quantized_path,
                max_shard_size="1GB"
            )
            
            # トークナイザーも保存
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(quantized_path)
            
            logger.success(f"Quantized model saved to: {quantized_path}")
            return quantized_path
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
            
    def create_ollama_instructions(self):
        """Ollama用の手順を生成"""
        instructions = f"""
# Ollama統合手順

## 1. 量子化済みモデルの場所
{self.output_dir}/quantized_4bit

## 2. GGUF変換（手動実行が必要）
```bash
# llama.cppのインストール
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

# 変換実行
python convert-hf-to-gguf.py {self.output_dir}/quantized_4bit \\
    --outfile {self.output_dir}/model.gguf \\
    --outtype q4_K_M
```

## 3. Ollamaモデル作成
```bash
cat > {self.output_dir}/Modelfile << EOF
FROM {self.output_dir}/model.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|endoftext|>"

SYSTEM "あなたは道路設計の専門家です。"
EOF

ollama create rag-deepseek-q4 -f {self.output_dir}/Modelfile
```

## 4. テスト
```bash
ollama run rag-deepseek-q4 "設計速度100km/hの最小曲線半径は？"
```

## 5. RAG設定更新
src/rag/config/rag_config.yaml:
```yaml
llm:
  use_ollama_fallback: true
  ollama_model: rag-deepseek-q4
```
"""
        
        instructions_path = self.output_dir / "ollama_instructions.md"
        instructions_path.write_text(instructions)
        logger.info(f"Instructions saved to: {instructions_path}")
        
    def run(self) -> bool:
        """量子化プロセス全体を実行"""
        try:
            # 1. LoRAマージ（CPU処理）
            merged_path = self.merge_lora_efficient()
            
            # 2. 量子化（マルチGPU）
            quantized_path = self.quantize_with_multi_gpu(merged_path)
            
            # 3. Ollama手順を生成
            self.create_ollama_instructions()
            
            logger.success("=== Process Complete ===")
            logger.info(f"Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Process failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Efficient LoRA quantization")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", default="./outputs/efficient_quantized", help="Output directory")
    
    args = parser.parse_args()
    
    quantizer = EfficientQuantizer(
        lora_path=args.lora_path,
        output_dir=args.output_dir
    )
    
    success = quantizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()