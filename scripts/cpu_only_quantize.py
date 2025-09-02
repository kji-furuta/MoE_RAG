#!/usr/bin/env python3
"""
CPU専用量子化スクリプト - GPUメモリ不足を回避
"""

import os
import sys
import json
import gc
import shutil
from pathlib import Path
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class CPUOnlyQuantizer:
    """CPU専用の量子化処理"""
    
    def __init__(self, 
                 lora_path: str,
                 output_dir: str = "./outputs/cpu_quantized"):
        self.lora_path = Path(lora_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LoRA設定を読み込み
        with open(self.lora_path / "adapter_config.json", 'r') as f:
            self.lora_config = json.load(f)
        self.base_model_name = self.lora_config.get("base_model_name_or_path")
        
    def prepare_for_ollama(self):
        """Ollama用の準備（実際の量子化はOllama側で実行）"""
        logger.info("=== Preparing for Ollama Quantization ===")
        
        # Step 1: LoRAアダプタ情報を準備
        adapter_info = {
            "base_model": self.base_model_name,
            "lora_path": str(self.lora_path),
            "adapter_config": self.lora_config,
            "quantization_target": "Q4_K_M"
        }
        
        info_path = self.output_dir / "adapter_info.json"
        with open(info_path, 'w') as f:
            json.dump(adapter_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Adapter info saved to: {info_path}")
        
        # Step 2: マージスクリプトを生成（小バッチ処理）
        merge_script = f"""#!/usr/bin/env python3
# CPU専用マージスクリプト
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import gc

# CPUのみ使用
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.base_model_name}",
    torch_dtype=torch.float32,  # CPU用にFP32
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "{self.lora_path}",
    device_map="cpu"
)

print("Merging (this will take time)...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(
    "{self.output_dir}/merged_cpu",
    max_shard_size="500MB"  # 小さいシャード
)

tokenizer = AutoTokenizer.from_pretrained("{self.base_model_name}")
tokenizer.save_pretrained("{self.output_dir}/merged_cpu")

print("Done!")
"""
        
        script_path = self.output_dir / "merge_cpu.py"
        script_path.write_text(merge_script)
        script_path.chmod(0o755)
        
        logger.info(f"Merge script created: {script_path}")
        
        # Step 3: 代替案を提示
        self.create_alternative_solutions()
        
        return True
        
    def create_alternative_solutions(self):
        """代替ソリューションを提案"""
        
        solutions = f"""
# 量子化の代替ソリューション

## 方法1: Ollamaで事前量子化モデルを使用
最も簡単な方法は、既に量子化されたモデルを使用することです：

```bash
# Qwen 32B量子化版を使用
ollama pull qwen2.5:32b-instruct-q4_K_M

# RAG設定を更新
# src/rag/config/rag_config.yaml
llm:
  use_ollama_fallback: true
  ollama_model: qwen2.5:32b-instruct-q4_K_M
```

## 方法2: より小さいモデルを使用
メモリ制約がある場合は、小さいモデルを使用：

```bash
# 7Bモデルを使用（高速・低メモリ）
ollama pull qwen2.5:7b-instruct-q4_K_M

# または14Bモデル（バランス型）
ollama pull qwen2.5:14b-instruct-q4_K_M
```

## 方法3: クラウドGPUを使用
大規模モデルの量子化にはクラウドGPUを使用：

1. Google Colab Pro+ (A100 40GB)
2. Paperspace Gradient (A100 80GB)
3. AWS EC2 p3/p4インスタンス

## 方法4: CPU専用マージ（時間がかかる）
```bash
# CPUのみでマージ（数時間かかる可能性）
python {self.output_dir}/merge_cpu.py

# 完了後、llama.cppで量子化
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

# GGUF変換
python convert-hf-to-gguf.py {self.output_dir}/merged_cpu \\
    --outfile {self.output_dir}/model.gguf \\
    --outtype q4_K_M
```

## 方法5: API経由でファインチューニング済みモデルを使用
量子化せずにAPIで使用：

```python
from transformers import pipeline

# CPUで推論（遅いが動作する）
pipe = pipeline(
    "text-generation",
    model="{self.lora_path}",
    device="cpu",
    torch_dtype=torch.float32
)
```

## 推奨事項
現在のハードウェア（GPU 48GB）では、以下を推奨します：

1. **即座に使用可能**: Ollama公式のQwen 32B Q4_K_Mモデルを使用
2. **品質重視**: 14Bモデルを使用（メモリ効率的）
3. **速度重視**: 7Bモデルを使用（最速）

これらのモデルは既に最適化されており、すぐに使用できます。
"""
        
        solutions_path = self.output_dir / "alternative_solutions.md"
        solutions_path.write_text(solutions)
        
        logger.info(f"Alternative solutions saved to: {solutions_path}")
        
        # Ollama設定ファイルも生成
        ollama_config = {
            "recommended_models": [
                {
                    "name": "qwen2.5:32b-instruct-q4_K_M",
                    "size": "18GB",
                    "quality": "High",
                    "speed": "Medium",
                    "command": "ollama pull qwen2.5:32b-instruct-q4_K_M"
                },
                {
                    "name": "qwen2.5:14b-instruct-q4_K_M",
                    "size": "8GB",
                    "quality": "Good",
                    "speed": "Fast",
                    "command": "ollama pull qwen2.5:14b-instruct-q4_K_M"
                },
                {
                    "name": "qwen2.5:7b-instruct-q4_K_M",
                    "size": "4GB",
                    "quality": "Moderate",
                    "speed": "Very Fast",
                    "command": "ollama pull qwen2.5:7b-instruct-q4_K_M"
                }
            ],
            "rag_config_update": {
                "llm": {
                    "use_ollama_fallback": True,
                    "ollama_model": "qwen2.5:32b-instruct-q4_K_M",
                    "ollama_host": "http://localhost:11434"
                }
            }
        }
        
        config_path = self.output_dir / "ollama_config.json"
        with open(config_path, 'w') as f:
            json.dump(ollama_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ollama config saved to: {config_path}")
        
    def run(self):
        """処理を実行"""
        logger.info("=== CPU-Only Quantization Preparation ===")
        logger.info(f"Base model: {self.base_model_name}")
        logger.info(f"LoRA adapter: {self.lora_path}")
        
        # Ollama用の準備
        self.prepare_for_ollama()
        
        logger.success("=== Preparation Complete ===")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Check alternative_solutions.md for immediate options")
        logger.info("2. Use Ollama pre-quantized models for quick setup")
        logger.info("3. Or run merge_cpu.py for CPU-based merging (slow)")
        
        # RAG設定更新スクリプトも生成
        self.create_rag_update_script()
        
        return True
        
    def create_rag_update_script(self):
        """RAG設定更新スクリプトを生成"""
        update_script = f"""#!/bin/bash
# RAG設定を更新するスクリプト

echo "Updating RAG configuration..."

# Ollama モデルをプル
ollama pull qwen2.5:32b-instruct-q4_K_M

# RAG設定を更新
cat > /tmp/rag_config_update.yaml << EOF
llm:
  use_finetuned: false
  use_ollama_fallback: true
  ollama_model: qwen2.5:32b-instruct-q4_K_M
  ollama_host: http://localhost:11434
  
embedding:
  model_name: intfloat/multilingual-e5-large
  dimension: 1024
  
vector_store:
  collection_name: road_design_docs
  similarity_threshold: 0.7
EOF

echo "Configuration template created at /tmp/rag_config_update.yaml"
echo "Please update src/rag/config/rag_config.yaml with these settings"

# テスト
echo "Testing Ollama connection..."
curl -s http://localhost:11434/api/tags | python3 -m json.tool

echo "Done!"
"""
        
        script_path = self.output_dir / "update_rag_config.sh"
        script_path.write_text(update_script)
        script_path.chmod(0o755)
        
        logger.info(f"RAG update script created: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only quantization preparation"
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/cpu_quantized",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    quantizer = CPUOnlyQuantizer(
        lora_path=args.lora_path,
        output_dir=args.output_dir
    )
    
    success = quantizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()