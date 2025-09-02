#!/usr/bin/env python3
"""
ファインチューニング済みLoRAモデルをGGUF形式に変換してOllamaで使用
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class GGUFConverter:
    """GGUF形式への変換"""
    
    def __init__(self, lora_path: str, output_dir: str = "./outputs/gguf_models"):
        self.lora_path = Path(lora_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LoRA設定を読み込み
        with open(self.lora_path / "adapter_config.json", 'r') as f:
            self.lora_config = json.load(f)
        self.base_model_name = self.lora_config.get("base_model_name_or_path")
        
    def install_llama_cpp(self):
        """llama.cppのインストール"""
        logger.info("Installing llama.cpp...")
        
        llama_cpp_dir = self.output_dir / "llama.cpp"
        
        if not llama_cpp_dir.exists():
            # llama.cppをクローン
            cmd = f"git clone https://github.com/ggerganov/llama.cpp {llama_cpp_dir}"
            subprocess.run(cmd, shell=True, check=True)
            
            # CMakeでビルド
            build_commands = f"""
            cd {llama_cpp_dir} && \
            cmake -B build && \
            cmake --build build --config Release -j 8
            """
            subprocess.run(build_commands, shell=True, check=True)
            
            logger.info("llama.cpp installed successfully")
        else:
            logger.info("llama.cpp already installed")
            
        return llama_cpp_dir
        
    def merge_and_convert(self):
        """LoRAマージとGGUF変換の手順を生成"""
        
        # Step 1: マージスクリプトを生成
        merge_script = f"""#!/usr/bin/env python3
# LoRAとベースモデルをマージしてGGUF用に準備

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# CPUでメモリ効率的に処理
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.base_model_name}",
    torch_dtype=torch.float16,
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

print("Merging LoRA with base model...")
merged = model.merge_and_unload()

print("Saving merged model...")
output_path = "{self.output_dir}/merged_for_gguf"
merged.save_pretrained(
    output_path,
    safe_serialization=True,
    max_shard_size="1GB"
)

tokenizer = AutoTokenizer.from_pretrained("{self.base_model_name}")
tokenizer.save_pretrained(output_path)

print("Merge complete! Now convert to GGUF format.")
"""
        
        script_path = self.output_dir / "merge_for_gguf.py"
        script_path.write_text(merge_script)
        
        logger.info(f"Merge script created: {script_path}")
        
        # Step 2: GGUF変換コマンドを生成
        convert_commands = f"""
# Step 1: LoRAとベースモデルをマージ（CPU使用、時間がかかります）
python {script_path}

# Step 2: GGUF形式に変換
cd {self.output_dir}/llama.cpp
python convert-hf-to-gguf.py {self.output_dir}/merged_for_gguf \
    --outfile {self.output_dir}/model-f16.gguf \
    --outtype f16

# Step 3: 量子化（Q4_K_M推奨）
./build/bin/quantize {self.output_dir}/model-f16.gguf \
    {self.output_dir}/model-q4_k_m.gguf Q4_K_M

# Step 4: Ollamaに登録
cat > {self.output_dir}/Modelfile << EOF
FROM {self.output_dir}/model-q4_k_m.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|endoftext|>"

SYSTEM "あなたは道路設計の専門家です。ファインチューニングされた知識を活用して正確な回答を提供します。"
EOF

ollama create deepseek-finetuned -f {self.output_dir}/Modelfile
"""
        
        commands_path = self.output_dir / "convert_commands.sh"
        commands_path.write_text(convert_commands)
        commands_path.chmod(0o755)
        
        logger.info(f"Conversion commands saved to: {commands_path}")
        
        return commands_path
        
    def create_quick_solution(self):
        """即座に使える代替案"""
        
        solution = f"""
# 即座に使える解決策

## 方法1: 事前量子化されたベースモデルを使用
既にGGUF形式のベースモデルがある場合、それを使用してファインチューニングの知識をプロンプトで補完

```bash
# Modelfileを作成（ファインチューニングの知識を含む）
cat > FinetunePrompt.Modelfile << 'EOF'
FROM ./cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese.gguf

SYSTEM "あなたは道路設計の専門家です。
ファインチューニングで学習した以下の知識を持っています：
- 設計速度100km/h → 最小曲線半径460m
- 設計速度80km/h → 最小曲線半径280m
- 設計速度60km/h → 最小曲線半径150m
- 設計速度50km/h → 最小曲線半径100m
- 設計速度40km/h → 最小曲線半径60m
これらの正確な数値に基づいて回答してください。"

PARAMETER temperature 0.1
EOF

# Ollamaに登録
ollama create deepseek-finetuned-prompt -f FinetunePrompt.Modelfile
```

## 方法2: 小規模モデルでテスト
メモリ制約が少ない7Bモデルで先にテスト

```bash
# 7Bモデルなら現在のGPUでマージ可能
ollama pull qwen2.5:7b-instruct-q4_k_m
```

## 方法3: Docker内で段階的に処理
```bash
# Dockerコンテナ内で実行
docker exec -it ai-ft-container bash

# CPU専用でマージ（メモリ節約）
cd /workspace/scripts
python merge_for_gguf.py
```

## 注意事項
- **マージには大量のRAM必要**（32Bモデル：~100GB）
- **GGUF変換には時間がかかる**（数時間）
- **量子化で精度は若干低下**するが実用的

## RAG設定更新
```yaml
# src/rag/config/rag_config.yaml
llm:
  use_ollama_fallback: true
  ollama_model: deepseek-finetuned  # 作成したモデル名
  ollama_host: http://localhost:11434
```
"""
        
        solution_path = self.output_dir / "quick_solution.md"
        solution_path.write_text(solution)
        
        logger.info(f"Quick solution saved to: {solution_path}")
        
    def run(self):
        """変換プロセスを実行"""
        logger.info("=== GGUF Conversion Setup ===")
        logger.info(f"LoRA adapter: {self.lora_path}")
        logger.info(f"Base model: {self.base_model_name}")
        
        # llama.cppをインストール
        llama_cpp_dir = self.install_llama_cpp()
        
        # 変換手順を生成
        commands_path = self.merge_and_convert()
        
        # 即座に使える代替案を生成
        self.create_quick_solution()
        
        logger.success("=== Setup Complete ===")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("\n次のステップ:")
        logger.info(f"1. 変換を実行: bash {commands_path}")
        logger.info(f"2. または代替案を確認: cat {self.output_dir}/quick_solution.md")
        logger.info("\n⚠️ 注意:")
        logger.info("- 32Bモデルのマージには100GB+のRAMが必要")
        logger.info("- 処理には数時間かかる可能性があります")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA to GGUF for Ollama"
    )
    parser.add_argument(
        "--lora-path",
        default="/workspace/outputs/lora_20250830_223432",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/gguf_models",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    converter = GGUFConverter(
        lora_path=args.lora_path,
        output_dir=args.output_dir
    )
    
    success = converter.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()