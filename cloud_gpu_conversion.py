#!/usr/bin/env python3
"""
クラウドGPU（Google Colab/Kaggle/AWS等）でLoRAをGGUF形式に変換
DeepSeek-R1-Distill-Qwen-32B-Japanese用
"""

# ============================================
# このスクリプトをGoogle Colab等で実行
# 必要メモリ: 80GB+ (A100推奨)
# ============================================

import os
import sys
import gc
import torch
from pathlib import Path
import time

# Step 1: 環境セットアップ
print("=" * 60)
print("Step 1: 環境セットアップ")
print("=" * 60)

# GPU情報を表示
!nvidia-smi

# 必要なパッケージをインストール
print("Installing required packages...")
!pip install -q transformers==4.44.0 peft accelerate bitsandbytes
!pip install -q sentencepiece protobuf

# llama.cppをインストール
print("Installing llama.cpp...")
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make clean && make LLAMA_CUDA=1 -j8

# Step 2: LoRAアダプタとベースモデルをダウンロード
print("\n" + "=" * 60)
print("Step 2: モデルのダウンロード")
print("=" * 60)

from huggingface_hub import snapshot_download
from google.colab import files
import zipfile

# 方法A: Google Driveからアップロード
print("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# LoRAアダプタのパス（Google Driveに事前にアップロード）
lora_path = "/content/drive/MyDrive/lora_20250830_223432"

# tar.gzファイルがある場合は解凍
if os.path.exists("/content/drive/MyDrive/lora_20250830_223432.tar.gz"):
    print("Extracting LoRA adapter from tar.gz...")
    !tar -xzf /content/drive/MyDrive/lora_20250830_223432.tar.gz -C /content/
    lora_path = "/content/lora_20250830_223432"

# または方法B: 直接アップロード
# print("LoRAアダプタ（lora_20250830_223432.tar.gz）をアップロードしてください")
# uploaded = files.upload()
# import tarfile
# with tarfile.open('lora_20250830_223432.tar.gz', 'r:gz') as tar:
#     tar.extractall('/content/')
# lora_path = "/content/lora_20250830_223432"

# ベースモデル名
base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"

# Step 3: メモリ効率的なマージ
print("\n" + "=" * 60)
print("Step 3: LoRAマージ（メモリ最適化）")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# GPUメモリ状況を確認
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# メモリをクリア
torch.cuda.empty_cache()
gc.collect()

# ベースモデルをロード（8bit量子化で省メモリ）
print("ベースモデルをロード中...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,  # 8bit量子化でロード
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# LoRAアダプタをロード
print("LoRAアダプタをロード中...")
model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    torch_dtype=torch.float16
)

# マージ実行
print("マージ中...（10-20分かかります）")
model = model.merge_and_unload()

# FP16で保存（GGUF変換用）
print("マージ済みモデルを保存中...")
merged_path = "/content/merged_model"
model.save_pretrained(
    merged_path,
    torch_dtype=torch.float16,
    safe_serialization=True,
    max_shard_size="2GB"
)

# トークナイザも保存
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(merged_path)

# メモリクリア
del model, base_model
torch.cuda.empty_cache()
gc.collect()

print(f"✅ マージ完了: {merged_path}")

# Step 4: GGUF形式に変換
print("\n" + "=" * 60)
print("Step 4: GGUF変換")
print("=" * 60)

# FP16でGGUF変換
!cd llama.cpp && python convert-hf-to-gguf.py \
    /content/merged_model \
    --outfile /content/model-f16.gguf \
    --outtype f16

print("✅ GGUF変換完了: model-f16.gguf")

# Step 5: 量子化（Q4_K_M）
print("\n" + "=" * 60)
print("Step 5: 量子化（Q4_K_M）")
print("=" * 60)

# 量子化実行
!cd llama.cpp && ./quantize \
    /content/model-f16.gguf \
    /content/deepseek-finetuned-q4_k_m.gguf \
    Q4_K_M

# ファイルサイズ確認
import os
size_gb = os.path.getsize("/content/deepseek-finetuned-q4_k_m.gguf") / (1024**3)
print(f"✅ 量子化完了: deepseek-finetuned-q4_k_m.gguf ({size_gb:.2f} GB)")

# Step 6: Modelfile作成
print("\n" + "=" * 60)
print("Step 6: Ollama用Modelfile作成")
print("=" * 60)

modelfile_content = """FROM ./deepseek-finetuned-q4_k_m.gguf

# パラメータ設定
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"

# システムプロンプト（ファインチューニング済み）
SYSTEM "あなたは日本の道路設計の専門家です。道路構造令と設計基準に基づいて正確な技術的回答を提供してください。"

# DeepSeek用テンプレート
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""
"""

with open("/content/Modelfile", "w") as f:
    f.write(modelfile_content)

print("✅ Modelfile作成完了")

# Step 7: ダウンロード準備
print("\n" + "=" * 60)
print("Step 7: ダウンロード")
print("=" * 60)

# 方法A: Google Driveに保存
import shutil
output_dir = "/content/drive/MyDrive/deepseek_gguf"
os.makedirs(output_dir, exist_ok=True)

shutil.copy("/content/deepseek-finetuned-q4_k_m.gguf", output_dir)
shutil.copy("/content/Modelfile", output_dir)

print(f"✅ Google Driveに保存: {output_dir}")

# 方法B: 直接ダウンロード
from google.colab import files
print("\n以下のファイルをダウンロード:")
files.download("/content/deepseek-finetuned-q4_k_m.gguf")
files.download("/content/Modelfile")

# Step 8: ローカルでの使用方法
print("\n" + "=" * 60)
print("Step 8: ローカル（WSL2）での使用方法")
print("=" * 60)

instructions = """
### ローカルでの設定手順

1. ダウンロードしたファイルをWSL2にコピー:
   ```bash
   # WSL2で実行
   cp /mnt/c/Users/[ユーザー名]/Downloads/deepseek-finetuned-q4_k_m.gguf ~/
   cp /mnt/c/Users/[ユーザー名]/Downloads/Modelfile ~/
   ```

2. Dockerコンテナにコピー:
   ```bash
   docker cp ~/deepseek-finetuned-q4_k_m.gguf ai-ft-container:/workspace/
   docker cp ~/Modelfile ai-ft-container:/workspace/
   ```

3. Ollamaに登録:
   ```bash
   docker exec ai-ft-container ollama create deepseek-finetuned -f /workspace/Modelfile
   ```

4. テスト:
   ```bash
   docker exec ai-ft-container ollama run deepseek-finetuned "設計速度100km/hの最小曲線半径は？"
   ```

5. RAG設定更新:
   ```yaml
   # src/rag/config/rag_config.yaml
   llm:
     use_ollama_fallback: true
     ollama_model: deepseek-finetuned
     ollama_host: http://localhost:11434
   ```

6. RAGシステムでテスト:
   ```bash
   curl -X POST "http://localhost:8050/rag/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "設計速度100km/hの最小曲線半径は？", "top_k": 3}'
   ```

期待される結果: 460m（ファインチューニング済みの正確な回答）
"""

print(instructions)

# 完了メッセージ
print("\n" + "=" * 60)
print("✅ 変換プロセス完了！")
print("=" * 60)
print(f"生成ファイル:")
print(f"  - deepseek-finetuned-q4_k_m.gguf ({size_gb:.2f} GB)")
print(f"  - Modelfile")
print("\nこれらのファイルをダウンロードして、上記の手順でローカルOllamaに登録してください。")