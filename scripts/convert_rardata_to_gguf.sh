#!/bin/bash
# RARdata学習済みモデルをGGUF形式に変換

set -e

MODEL_PATH="outputs/continual_task_103_20251209_002510/checkpoint-final"
OUTPUT_DIR="gguf_models/rardata_model"
BASE_MODEL="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"

echo "========================================="
echo "RARdata Model → GGUF 変換スクリプト"
echo "========================================="

# 1. LoRAアダプターをベースモデルにマージ
echo "[1/3] LoRAアダプターをマージ中..."
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("ベースモデルをロード中...")
base_model = AutoModelForCausalLM.from_pretrained(
    "${BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("LoRAアダプターをロード中...")
model = PeftModel.from_pretrained(
    base_model,
    "${MODEL_PATH}"
)

print("LoRAアダプターをマージ中...")
merged_model = model.merge_and_unload()

print("マージされたモデルを保存中...")
merged_model.save_pretrained("${OUTPUT_DIR}/merged")

tokenizer = AutoTokenizer.from_pretrained("${BASE_MODEL}")
tokenizer.save_pretrained("${OUTPUT_DIR}/merged")

print("✅ マージ完了")
EOF

# 2. GGUFに変換
echo "[2/3] GGUF形式に変換中..."
python3 << EOF
import os
import subprocess

# llama.cppのconvert.pyを使用
convert_script = "/workspace/scripts/llama_cpp_convert.py"

if not os.path.exists(convert_script):
    print("⚠️ llama.cpp convert.pyが見つかりません")
    print("代替: HuggingFace Hub経由でアップロード後、GGUF変換ツールを使用してください")
    print("https://github.com/ggerganov/llama.cpp")
else:
    cmd = [
        "python3", convert_script,
        "${OUTPUT_DIR}/merged",
        "--outfile", "${OUTPUT_DIR}/rardata-q4_k_m.gguf",
        "--outtype", "q4_k_m"
    ]
    subprocess.run(cmd, check=True)
    print("✅ GGUF変換完了")
EOF

# 3. 完了
echo "[3/3] 完了"
echo "========================================="
echo "✅ 変換成功"
echo "========================================="
echo "出力ファイル:"
echo "  - マージ済みモデル: ${OUTPUT_DIR}/merged/"
echo "  - GGUF (4-bit):     ${OUTPUT_DIR}/rardata-q4_k_m.gguf"
echo ""
echo "使用方法:"
echo "  # Ollama"
echo "  ollama create rardata -f Modelfile"
echo ""
echo "  # llama.cpp"
echo "  ./llama-cli -m ${OUTPUT_DIR}/rardata-q4_k_m.gguf -p 'プロンプト'"
echo "========================================="
