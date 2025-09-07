#!/bin/bash

# GPT-NeoX-20B用Ollama設定スクリプト
# ファインチューニング前のベースモデル準備とファインチューニング後の自動変換

echo "========================================="
echo "Setting up GPT-NeoX-20B for Ollama"
echo "========================================="

# ワークスペースディレクトリを環境に応じて設定
if [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    WORKSPACE_DIR="$(dirname "$(dirname "$(realpath "$0")")")"  # スクリプトの親ディレクトリ
fi

echo "Using workspace directory: $WORKSPACE_DIR"

# 1. Hugging FaceからGGUF版を探す（コミュニティ版）
echo "Checking for GPT-NeoX-20B GGUF versions..."

# TensorBlock版のGGUFモデルを使用
GGUF_MODEL_URL="https://huggingface.co/tensorblock/gpt-neox-20b-GGUF/resolve/main/gpt-neox-20b-Q4_K_M.gguf"
GGUF_MODEL_PATH="$WORKSPACE_DIR/models/gpt-neox-20b.Q4_K_M.gguf"

# GGUFファイルが存在しない場合はダウンロード
if [ ! -f "$GGUF_MODEL_PATH" ]; then
    echo "Downloading GPT-NeoX-20B GGUF (Q4_K_M quantization)..."
    mkdir -p $WORKSPACE_DIR/models
    
    # wgetでダウンロード（進捗表示付き）
    wget -c "$GGUF_MODEL_URL" -O "$GGUF_MODEL_PATH" --progress=bar:force
    
    if [ $? -eq 0 ]; then
        echo "✅ GPT-NeoX-20B GGUF downloaded successfully!"
    else
        echo "⚠️  Failed to download GGUF. Will need to convert from HF format after finetuning."
        rm -f "$GGUF_MODEL_PATH"  # 失敗時は削除
    fi
else
    echo "✅ GPT-NeoX-20B GGUF already exists"
fi

# 2. OllamaにGGUFモデルを登録
if [ -f "$GGUF_MODEL_PATH" ]; then
    echo "Creating Ollama Modelfile for GPT-NeoX-20B..."
    
    cat > $WORKSPACE_DIR/models/Modelfile_gpt_neox_20b << EOF
FROM $WORKSPACE_DIR/models/gpt-neox-20b.Q4_K_M.gguf

# GPT-NeoX-20B Base Model
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are GPT-NeoX, a 20B parameter language model trained by EleutherAI. 
You are designed to be helpful, harmless, and honest in your responses."""
EOF

    # Ollamaにモデルを登録
    if ! ollama list | grep -q "gpt-neox:20b"; then
        echo "Registering GPT-NeoX-20B in Ollama..."
        cd $WORKSPACE_DIR/models
        ollama create gpt-neox:20b -f Modelfile_gpt_neox_20b
        
        if [ $? -eq 0 ]; then
            echo "✅ GPT-NeoX-20B registered in Ollama successfully!"
        else
            echo "❌ Failed to register GPT-NeoX-20B in Ollama"
        fi
    else
        echo "✅ GPT-NeoX-20B already registered in Ollama"
    fi
fi

# 3. ファインチューニング後の自動変換用フック設定
echo "Setting up post-finetuning hooks..."

# 自動変換スクリプト作成
cat > $WORKSPACE_DIR/scripts/auto_convert_gpt_neox.py << 'EOF'
#!/usr/bin/env python3
"""
GPT-NeoX-20Bファインチューニング済みモデルの自動GGUF変換
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def find_latest_finetuned_model():
    """最新のファインチューニング済みモデルを検索"""
    outputs_dir = Path("/workspace/outputs")
    model_dirs = []
    
    # ファインチューニング済みモデルのディレクトリを検索
    for d in outputs_dir.glob("*"):
        if d.is_dir():
            info_file = d / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                    if "gpt-neox-20b" in info.get("base_model", "").lower():
                        model_dirs.append(d)
    
    # 最新のものを返す
    if model_dirs:
        return sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1]
    return None

def convert_to_gguf(model_path, output_path):
    """モデルをGGUF形式に変換"""
    print(f"Converting {model_path} to GGUF...")
    
    # 変換スクリプトを実行
    cmd = [
        "python", "/workspace/scripts/convert_to_gguf.py",
        "--model-path", str(model_path),
        "--output-path", str(output_path),
        "--quantization", "Q4_K_M"  # 4bit量子化
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GGUF conversion successful!")
        return True
    else:
        print(f"❌ GGUF conversion failed: {result.stderr}")
        return False

def register_in_ollama(gguf_path, model_name="gpt-neox:20b-finetuned"):
    """GGUFファイルをOllamaに登録"""
    print(f"Registering {model_name} in Ollama...")
    
    # Modelfile作成
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM "You are a finetuned GPT-NeoX-20B model, specialized for your specific task."
"""
    
    modelfile_path = Path("/workspace/models") / f"Modelfile_{model_name.replace(':', '_')}"
    modelfile_path.write_text(modelfile_content)
    
    # Ollamaに登録
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Model {model_name} registered in Ollama!")
        return True
    else:
        print(f"❌ Failed to register in Ollama: {result.stderr}")
        return False

if __name__ == "__main__":
    # 最新のファインチューニング済みモデルを検索
    model_path = find_latest_finetuned_model()
    
    if model_path:
        print(f"Found finetuned model: {model_path}")
        
        # GGUF変換
        gguf_path = Path("/workspace/models") / "gpt-neox-20b-finetuned.gguf"
        if convert_to_gguf(model_path, gguf_path):
            # Ollama登録
            register_in_ollama(gguf_path)
    else:
        print("No finetuned GPT-NeoX-20B model found.")
EOF

chmod +x $WORKSPACE_DIR/scripts/auto_convert_gpt_neox.py

echo "========================================="
echo "GPT-NeoX-20B setup complete!"
echo "========================================="
echo ""
echo "Usage:"
echo "1. Base model: ollama run gpt-neox:20b"
echo "2. After finetuning: python $WORKSPACE_DIR/scripts/auto_convert_gpt_neox.py"
echo ""