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
