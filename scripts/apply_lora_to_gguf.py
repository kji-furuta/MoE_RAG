#!/usr/bin/env python3
"""
LoRAアダプターをGGUFベースモデルに適用
"""

import os
import sys
import subprocess
from pathlib import Path

def step1_convert_lora_to_gguf():
    """Step 1: LoRAアダプターをGGUF形式に変換"""
    print("="*60)
    print("Step 1: LoRAアダプター → GGUF変換")
    print("="*60)
    
    # LoRAアダプターのパス（解凍済み）
    lora_path = "/workspace/outputs/lora_20250830_223432_extracted/lora_20250830_223432"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRAアダプターが見つかりません: {lora_path}")
        return None
    
    print(f"✅ LoRAアダプター検出: {lora_path}")
    
    # llama.cppのconvert_lora_to_gguf.pyを使用
    convert_script = "/workspace/llama.cpp/convert_lora_to_gguf.py"
    
    if not os.path.exists(convert_script):
        print(f"❌ 変換スクリプトが見つかりません: {convert_script}")
        print("llama.cppをセットアップしてください")
        return None
    
    # 出力ファイル
    output_file = "/workspace/models/lora_adapter.gguf"
    
    # ベースモデルの設定ファイルを確認
    base_model_config = "/workspace/models/deepseek-base"
    
    if os.path.exists(f"{base_model_config}/config.json"):
        # 設定ファイルがある場合は使用
        cmd = f"python {convert_script} {lora_path} --outfile {output_file} --base {base_model_config}"
        print(f"ベースモデル設定: {base_model_config}")
    else:
        # ない場合はLoRAのadapter_config.jsonから取得
        print("⚠️ ベースモデル設定が見つかりません")
        print("設定ファイルをダウンロードしてください:")
        print("bash /workspace/scripts/prepare_base_model_config.sh")
        cmd = f"python {convert_script} {lora_path} --outfile {output_file}"
    
    print(f"\n実行コマンド: {cmd}")
    print("変換中...")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ GGUF変換成功: {output_file}")
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024*1024)
            print(f"   ファイルサイズ: {size_mb:.2f} MB")
        return output_file
    else:
        print(f"❌ 変換失敗")
        print(f"エラー: {result.stderr}")
        return None

def step2_merge_lora_with_base():
    """Step 2: LoRA GGUFをベースモデルとマージ"""
    print("\n" + "="*60)
    print("Step 2: LoRAアダプターとベースモデルのマージ")
    print("="*60)
    
    base_model = "/workspace/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
    lora_gguf = "/workspace/models/lora_adapter.gguf"
    output_model = "/workspace/models/deepseek-32b-finetuned.gguf"
    
    # ファイル存在確認
    if not os.path.exists(base_model):
        print(f"❌ ベースモデルが見つかりません: {base_model}")
        print("先にベースモデルをダウンロードしてください：")
        print("docker exec -it ai-ft-container wget -P /workspace/models/ \\")
        print("  https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf")
        return None
    
    if not os.path.exists(lora_gguf):
        print(f"❌ LoRA GGUFが見つかりません: {lora_gguf}")
        print("Step 1を先に実行してください")
        return None
    
    print(f"ベースモデル: {base_model}")
    print(f"LoRAアダプター: {lora_gguf}")
    print(f"出力先: {output_model}")
    
    # llama.cppのllama-export-loraを使用してマージ
    export_lora = "/workspace/llama.cpp/build/bin/llama-export-lora"
    
    if not os.path.exists(export_lora):
        print("⚠️ llama-export-loraが見つかりません")
        print("代替方法: 実行時にLoRAを適用します（マージなし）")
        return base_model, lora_gguf
    
    # マージコマンド
    cmd = f"{export_lora} -m {base_model} -o {output_model} --lora {lora_gguf}"
    
    print(f"\n実行コマンド: {cmd}")
    print("マージ中...")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ マージ成功: {output_model}")
        if os.path.exists(output_model):
            size_gb = os.path.getsize(output_model) / (1024*1024*1024)
            print(f"   ファイルサイズ: {size_gb:.2f} GB")
        return output_model, None
    else:
        print(f"⚠️ マージ失敗（実行時適用モードを使用）")
        print(f"エラー: {result.stderr}")
        return base_model, lora_gguf

def step3_create_ollama_model(model_path, lora_path=None):
    """Step 3: Ollamaモデルを作成"""
    print("\n" + "="*60)
    print("Step 3: Ollamaモデル作成")
    print("="*60)
    
    # Modelfileを作成
    modelfile_path = "/workspace/models/Modelfile_finetuned"
    
    if lora_path:
        # LoRAを実行時に適用する場合
        modelfile_content = f"""FROM {model_path}
ADAPTER {lora_path}

# ファインチューニング済みDeepSeek-32B（LoRA適用）
PARAMETER temperature 0.6
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
        print("モード: LoRA実行時適用")
    else:
        # マージ済みモデルの場合
        modelfile_content = f"""FROM {model_path}

# ファインチューニング済みDeepSeek-32B（マージ済み）
PARAMETER temperature 0.6
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
        print("モード: マージ済みモデル")
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile作成: {modelfile_path}")
    
    # Ollamaに登録
    model_name = "deepseek-32b-finetuned"
    
    print(f"\nOllamaに '{model_name}' として登録中...")
    cmd = f"cd /workspace/models && ollama create {model_name} -f Modelfile_finetuned"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Ollama登録成功: {model_name}")
        print("\n使用方法:")
        print(f"  ollama run {model_name}")
        print("\nRAGシステムでの使用:")
        print(f"  設定でモデル名を '{model_name}' に変更")
        return True
    else:
        print(f"❌ Ollama登録失敗")
        print(f"エラー: {result.stderr}")
        return False

def main():
    """メイン処理"""
    print("LoRAファインチューニング済みアダプターの適用")
    print("="*60)
    
    # Step 1: LoRA → GGUF変換
    lora_gguf = step1_convert_lora_to_gguf()
    if not lora_gguf:
        print("\n❌ LoRA変換に失敗しました")
        return 1
    
    # Step 2: マージまたは実行時適用
    result = step2_merge_lora_with_base()
    if result is None:
        print("\n❌ ベースモデルのダウンロードが必要です")
        return 1
    
    model_path, lora_path = result
    if not model_path:
        print("\n❌ モデル準備に失敗しました")
        return 1
    
    # Step 3: Ollama登録
    if step3_create_ollama_model(model_path, lora_path):
        print("\n" + "="*60)
        print("✅ 完了！")
        print("="*60)
        print("\nファインチューニング済みモデルが使用可能になりました")
        print("RAGシステムで 'deepseek-32b-finetuned' を選択してください")
        return 0
    else:
        print("\n❌ Ollama登録に失敗しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())