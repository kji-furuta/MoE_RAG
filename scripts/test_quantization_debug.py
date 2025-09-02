#!/usr/bin/env python3
"""
量子化処理のデバッグスクリプト
エラーの詳細を確認
"""

import os
import sys
import subprocess
from pathlib import Path

# ワークスペースに追加
sys.path.append('/workspace')

def run_command_with_output(cmd):
    """コマンド実行と詳細出力"""
    print(f"\n実行コマンド: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(f"リターンコード: {result.returncode}")
    
    if result.stdout:
        print("標準出力:")
        print(result.stdout)
    
    if result.stderr:
        print("エラー出力:")
        print(result.stderr)
    
    print("-" * 60)
    return result.returncode == 0

def check_environment():
    """環境チェック"""
    print("="*60)
    print("環境チェック")
    print("="*60)
    
    # llama.cppの確認
    llama_cpp_dir = "/workspace/llama.cpp"
    
    print(f"\n1. llama.cppディレクトリ: {llama_cpp_dir}")
    if os.path.exists(llama_cpp_dir):
        print("   ✅ 存在します")
        
        # 重要なファイルの確認
        files_to_check = [
            "build/bin/llama-quantize",
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf.py"
        ]
        
        for file in files_to_check:
            full_path = os.path.join(llama_cpp_dir, file)
            if os.path.exists(full_path):
                print(f"   ✅ {file} が存在")
            else:
                print(f"   ❌ {file} が見つかりません")
    else:
        print("   ❌ ディレクトリが存在しません")
        return False
    
    # Pythonパッケージの確認
    print("\n2. 必要なPythonパッケージ:")
    packages = ["gguf", "sentencepiece", "protobuf", "transformers", "peft"]
    
    for package in packages:
        result = subprocess.run(
            f"python -c 'import {package}; print({package}.__version__ if hasattr({package}, \"__version__\") else \"OK\")'",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ {package}: {version}")
        else:
            print(f"   ❌ {package}: インストールされていません")
    
    # outputsディレクトリの確認
    print("\n3. outputsディレクトリ:")
    outputs_dir = Path("/workspace/outputs")
    
    if outputs_dir.exists():
        print(f"   ✅ 存在します: {outputs_dir}")
        
        # LoRAモデルを探す
        lora_dirs = list(outputs_dir.glob("lora_*")) + \
                   list(outputs_dir.glob("*/final_lora_model")) + \
                   list(outputs_dir.glob("*/best_lora_model"))
        
        if lora_dirs:
            print(f"   ✅ LoRAモデル検出: {len(lora_dirs)}個")
            for lora_dir in lora_dirs[:3]:  # 最初の3個を表示
                print(f"      - {lora_dir}")
        else:
            print("   ⚠️ LoRAモデルが見つかりません")
    else:
        print(f"   ❌ ディレクトリが存在しません: {outputs_dir}")
    
    return True

def test_simple_conversion():
    """簡単なGGUF変換テスト"""
    print("\n" + "="*60)
    print("GGUF変換テスト")
    print("="*60)
    
    # テスト用の小さなモデルで試す
    test_model = "/workspace/outputs/merged_model_fp16"
    
    if not os.path.exists(test_model):
        print(f"❌ テストモデルが見つかりません: {test_model}")
        print("   最初にLoRAマージを実行する必要があります")
        return False
    
    print(f"テストモデル: {test_model}")
    
    # convert_hf_to_gguf.pyの存在確認
    convert_scripts = [
        "/workspace/llama.cpp/convert_hf_to_gguf.py",
        "/workspace/llama.cpp/convert-hf-to-gguf.py",
        "/workspace/llama.cpp/convert.py"
    ]
    
    convert_script = None
    for script in convert_scripts:
        if os.path.exists(script):
            convert_script = script
            break
    
    if not convert_script:
        print("❌ GGUF変換スクリプトが見つかりません")
        print("   試したパス:")
        for script in convert_scripts:
            print(f"   - {script}")
        return False
    
    print(f"使用するスクリプト: {convert_script}")
    
    # テスト変換
    output_file = "/tmp/test_model.gguf"
    cmd = f"cd /workspace/llama.cpp && python {convert_script} {test_model} --outfile {output_file} --outtype f16"
    
    print("\n変換コマンドを実行:")
    success = run_command_with_output(cmd)
    
    if success and os.path.exists(output_file):
        print(f"✅ GGUF変換成功: {output_file}")
        print(f"   ファイルサイズ: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        os.remove(output_file)  # テストファイルを削除
        return True
    else:
        print("❌ GGUF変換失敗")
        return False

def test_quantization_script():
    """qlora_to_ollama.pyの直接実行テスト"""
    print("\n" + "="*60)
    print("量子化スクリプトの直接実行テスト")
    print("="*60)
    
    script_path = "/workspace/scripts/qlora_to_ollama.py"
    
    if not os.path.exists(script_path):
        print(f"❌ スクリプトが見つかりません: {script_path}")
        return False
    
    # 環境変数設定
    os.environ['PYTHONPATH'] = '/workspace:' + os.environ.get('PYTHONPATH', '')
    
    # 直接実行（詳細なエラーを取得）
    cmd = f"python {script_path} 2>&1"
    
    print("スクリプトを実行:")
    run_command_with_output(cmd)

def main():
    """メイン処理"""
    print("量子化処理デバッグツール")
    print("="*60)
    
    # 環境チェック
    if not check_environment():
        print("\n❌ 環境チェックに失敗しました")
        return 1
    
    # GGUF変換テスト
    if not test_simple_conversion():
        print("\n⚠️ GGUF変換テストに失敗しました")
        print("llama.cppの再セットアップが必要かもしれません")
    
    # 量子化スクリプトの直接テスト
    print("\n最後に、完全な量子化パイプラインをテストします...")
    test_quantization_script()
    
    print("\n" + "="*60)
    print("デバッグ完了")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())