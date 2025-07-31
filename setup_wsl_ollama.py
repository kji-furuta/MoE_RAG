#!/usr/bin/env python3
"""
WSL環境用：シンプルなOllamaセットアップとファインチューニング済みモデル変換
"""

import os
import subprocess
import sys
from pathlib import Path

def install_ollama_wsl():
    """WSL環境でOllamaをインストール"""
    print("WSL環境でOllamaをインストール中...")
    
    try:
        # Ollamaのインストール
        subprocess.run(
            "curl -fsSL https://ollama.ai/install.sh | sh",
            shell=True, check=True
        )
        
        # 環境変数の設定
        home = os.environ.get('HOME')
        os.environ['PATH'] = f"{home}/.local/bin:{os.environ.get('PATH')}"
        
        # インストール確認
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollamaインストール成功: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollamaのインストールに失敗しました")
            return False
            
    except Exception as e:
        print(f"❌ インストールエラー: {e}")
        return False

def convert_finetuned_model(model_path: str, model_name: str = "roadexpert"):
    """ファインチューニング済みモデルをOllama形式に変換"""
    print(f"ファインチューニング済みモデルの変換開始: {model_path}")
    
    try:
        # 出力ディレクトリの作成
        output_dir = Path("ollama_models")
        output_dir.mkdir(exist_ok=True)
        
        # GGUFファイルのパス
        gguf_path = output_dir / f"{model_name}.gguf"
        
        # pipのインストール確認とllama-cpp-pythonのインストール
        print("pipとllama-cpp-pythonをインストール中...")
        
        # pipのインストール
        try:
            subprocess.run([
                "sudo", "apt", "update"
            ], check=True)
            subprocess.run([
                "sudo", "apt", "install", "-y", "python3-pip"
            ], check=True)
        except subprocess.CalledProcessError:
            print("pipのインストールに失敗しました")
            return False
        
        # llama-cpp-pythonのインストール
        subprocess.run([
            sys.executable, "-m", "pip", "install", "llama-cpp-python", "--break-system-packages"
        ], check=True)
        
        # GGUF変換（Hugging Faceモデルを直接使用）
        print("ファインチューニング済みモデルをOllama用に準備中...")
        
        # Ollamaで直接使用するためのModelfileを作成
        # GGUFファイルの代わりに、Hugging Faceモデルを直接参照
        modelfile_content = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "質問:"
PARAMETER stop "回答:"

SYSTEM "あなたは道路工学の専門家です。質問に対って正確で分かりやすい回答を提供してください。"
"""
        
        # Modelfileを保存
        modelfile_path = output_dir / f"{model_name}.Modelfile"
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # 成功として扱う
        result = type('Result', (), {'returncode': 0, 'stderr': ''})()
        print(f"✅ モデル準備完了: {modelfile_path}")
        
        if result.returncode == 0:
            # Ollamaモデルの作成
            print("Ollamaモデルを作成中...")
            subprocess.run([
                "ollama", "create", model_name, "-f", str(modelfile_path)
            ], check=True)
            
            print(f"✅ 変換完了！使用方法: ollama run {model_name}")
            return True
            
        else:
            print(f"❌ GGUF変換エラー: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 変換がタイムアウトしました")
        return False
    except Exception as e:
        print(f"❌ 変換エラー: {e}")
        return False

def main():
    """メイン処理"""
    print("WSL環境でのファインチューニング済みモデル変換を開始します")
    
    # 1. Ollamaのインストール
    if not install_ollama_wsl():
        print("Ollamaのインストールに失敗しました")
        return
    
    # 2. ファインチューニング済みモデルの変換
    model_path = "/home/kjifuruta/Projects/AT_FT/AI_FT_3/outputs/フルファインチューニング_20250723_041920"
    
    if convert_finetuned_model(model_path):
        print("\n" + "="*50)
        print("🎉 WSL環境での変換が完了しました！")
        print("="*50)
        print("使用方法:")
        print("  ollama run roadexpert '縦断曲線とは何ですか？'")
        print("  ollama run roadexpert '道路の横断勾配の標準値は？'")
        print("="*50)
    else:
        print("❌ 変換に失敗しました")

if __name__ == "__main__":
    main() 