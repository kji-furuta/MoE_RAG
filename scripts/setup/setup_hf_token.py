#!/usr/bin/env python3
"""
Hugging Faceトークン設定スクリプト
Windowsターミナルでのトークン入力問題を解決
"""

import os
import getpass
from huggingface_hub import login

def setup_hf_token():
    """Hugging Faceトークンを設定"""
    print("=== Hugging Faceトークン設定 ===")
    print()
    
    # 方法1: 環境変数から取得
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("環境変数HF_TOKENが設定されていません。")
        print("以下のいずれかの方法でトークンを設定してください：")
        print()
        print("方法1: 環境変数を設定")
        print("  Windows: set HF_TOKEN=your_token_here")
        print("  Linux/Mac: export HF_TOKEN=your_token_here")
        print()
        print("方法2: 直接入力（Windowsでは表示されない場合があります）")
        print("方法3: ファイルから読み込み")
        print()
        
        choice = input("方法を選択してください (1/2/3): ").strip()
        
        if choice == "1":
            print("環境変数HF_TOKENを設定してから、このスクリプトを再実行してください。")
            return False
            
        elif choice == "2":
            try:
                token = getpass.getpass("Hugging Faceトークンを入力してください: ")
            except:
                print("トークン入力に失敗しました。")
                return False
                
        elif choice == "3":
            token_file = input("トークンファイルのパスを入力してください: ").strip()
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
            except:
                print("ファイルの読み込みに失敗しました。")
                return False
        else:
            print("無効な選択です。")
            return False
    
    if not token:
        print("トークンが設定されていません。")
        return False
    
    try:
        # Hugging Faceにログイン
        login(token=token)
        print("✅ Hugging Faceトークンの設定が完了しました。")
        return True
        
    except Exception as e:
        print(f"❌ ログインに失敗しました: {e}")
        return False

def test_models():
    """利用可能なモデルをテスト"""
    print("\n=== モデルアクセステスト ===")
    
    test_models = [
        "Qwen/Qwen2.5-14B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct"
    ]
    
    for model in test_models:
        print(f"テスト中: {model}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            print(f"  ✅ {model}: アクセス成功")
        except Exception as e:
            print(f"  ❌ {model}: アクセス失敗 - {str(e)[:100]}...")

if __name__ == "__main__":
    if setup_hf_token():
        test_models()
    else:
        print("\nトークン設定に失敗しました。")
        print("Hugging Faceのウェブサイトからトークンを取得してください：")
        print("https://huggingface.co/settings/tokens") 