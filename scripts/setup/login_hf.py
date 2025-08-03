#!/usr/bin/env python3
"""
Hugging Faceログインスクリプト
"""

import os
import getpass
from huggingface_hub import login

def login_to_hf():
    """Hugging Faceにログイン"""
    print("=== Hugging Faceログイン ===")
    print()
    print("Hugging Faceトークンが必要です。")
    print("1. https://huggingface.co/settings/tokens にアクセス")
    print("2. 新しいトークンを作成")
    print("3. 以下のトークンを入力してください")
    print()
    
    try:
        token = getpass.getpass("Hugging Faceトークンを入力してください: ")
        
        if not token:
            print("トークンが入力されていません。")
            return False
        
        # ログイン
        login(token=token)
        print("✅ Hugging Faceログインが完了しました。")
        
        # ユーザー情報を確認
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"ログインユーザー: {user_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ログインに失敗しました: {e}")
        return False

def check_model_access():
    """モデルアクセスを確認"""
    print("\n=== モデルアクセス確認 ===")
    
    test_models = [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "microsoft/Phi-3.5-32B-Instruct"
    ]
    
    for model in test_models:
        print(f"確認中: {model}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            print(f"  ✅ アクセス可能: {model}")
        except Exception as e:
            print(f"  ❌ アクセス不可: {model} - {str(e)[:100]}...")

if __name__ == "__main__":
    if login_to_hf():
        check_model_access()
    else:
        print("\nログインに失敗しました。")
        print("トークンを確認してから再試行してください。") 