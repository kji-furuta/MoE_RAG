#!/usr/bin/env python3
"""
トークンファイル作成スクリプト
Windowsターミナルでのトークン入力問題を回避
"""

import os
import getpass

def create_token_file():
    """トークンファイルを作成"""
    print("=== Hugging Faceトークンファイル作成 ===")
    print()
    print("このスクリプトは、Hugging Faceトークンをファイルに保存します。")
    print("注意: トークンファイルは安全に保管してください。")
    print()
    
    # トークンを入力（表示されない場合があります）
    try:
        token = getpass.getpass("Hugging Faceトークンを入力してください: ")
    except:
        print("トークン入力に失敗しました。")
        print("手動でトークンファイルを作成してください。")
        return False
    
    if not token:
        print("トークンが入力されていません。")
        return False
    
    # ファイル名を指定
    token_file = ".hf_token"
    
    try:
        # トークンをファイルに保存
        with open(token_file, 'w') as f:
            f.write(token)
        
        # ファイルの権限を制限（Unix系のみ）
        try:
            os.chmod(token_file, 0o600)
        except:
            pass
        
        print(f"✅ トークンが '{token_file}' に保存されました。")
        print()
        print("使用方法:")
        print("1. 環境変数を設定:")
        print(f"   Windows: set HF_TOKEN=$(type {token_file})")
        print(f"   Linux/Mac: export HF_TOKEN=$(cat {token_file})")
        print()
        print("2. または、setup_hf_token.pyスクリプトを使用")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ ファイルの保存に失敗しました: {e}")
        return False

if __name__ == "__main__":
    create_token_file() 