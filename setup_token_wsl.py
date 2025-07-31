#!/usr/bin/env python3
"""
WSL環境でのトークン設定スクリプト
"""

import os
import json
from pathlib import Path

def setup_token_wsl():
    """WSL環境でトークンを設定"""
    print("=== WSLトークン設定 ===")
    print()
    
    print("WSL環境では、以下の方法でトークンを設定してください：")
    print()
    print("方法1: 環境変数を直接設定")
    print("export HF_TOKEN=your_token_here")
    print()
    print("方法2: トークンファイルを作成")
    print("1. .hf_tokenファイルを作成")
    print("2. トークンを保存")
    print("3. 環境変数を設定")
    print()
    print("方法3: .bashrcに追加")
    print("echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc")
    print()
    
    # 現在のトークンを確認
    token = os.environ.get('HF_TOKEN')
    if token:
        print(f"✅ 現在のトークン: {token[:10]}...")
    else:
        print("❌ トークンが設定されていません")
    
    # トークンファイルを確認
    token_file = Path.home() / ".hf_token"
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                file_token = f.read().strip()
            print(f"✅ トークンファイル: {file_token[:10]}...")
        except:
            print("❌ トークンファイルの読み込みに失敗")
    else:
        print("❌ トークンファイルが存在しません")
    
    print()
    
    # トークンファイルを作成するか確認
    choice = input("トークンファイルを作成しますか？ (y/n): ").strip().lower()
    
    if choice == 'y':
        create_token_file()
    else:
        print("手動でトークンを設定してください。")

def create_token_file():
    """トークンファイルを作成"""
    print("\n=== トークンファイル作成 ===")
    
    token_file = Path.home() / ".hf_token"
    
    print("以下の手順でトークンファイルを作成してください：")
    print()
    print("1. メモ帳またはテキストエディタでファイルを作成")
    print(f"   ファイル名: {token_file}")
    print()
    print("2. ファイルにHugging Faceトークンを入力")
    print("   （1行のみ、改行なし）")
    print()
    print("3. ファイルを保存")
    print()
    print("4. 権限を設定:")
    print(f"   chmod 600 {token_file}")
    print()
    print("5. 環境変数を設定:")
    print(f"   export HF_TOKEN=$(cat {token_file})")
    print()
    print("6. 永続化する場合:")
    print(f"   echo 'export HF_TOKEN=\$(cat {token_file})' >> ~/.bashrc")
    
    # ファイルが存在するか確認
    if token_file.exists():
        print(f"\n✅ トークンファイルが既に存在します: {token_file}")
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
            print(f"   トークン: {token[:10]}...")
        except:
            print("   ファイルの読み込みに失敗")
    else:
        print(f"\n❌ トークンファイルが存在しません: {token_file}")

def test_token():
    """トークンをテスト"""
    print("\n=== トークンテスト ===")
    
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ 環境変数HF_TOKENが設定されていません")
        return False
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ トークンが有効です")
        print(f"   ユーザー: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ トークンが無効です: {e}")
        return False

def main():
    """メイン処理"""
    setup_token_wsl()
    
    # トークンテスト
    if test_token():
        print("\n✅ トークン設定が完了しました")
        print("モデルダウンロードを試行できます")
    else:
        print("\n❌ トークン設定が必要です")
        print("上記の手順でトークンを設定してください")

if __name__ == "__main__":
    main() 