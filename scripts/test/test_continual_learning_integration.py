#!/usr/bin/env python3
"""
継続学習管理システムのコンテナ統合テストスクリプト

このスクリプトは、継続学習管理システムがコンテナと正しく統合されているかをテストします。
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_continual_learning_api():
    """継続学習APIのテスト"""
    print("=== 継続学習管理システム統合テスト ===")
    
    base_url = "http://localhost:8050"
    
    # 1. 継続学習用モデル取得APIのテスト
    print("\n1. 継続学習用モデル取得APIのテスト")
    try:
        response = requests.get(f"{base_url}/api/continual-learning/models")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"取得したモデル数: {len(models)}")
            for i, model in enumerate(models[:3]):  # 最初の3つを表示
                print(f"  {i+1}. {model.get('name', 'N/A')} ({model.get('type', 'N/A')})")
        else:
            print(f"エラー: {response.text}")
            return False
    except Exception as e:
        print(f"API呼び出しエラー: {e}")
        return False
    
    # 2. 継続学習タスク取得APIのテスト
    print("\n2. 継続学習タスク取得APIのテスト")
    try:
        response = requests.get(f"{base_url}/api/continual-learning/tasks")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            tasks = response.json()
            print(f"実行中タスク数: {len(tasks)}")
        else:
            print(f"エラー: {response.text}")
    except Exception as e:
        print(f"API呼び出しエラー: {e}")
    
    # 3. 継続学習履歴取得APIのテスト
    print("\n3. 継続学習履歴取得APIのテスト")
    try:
        response = requests.get(f"{base_url}/api/continual-learning/history")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            history = response.json()
            print(f"履歴数: {len(history)}")
        else:
            print(f"エラー: {response.text}")
    except Exception as e:
        print(f"API呼び出しエラー: {e}")
    
    # 4. Webページアクセステスト
    print("\n4. 継続学習管理ページアクセステスト")
    try:
        response = requests.get(f"{base_url}/continual")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("継続学習管理ページにアクセス成功")
            if "継続学習管理システム" in response.text:
                print("✓ ページタイトルが正しく表示されています")
            else:
                print("⚠ ページタイトルが見つかりません")
        else:
            print(f"エラー: {response.text}")
            return False
    except Exception as e:
        print(f"ページアクセスエラー: {e}")
        return False
    
    # 5. システム情報APIのテスト
    print("\n5. システム情報APIのテスト")
    try:
        response = requests.get(f"{base_url}/api/system-info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            system_info = response.json()
            print("システム情報取得成功")
            if "gpu_info" in system_info:
                print("✓ GPU情報が含まれています")
            if "memory_info" in system_info:
                print("✓ メモリ情報が含まれています")
        else:
            print(f"エラー: {response.text}")
    except Exception as e:
        print(f"システム情報取得エラー: {e}")
    
    print("\n=== テスト完了 ===")
    return True

def test_container_integration():
    """コンテナ統合のテスト"""
    print("\n=== コンテナ統合テスト ===")
    
    # Dockerコンテナの状態確認
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ai-ft-container"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if "ai-ft-container" in result.stdout:
                print("✓ AI-FTコンテナが実行中です")
            else:
                print("⚠ AI-FTコンテナが見つかりません")
                return False
        else:
            print("⚠ Dockerコマンドの実行に失敗しました")
            return False
    except Exception as e:
        print(f"⚠ Docker確認エラー: {e}")
        return False
    
    # ポート8050の確認
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8050))
        sock.close()
        
        if result == 0:
            print("✓ ポート8050でサービスが応答しています")
        else:
            print("⚠ ポート8050でサービスが応答していません")
            return False
    except Exception as e:
        print(f"⚠ ポート確認エラー: {e}")
        return False
    
    return True

def main():
    """メイン関数"""
    print("継続学習管理システム統合テストを開始します...")
    
    # コンテナ統合テスト
    if not test_container_integration():
        print("\n❌ コンテナ統合テストが失敗しました")
        print("以下のコマンドでコンテナを起動してください:")
        print("cd docker && docker-compose up -d")
        sys.exit(1)
    
    # API統合テスト
    if not test_continual_learning_api():
        print("\n❌ API統合テストが失敗しました")
        print("Webサーバーが起動していない可能性があります")
        print("以下のコマンドでWebサーバーを起動してください:")
        print("docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload")
        sys.exit(1)
    
    print("\n✅ すべてのテストが成功しました！")
    print("継続学習管理システムは正常に動作しています。")
    print("ブラウザで http://localhost:8050/continual にアクセスしてください。")

if __name__ == "__main__":
    main() 