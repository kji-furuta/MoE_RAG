#!/usr/bin/env python3
"""
RAGシステムのモデル選択機能テストスクリプト
"""

import requests
import json
import time
from pathlib import Path
import sys
import subprocess

# APIエンドポイント
BASE_URL = "http://localhost:8050"
RAG_API = f"{BASE_URL}/rag"

def check_ollama_models():
    """Ollamaで利用可能なモデルを確認"""
    print("=" * 60)
    print("📋 Ollamaモデル確認")
    print("=" * 60)
    
    try:
        # Ollamaモデルリストを取得
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ 利用可能なOllamaモデル:")
            print(result.stdout)
        else:
            print("❌ Ollamaモデル取得エラー")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Ollama確認エラー: {e}")
    
    print()

def test_model_update(model_name):
    """指定したモデルに設定を更新"""
    print(f"🔄 モデル設定を更新中: {model_name}")
    
    settings = {
        "llm_model": model_name,
        "embedding_model": "intfloat/multilingual-e5-large",
        "temperature": 0.6
    }
    
    try:
        response = requests.post(
            f"{RAG_API}/update-settings",
            json=settings,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print(f"✅ モデル設定更新成功: {model_name}")
                return True
            else:
                print(f"❌ 設定更新失敗: {data.get('message')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ 設定更新エラー: {e}")
    
    return False

def test_rag_query(query, model_name=None):
    """RAGクエリテスト"""
    print(f"\n📝 クエリテスト: '{query}'")
    if model_name:
        print(f"   使用モデル: {model_name}")
    
    request_data = {
        "query": query,
        "top_k": 3,
        "search_type": "hybrid",
        "include_sources": True
    }
    
    try:
        response = requests.post(
            f"{RAG_API}/query",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ クエリ成功")
            print(f"   回答文字数: {len(data.get('answer', ''))}")
            
            # 使用されたモデルを確認（ログから推測）
            answer = data.get('answer', '')
            if 'DeepSeek' in answer or 'deepseek' in answer.lower():
                print("   推定使用モデル: DeepSeek-32B")
            elif 'Llama' in answer or 'llama' in answer.lower():
                print("   推定使用モデル: Llama 3.2")
            
            # 回答の最初の200文字を表示
            print(f"   回答冒頭: {answer[:200]}...")
            return True
            
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"   詳細: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ クエリエラー: {e}")
    
    return False

def get_current_config():
    """現在の設定を取得"""
    print("\n📊 現在のRAG設定確認")
    
    try:
        response = requests.get(
            f"{RAG_API}/system-info",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            config = data.get('system_info', {}).get('config', {})
            llm_config = config.get('llm', {})
            
            print(f"✅ 現在の設定:")
            print(f"   Provider: {llm_config.get('provider', '不明')}")
            print(f"   Model Name: {llm_config.get('model_name', '不明')}")
            print(f"   Ollama Model: {llm_config.get('ollama_model', '不明')}")
            print(f"   Temperature: {llm_config.get('temperature', '不明')}")
            
            if llm_config.get('ollama'):
                print(f"   Ollama Config:")
                print(f"     - Model: {llm_config['ollama'].get('model', '不明')}")
                print(f"     - Base URL: {llm_config['ollama'].get('base_url', '不明')}")
            
            return llm_config
        else:
            print(f"❌ 設定取得失敗: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ 設定取得エラー: {e}")
    
    return None

def main():
    """メインテスト実行"""
    print("🚀 RAGモデル選択機能テスト開始")
    print("=" * 60)
    
    # 1. Ollamaモデル確認
    check_ollama_models()
    
    # 2. 現在の設定確認
    current_config = get_current_config()
    
    # 3. テストクエリ
    test_query = "道路の設計速度80km/hの場合の最小曲線半径は？"
    
    print("\n" + "=" * 60)
    print("📝 モデル切り替えテスト")
    print("=" * 60)
    
    # 4. Llama 3.2でテスト
    print("\n--- Test 1: Llama 3.2 3B ---")
    if test_model_update("ollama:llama3.2:3b"):
        time.sleep(2)  # 設定反映待ち
        test_rag_query(test_query, "llama3.2:3b")
    
    # 5. DeepSeek-32Bでテスト
    print("\n--- Test 2: DeepSeek-32B Finetuned ---")
    if test_model_update("ollama:deepseek-32b-finetuned"):
        time.sleep(2)  # 設定反映待ち
        test_rag_query(test_query, "deepseek-32b-finetuned")
    
    # 6. 最終設定確認
    print("\n" + "=" * 60)
    print("📊 テスト後の設定確認")
    print("=" * 60)
    get_current_config()
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
    print("=" * 60)
    print("\n💡 Web UIでの確認方法:")
    print("1. http://localhost:8050/rag にアクセス")
    print("2. 'RAGシステム設定'タブを開く")
    print("3. 'LLM（大規模言語モデル）選択'プルダウンからモデルを選択")
    print("4. '設定を保存'ボタンをクリック")
    print("5. 'ハイブリッド検索・質問応答'タブで質問を実行")

if __name__ == "__main__":
    main()