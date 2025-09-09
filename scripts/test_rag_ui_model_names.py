#!/usr/bin/env python3
"""
RAG UIのモデル名表示をテストする
"""

import requests
import json

def test_model_display():
    print("=" * 60)
    print("RAG UIモデル名表示テスト")
    print("=" * 60)
    
    # 1. Ollamaから直接モデルリストを取得
    print("\n1. Ollamaモデルリスト (ollama list と同じ):")
    print("-" * 40)
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            for model in data.get("models", []):
                print(f"  ✓ {model['name']}")
        else:
            print(f"  ✗ エラー: {response.status_code}")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
    
    # 2. 統合APIからモデルリストを取得
    print("\n2. 統合APIモデルリスト (/api/models):")
    print("-" * 40)
    try:
        response = requests.get("http://localhost:8050/api/models")
        if response.status_code == 200:
            data = response.json()
            
            # Ollamaモデル
            ollama_models = data.get("ollama_models", [])
            if ollama_models:
                print("  Ollamaモデル:")
                for model in ollama_models:
                    print(f"    ✓ {model['name']}")
            else:
                print("  ✗ Ollamaモデルが取得できません")
                
            # ファインチューニング済みモデル
            finetuned = data.get("finetuned_models", [])
            if finetuned:
                print("  ファインチューニング済みモデル:")
                for model in finetuned:
                    print(f"    ✓ {model['name']}")
        else:
            print(f"  ✗ エラー: {response.status_code}")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
    
    # 3. UIで表示される名前の確認
    print("\n3. UI表示名の確認:")
    print("-" * 40)
    print("  RAG UIでは以下のように表示されるはずです:")
    print("  - 5_deepseek-32b-finetuned:latest (そのまま)")
    print("  - deepseek-32b-finetuned:latest (そのまま)")
    print("  - 4_deepseek-32b-finetuned:latest (そのまま)")
    print("  - gpt-neox-20b-finetuned:latest (そのまま)")
    print("  - gpt-neox-20b-base:latest (そのまま)")
    print("  - deepseek-32b-base:latest (そのまま)")
    print("\n  ※ 以前のような変換はされません:")
    print("  ✗ DeepSeek-32B(ファインチューニング済み)(Unknown)")
    print("  ✓ 5_deepseek-32b-finetuned:latest")
    
    # 4. RAG設定の確認
    print("\n4. RAG設定ファイルの確認:")
    print("-" * 40)
    try:
        import yaml
        with open("/workspace/src/rag/config/rag_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            current_model = config.get('llm', {}).get('ollama', {}).get('model', 'Not set')
            print(f"  現在の設定モデル: {current_model}")
            
            if current_model == "5_deepseek-32b-finetuned:latest":
                print("  ✓ 設定は正しいモデル名を使用しています")
            else:
                print(f"  ⚠ 設定を '5_deepseek-32b-finetuned:latest' に更新してください")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print("\n📌 確認方法:")
    print("1. ブラウザで http://localhost:8050/rag にアクセス")
    print("2. 「LLM（大規模言語モデル）選択」ドロップダウンを確認")
    print("3. モデル名が 'ollama list' と同じ表示になっていることを確認")

if __name__ == "__main__":
    test_model_display()