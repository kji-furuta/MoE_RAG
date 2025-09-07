#!/usr/bin/env python3
"""RAGモデル選択機能のテスト"""

import requests
import json
import sys

def test_rag_query_with_model(model_name, query="道路舗装材料の種別について解説してください。"):
    """指定されたモデルでRAGクエリをテスト"""
    
    url = "http://localhost:8050/rag/query"
    
    payload = {
        "query": query,
        "top_k": 5,
        "search_type": "hybrid",
        "model": model_name,  # モデルを指定
        "include_sources": True
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\n{'='*60}")
    print(f"Testing RAG query with model: {model_name}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response received.")
            print(f"Answer preview: {result.get('answer', '')[:200]}...")
            
            # メタデータからモデル情報を確認
            metadata = result.get('metadata', {})
            if metadata:
                print(f"Metadata: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
            
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """複数のモデルでテストを実行"""
    
    # テストするモデルのリスト
    test_models = [
        "ollama:llama3.2:3b",
        "ollama:deepseek-32b-finetuned",
        "ollama:deepseekrag",
        None  # デフォルトモデルのテスト
    ]
    
    results = []
    
    for model in test_models:
        model_name = model if model else "Default (from config)"
        success = test_rag_query_with_model(model)
        results.append((model_name, success))
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("Test Results Summary:")
    print(f"{'='*60}")
    
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model_name}")
    
    # 全体の成功判定
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()