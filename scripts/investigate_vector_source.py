#!/usr/bin/env python3
"""
ベクトル検索結果の出所を調査
"""

import sys
sys.path.insert(0, '/workspace')

import requests
import json

def investigate():
    """検索結果がどこから来ているか調査"""
    
    print("=" * 60)
    print("ベクトル検索結果の出所調査")
    print("=" * 60)
    
    # 1. Qdrantの別のコレクションを確認
    print("\n1. Qdrantの全コレクションを確認:")
    response = requests.get("http://qdrant:6333/collections")
    if response.status_code == 200:
        data = response.json()
        collections = data.get('result', {}).get('collections', [])
        for coll in collections:
            name = coll.get('name')
            print(f"\nコレクション: {name}")
            
            # 各コレクションの詳細
            detail_response = requests.get(f"http://qdrant:6333/collections/{name}")
            if detail_response.status_code == 200:
                detail = detail_response.json()
                result = detail.get('result', {})
                print(f"  ベクトル数: {result.get('vectors_count', 0)}")
                print(f"  ステータス: {result.get('status', 'unknown')}")
    
    # 2. ローカルQdrantのパスを確認
    print("\n2. Qdrantのストレージ設定:")
    import os
    qdrant_paths = [
        "/workspace/qdrant_data",
        "./qdrant_data",
        "/tmp/qdrant_data"
    ]
    
    for path in qdrant_paths:
        if os.path.exists(path):
            print(f"  ✅ 存在: {path}")
            # ディレクトリ内容を確認
            try:
                contents = os.listdir(path)
                print(f"     内容: {contents[:5]}")
            except:
                pass
        else:
            print(f"  ❌ 存在しない: {path}")
    
    # 3. RAG設定を確認
    print("\n3. RAG設定の確認:")
    try:
        from src.rag.config.rag_config import load_config
        config = load_config()
        print(f"  ベクトルストアタイプ: {config.vector_store.type}")
        print(f"  ベクトルストアパス: {config.vector_store.path}")
        print(f"  ベクトルストアURL: {getattr(config.vector_store, 'url', 'なし')}")
    except Exception as e:
        print(f"  設定読み込みエラー: {e}")
    
    # 4. 実際に使われているQdrantクライアントを確認
    print("\n4. 実際のQdrantクライアント設定:")
    try:
        from src.rag.indexing.vector_store import QdrantVectorStore
        vector_store = QdrantVectorStore()
        
        # クライアントの詳細を取得
        print(f"  コレクション名: {vector_store.collection_name}")
        info = vector_store.get_collection_info()
        print(f"  ベクトル数: {info.get('vectors_count', 0)}")
        
        # 実際に検索してみる
        import numpy as np
        dummy_vector = np.random.randn(1024)
        results = vector_store.search(dummy_vector, top_k=1)
        print(f"  検索結果数: {len(results)}")
        if results:
            print(f"  最初の結果ID: {results[0].id}")
            
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate()