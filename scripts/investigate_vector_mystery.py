#!/usr/bin/env python3
"""
ベクトル検索ができるのにQdrantが0件の謎を調査
"""

import sys
sys.path.insert(0, '/workspace')

import requests
import json

def investigate_mystery():
    """ベクトルの謎を調査"""
    
    print("=" * 60)
    print("ベクトル検索の謎を調査")
    print("=" * 60)
    
    # 1. Qdrantの直接確認
    print("\n1. Qdrant APIで直接確認")
    print("-" * 40)
    
    # コレクション情報
    response = requests.get("http://qdrant:6333/collections/road_design_docs")
    if response.status_code == 200:
        info = response.json()
        result = info.get('result', {})
        print(f"コレクション: road_design_docs")
        print(f"  ベクトル数: {result.get('vectors_count', 0)}")
        print(f"  ステータス: {result.get('status', 'unknown')}")
    else:
        print(f"❌ エラー: {response.status_code}")
    
    # 2. 別のQdrantインスタンスの可能性
    print("\n2. 他のQdrantインスタンスを探索")
    print("-" * 40)
    
    # ローカルQdrantを確認
    try:
        response = requests.get("http://localhost:6333/collections")
        if response.status_code == 200:
            print("✅ localhost:6333でもQdrantが動作中")
            collections = response.json()
            print(f"  コレクション数: {len(collections.get('result', {}).get('collections', []))}")
    except:
        print("❌ localhost:6333では接続できません")
    
    # コンテナ内のQdrantを確認
    try:
        response = requests.get("http://host.docker.internal:6333/collections")
        if response.status_code == 200:
            print("✅ host.docker.internal:6333でQdrantが動作中")
    except:
        print("❌ host.docker.internal:6333では接続できません")
    
    # 3. 実際の検索がどこから来ているか
    print("\n3. 実際の検索結果の確認")
    print("-" * 40)
    
    # RAG APIで検索
    response = requests.post(
        "http://localhost:8050/rag/query",
        json={
            "query": "設計速度",
            "top_k": 1,
            "search_type": "vector"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        sources = data.get('sources', [])
        if sources:
            print(f"✅ 検索結果: {len(sources)}件")
            source = sources[0]
            print(f"  タイトル: {source.get('title', 'なし')[:50]}")
            print(f"  スコア: {source.get('score', 0):.3f}")
            
            # メタデータからヒントを探す
            metadata = source.get('metadata', {})
            if metadata:
                print(f"  メタデータのキー: {list(metadata.keys())[:5]}")
        else:
            print("❌ 検索結果なし")
    else:
        print(f"❌ API エラー: {response.status_code}")
    
    # 4. ファイルシステムのQdrantデータを確認
    print("\n4. ファイルシステムのQdrantデータ")
    print("-" * 40)
    
    import os
    import glob
    
    # Qdrantデータディレクトリを探す
    qdrant_dirs = glob.glob("/workspace/**/qdrant*", recursive=True)
    for dir_path in qdrant_dirs[:10]:
        if os.path.isdir(dir_path):
            print(f"  📁 {dir_path}")
            # コレクションディレクトリを確認
            collection_path = os.path.join(dir_path, "collection")
            if os.path.exists(collection_path):
                collections = os.listdir(collection_path)
                print(f"     コレクション: {collections[:3]}")
    
    # 5. プロセスで開かれているファイルを確認
    print("\n5. RAGシステムの設定確認")
    print("-" * 40)
    
    try:
        from src.rag.config.rag_config import load_config
        config = load_config()
        
        print(f"ベクトルストア設定:")
        print(f"  タイプ: {config.vector_store.type}")
        print(f"  パス: {getattr(config.vector_store, 'path', 'なし')}")
        print(f"  URL: {getattr(config.vector_store, 'url', 'なし')}")
        
        # 実際に使用されているベクトルストアを確認
        from src.rag.indexing.vector_store import QdrantVectorStore
        
        # URLベースで接続
        print("\nURL接続テスト (http://qdrant:6333):")
        vs_url = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=1024,
            url="http://qdrant:6333"
        )
        info_url = vs_url.get_collection_info()
        print(f"  ベクトル数: {info_url.get('vectors_count', 0)}")
        
        # パスベースで接続
        print("\nパス接続テスト (./qdrant_data):")
        vs_path = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=1024,
            path="./qdrant_data"
        )
        info_path = vs_path.get_collection_info()
        print(f"  ベクトル数: {info_path.get('vectors_count', 0)}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    investigate_mystery()