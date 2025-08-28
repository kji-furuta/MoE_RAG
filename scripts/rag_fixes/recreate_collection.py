#!/usr/bin/env python3
"""
Qdrantコレクションを再作成して次元を統一
"""

import sys
import os
sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

import logging
from pathlib import Path

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_collection():
    """コレクションを再作成"""
    
    try:
        from src.rag.indexing.vector_store import QdrantVectorStore
        from qdrant_client.models import Distance, VectorParams
        import uuid
        import numpy as np
        
        print("=" * 60)
        print("Qdrantコレクション再作成")
        print("=" * 60)
        
        # 1. ベクトルストアの初期化（1024次元）
        print("\n1. ベクトルストアの初期化")
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=1024,
            path="./data/qdrant"
        )
        print(f"  ✅ 初期化完了（1024次元）")
        
        # 2. 既存コレクションの削除と再作成
        print("\n2. コレクションの再作成")
        try:
            # 既存コレクションを削除
            vector_store.client.delete_collection("road_design_docs")
            print(f"  ✅ 既存コレクションを削除")
        except Exception as e:
            print(f"  ℹ️  既存コレクションなし: {e}")
        
        # 新規作成
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff
        
        vector_store.client.create_collection(
            collection_name="road_design_docs",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=2
            )
        )
        print(f"  ✅ 新規コレクション作成（1024次元、Cosine距離）")
        
        # 3. テストデータの追加
        print("\n3. テストデータの追加")
        
        test_texts = [
            "道路設計における最小曲線半径の決定要因について説明します。",
            "設計速度60km/hの場合の安全基準と適用条件を確認してください。",
            "縦断勾配の制限値と適用条件について、道路構造令に基づいて解説します。"
        ]
        
        # ダミーの埋め込みベクトル（1024次元）
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in test_texts]
        
        # メタデータ
        metadatas = [
            {
                "doc_id": f"test_{i}",
                "title": f"テスト文書{i+1}",
                "category": "test",
                "source": "initialization"
            }
            for i in range(len(test_texts))
        ]
        
        # UUID生成
        ids = [str(uuid.uuid4()) for _ in test_texts]
        
        # データ追加
        vector_store.add_documents(
            texts=test_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  ✅ {len(test_texts)}件のテストデータを追加")
        
        # 4. コレクション情報の確認
        print("\n4. コレクション情報の確認")
        info = vector_store.get_collection_info()
        print(f"  ステータス: {info.get('status', 'unknown')}")
        print(f"  ベクトル数: {info.get('vectors_count', 0)}")
        print(f"  インデックス済み: {info.get('indexed_vectors_count', 0)}")
        
        # 5. 検索テスト
        print("\n5. 検索テスト")
        query_embedding = np.random.randn(1024).astype(np.float32)
        
        try:
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=3
            )
            print(f"  ✅ 検索成功: {len(results)}件の結果")
            for i, result in enumerate(results, 1):
                print(f"    {i}. スコア={result.score:.3f}, ID={result.id[:8]}...")
        except Exception as e:
            print(f"  ⚠️  検索エラー: {e}")
        
        print("\n" + "=" * 60)
        print("✅ コレクションの再作成が完了しました")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = recreate_collection()
    sys.exit(0 if success else 1)
