#!/usr/bin/env python3
"""
スコアの流れを詳細にデバッグ
"""

import sys
sys.path.insert(0, '/workspace')

import json
from loguru import logger

def debug_score_flow():
    """スコアがどこで失われているか追跡"""
    
    print("=" * 60)
    print("スコア伝達経路のデバッグ")
    print("=" * 60)
    
    # 1. HybridSearchEngineで直接検索
    print("\n1. HybridSearchEngineで直接検索")
    print("-" * 40)
    
    try:
        from src.rag.retrieval.hybrid_search import HybridSearchEngine, SearchQuery
        from src.rag.indexing.vector_store import QdrantVectorStore
        from src.rag.indexing.embedding_model import EmbeddingModelFactory
        
        # コンポーネント初期化
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=1024,
            url="http://qdrant:6333"
        )
        
        embedding_model = EmbeddingModelFactory.create("multilingual-e5-large")
        
        hybrid_search = HybridSearchEngine(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # 初期化（空のコーパスでも初期化は必要）
        hybrid_search.initialize([], [])
        
        # 検索クエリ
        query = SearchQuery(
            text="設計速度80km/hの道路",
            search_type="hybrid"
        )
        
        # 検索実行
        results = hybrid_search.search(query, top_k=1)
        
        if results:
            result = results[0]
            print(f"✅ 検索結果取得")
            print(f"   結果の型: {type(result)}")
            print(f"   vector_score: {result.vector_score:.3f}")
            print(f"   keyword_score: {result.keyword_score:.3f}")
            print(f"   hybrid_score: {result.hybrid_score:.3f}")
            
            # 計算チェック
            expected = result.vector_score * 0.7 + result.keyword_score * 0.3
            print(f"   期待値: {expected:.3f}")
            if abs(result.hybrid_score - expected) < 0.001:
                print(f"   ✅ 計算は正しい")
            else:
                print(f"   ⚠️ 計算に誤差: {abs(result.hybrid_score - expected):.3f}")
        else:
            print("❌ 検索結果なし")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. QueryEngineレベルでの確認
    print("\n2. QueryEngineレベルでの確認")
    print("-" * 40)
    
    try:
        from src.rag.core.rag_system import RAGSystem
        
        rag_system = RAGSystem()
        
        # 検索実行
        response = rag_system.query(
            query="設計速度80km/hの道路",
            top_k=1,
            search_type="hybrid"
        )
        
        if response and response.get('sources'):
            source = response['sources'][0]
            print(f"✅ QueryEngine結果取得")
            print(f"   vector_score: {source.get('vector_score', 'なし')}")
            print(f"   keyword_score: {source.get('keyword_score', 'なし')}")
            print(f"   hybrid_score: {source.get('hybrid_score', source.get('score', 'なし'))}")
            
            # フィールドをすべて表示
            print(f"\n   利用可能なフィールド:")
            for key in source.keys():
                if 'score' in key.lower():
                    print(f"     {key}: {source[key]}")
        else:
            print("❌ 結果なし")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    # 3. REST APIレベルでの確認
    print("\n3. REST APIレベルでの確認")
    print("-" * 40)
    
    try:
        import requests
        
        response = requests.post(
            "http://localhost:8050/rag/query",
            json={
                "query": "設計速度80km/hの道路",
                "top_k": 1,
                "search_type": "hybrid"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('sources'):
                source = data['sources'][0]
                print(f"✅ API結果取得")
                print(f"   vector_score: {source.get('vector_score', 'なし')}")
                print(f"   keyword_score: {source.get('keyword_score', 'なし')}")
                print(f"   hybrid_score: {source.get('hybrid_score', source.get('score', 'なし'))}")
                
                # 逆算
                v = source.get('vector_score', 0)
                h = source.get('hybrid_score', source.get('score', 0))
                if v > 0:
                    implied_k = (h - v * 0.7) / 0.3
                    print(f"\n   逆算されるキーワードスコア: {implied_k:.3f}")
        else:
            print(f"❌ API エラー: {response.status_code}")
            
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    debug_score_flow()