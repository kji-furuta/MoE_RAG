#!/usr/bin/env python3
"""
HybridSearchResultオブジェクトの属性を詳細に調査
"""

import sys
sys.path.insert(0, '/workspace')

from src.rag.retrieval.hybrid_search import HybridSearchEngine, SearchQuery
from src.rag.indexing.vector_store import QdrantVectorStore

# ハイブリッド検索を初期化
vector_store = QdrantVectorStore()
hybrid_search = HybridSearchEngine(vector_store=vector_store)

# 検索を実行
query = SearchQuery(text="設計速度80km/hの道路", search_type="hybrid")
results = hybrid_search.search(query, top_k=1)

if results:
    result = results[0]
    print("=" * 60)
    print("HybridSearchResultオブジェクトの分析")
    print("=" * 60)
    
    print("\n1. オブジェクトの型:")
    print(f"   {type(result)}")
    
    print("\n2. __dict__の内容:")
    print(f"   {result.__dict__}")
    
    print("\n3. 各属性の値:")
    print(f"   vector_score: {getattr(result, 'vector_score', 'NOT FOUND')}")
    print(f"   keyword_score: {getattr(result, 'keyword_score', 'NOT FOUND')}")
    print(f"   hybrid_score: {getattr(result, 'hybrid_score', 'NOT FOUND')}")
    
    print("\n4. hasattr チェック:")
    print(f"   hasattr(result, 'vector_score'): {hasattr(result, 'vector_score')}")
    print(f"   hasattr(result, 'keyword_score'): {hasattr(result, 'keyword_score')}")
    print(f"   hasattr(result, 'hybrid_score'): {hasattr(result, 'hybrid_score')}")
    
    print("\n5. dataclassフィールド（もしdataclassなら）:")
    if hasattr(result, '__dataclass_fields__'):
        for field_name in result.__dataclass_fields__:
            print(f"   {field_name}: {getattr(result, field_name)}")
else:
    print("検索結果がありません")