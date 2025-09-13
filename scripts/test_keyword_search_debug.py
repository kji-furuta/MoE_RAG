#!/usr/bin/env python3
"""
キーワード検索のデバッグスクリプト
ID不一致やTF-IDF学習状態を確認
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.core.query_engine import QueryEngine
from src.rag.config.rag_config import RAGConfig
from src.rag.retrieval.hybrid_search import SearchQuery
from loguru import logger
import json

def debug_keyword_search():
    """キーワード検索の問題をデバッグ"""
    
    print("=" * 60)
    print("キーワード検索デバッグ")
    print("=" * 60)
    
    try:
        # RAG設定を読み込み
        config = RAGConfig.from_yaml("configs/rag_config.yaml")
        
        # QueryEngineを初期化
        print("\n1. QueryEngineを初期化中...")
        query_engine = QueryEngine(config)
        query_engine.initialize()
        
        # HybridSearchEngineの状態を確認
        if hasattr(query_engine, 'hybrid_search') and query_engine.hybrid_search:
            hybrid_search = query_engine.hybrid_search
            keyword_engine = hybrid_search.keyword_engine
            
            print("\n2. KeywordSearchEngineの状態:")
            print(f"   - is_fitted: {keyword_engine.is_fitted}")
            
            if keyword_engine.corpus_texts:
                print(f"   - corpus_texts数: {len(keyword_engine.corpus_texts)}")
                print(f"   - corpus_ids数: {len(keyword_engine.corpus_ids)}")
                
                # コーパスIDのサンプルを表示
                print("\n   コーパスIDのサンプル（最初の5件）:")
                for i, (doc_id, text) in enumerate(zip(keyword_engine.corpus_ids[:5], 
                                                       keyword_engine.corpus_texts[:5]), 1):
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    print(f"     {i}. ID: {doc_id}")
                    print(f"        Text: {text_preview}")
            else:
                print("   ⚠️ corpus_textsが空です")
            
            # ベクトルストアのIDを確認
            print("\n3. ベクトルストアのIDを確認:")
            vector_store = query_engine.vector_store
            
            # スクロール検索でベクトルストアのIDを取得
            scroll_result = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=5,
                with_payload=False,
                with_vectors=False
            )
            
            if scroll_result and scroll_result[0]:
                points, _ = scroll_result
                print("   ベクトルストアIDのサンプル（最初の5件）:")
                for i, point in enumerate(points, 1):
                    print(f"     {i}. ID: {point.id} (タイプ: {type(point.id).__name__})")
            
            # テストクエリを実行してIDマッチングを確認
            print("\n4. テストクエリでIDマッチングを確認:")
            test_query = "設計速度"
            
            # キーワード検索を直接実行
            if keyword_engine.is_fitted and keyword_engine.corpus_vectors is not None:
                keyword_results = keyword_engine.search(test_query, top_k=5)
                print(f"   クエリ: '{test_query}'")
                print(f"   キーワード検索結果数: {len(keyword_results)}")
                
                if keyword_results:
                    print("\n   キーワード検索結果のID:")
                    for doc_id, score in keyword_results[:5]:
                        print(f"     - ID: {doc_id}, スコア: {score:.3f}")
                else:
                    print("   ⚠️ キーワード検索結果が0件です")
                    
                    # TF-IDFベクトライザーの状態を確認
                    if hasattr(keyword_engine.vectorizer, 'vocabulary_'):
                        vocab_size = len(keyword_engine.vectorizer.vocabulary_)
                        print(f"\n   TF-IDF語彙サイズ: {vocab_size}")
                        
                        if vocab_size > 0:
                            # 語彙のサンプルを表示
                            sample_vocab = list(keyword_engine.vectorizer.vocabulary_.keys())[:10]
                            print(f"   語彙サンプル: {sample_vocab}")
                    else:
                        print("   ⚠️ TF-IDFベクトライザーが学習されていません")
            
            # ハイブリッド検索を実行してIDマッチングを確認
            print("\n5. ハイブリッド検索でIDマッチングを確認:")
            search_query = SearchQuery(text=test_query, search_type="hybrid")
            hybrid_results = hybrid_search.search(search_query, top_k=3)
            
            for i, result in enumerate(hybrid_results, 1):
                print(f"\n   結果 {i}:")
                print(f"     - Result ID: {result.id}")
                print(f"     - ベクトルスコア: {result.vector_score:.3f}")
                print(f"     - キーワードスコア: {result.keyword_score:.3f}")
                print(f"     - ハイブリッドスコア: {result.hybrid_score:.3f}")
                
                # IDがキーワード検索結果に含まれているか確認
                if keyword_results:
                    keyword_dict = dict(keyword_results)
                    if result.id in keyword_dict:
                        print(f"     ✅ IDがキーワード検索結果に存在: {keyword_dict[result.id]:.3f}")
                    else:
                        print(f"     ❌ IDがキーワード検索結果に存在しません")
                        # ID形式の違いを確認
                        print(f"        Result ID type: {type(result.id).__name__}")
                        if keyword_engine.corpus_ids:
                            print(f"        Corpus ID type: {type(keyword_engine.corpus_ids[0]).__name__}")
            
        else:
            print("\n❌ HybridSearchEngineが見つかりません")
            
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("デバッグ完了")
    print("=" * 60)

if __name__ == "__main__":
    debug_keyword_search()