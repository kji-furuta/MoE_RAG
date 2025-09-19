#!/usr/bin/env python3
"""
キーワード検索エンジンのコーパス初期化を確認
"""

import sys
sys.path.insert(0, '/workspace')
from src.rag.core.query_engine import RoadDesignQueryEngine
from loguru import logger

def test_keyword_corpus():
    """キーワード検索エンジンのコーパス初期化を確認"""
    
    print("=" * 60)
    print("キーワード検索エンジンのコーパス初期化確認")
    print("=" * 60)
    
    try:
        # クエリエンジンを初期化
        print("\n1. QueryEngine初期化中...")
        qe = RoadDesignQueryEngine()
        qe.initialize()
        print("✅ QueryEngine初期化完了")
        
        # ハイブリッド検索エンジンの状態確認
        if hasattr(qe, 'hybrid_search'):
            hybrid = qe.hybrid_search
            print("\n2. HybridSearchEngine状態:")
            print(f"   - 初期化済み: {hybrid.is_ready}")
            
            # キーワードエンジンの状態確認
            if hasattr(hybrid, 'keyword_engine'):
                ke = hybrid.keyword_engine
                print("\n3. KeywordSearchEngine状態:")
                print(f"   - Fitted: {ke.is_fitted}")
                
                if hasattr(ke, 'corpus_texts'):
                    print(f"   - コーパスサイズ: {len(ke.corpus_texts)}")
                    if ke.corpus_texts:
                        print(f"   - サンプルテキスト: {ke.corpus_texts[0][:100]}...")
                
                if hasattr(ke, 'corpus_ids'):
                    print(f"   - ID数: {len(ke.corpus_ids)}")
                    if ke.corpus_ids:
                        print(f"   - サンプルID: {ke.corpus_ids[:3]}")
                
                if hasattr(ke, 'corpus_vectors') and ke.corpus_vectors is not None:
                    print(f"   - TF-IDF行列: {ke.corpus_vectors.shape}")
                    print(f"   - 特徴数: {ke.corpus_vectors.shape[1]}")
                else:
                    print("   - TF-IDF行列: None (空のコーパス)")
                
                # テスト検索
                if ke.is_fitted and ke.corpus_vectors is not None:
                    print("\n4. テスト検索:")
                    try:
                        results = ke.search("道路", top_k=3)
                        print(f"   - 検索結果数: {len(results)}")
                        for doc_id, score in results[:3]:
                            print(f"     - ID: {doc_id[:50]}, Score: {score:.4f}")
                    except Exception as search_error:
                        print(f"   ❌ 検索エラー: {search_error}")
                else:
                    print("\n4. テスト検索: スキップ（コーパスが空）")
            else:
                print("❌ keyword_engineが見つかりません")
        else:
            print("❌ hybrid_searchが見つかりません")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyword_corpus()