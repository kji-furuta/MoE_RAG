#!/usr/bin/env python3
"""
キーワードスコアの問題を詳細に調査
"""

import sys
sys.path.insert(0, '/workspace')
from src.rag.core.rag_system import RAGSystem
from loguru import logger

def test_keyword_scores():
    """キーワードスコア問題の調査"""
    
    print("=" * 60)
    print("キーワードスコア詳細調査")
    print("=" * 60)
    
    # RAGシステムを初期化
    logger.info("Initializing RAG system")
    rag = RAGSystem()
    
    # キーワード検索エンジンの状態確認
    if hasattr(rag, 'query_engine') and hasattr(rag.query_engine, 'hybrid_search'):
        hybrid = rag.query_engine.hybrid_search
        if hasattr(hybrid, 'keyword_engine'):
            ke = hybrid.keyword_engine
            print(f"\nキーワードエンジン状態:")
            print(f"  初期化済み: {ke.initialized}")
            if hasattr(ke, 'corpus_size'):
                print(f"  コーパスサイズ: {ke.corpus_size}")
            if hasattr(ke, 'vectorizer') and ke.vectorizer:
                vocab = ke.vectorizer.vocabulary_
                print(f"  語彙サイズ: {len(vocab) if vocab else 0}")
    
    # テストクエリ実行
    queries = [
        "設計速度80km/hの道路",
        "道路の横断勾配",
        "インターチェンジ"
    ]
    
    for query_text in queries:
        print(f"\n{'='*40}")
        print(f"クエリ: {query_text}")
        print("-" * 40)
        
        # 検索実行
        response = rag.query(
            query=query_text,
            top_k=3,
            search_type="hybrid"
        )
        
        if response and response.get("sources"):
            for i, source in enumerate(response["sources"][:3], 1):
                print(f"\n結果 {i}:")
                title = source.get("title", "N/A")
                print(f"  Title: {title[:50]}")
                
                # スコア取得
                v_score = source.get("vector_score", 0)
                k_score = source.get("keyword_score", 0)
                h_score = source.get("hybrid_score", source.get("score", 0))
                
                print(f"  Vector Score: {v_score:.4f}")
                print(f"  Keyword Score: {k_score:.4f}")
                print(f"  Hybrid Score: {h_score:.4f}")
                
                # 期待値計算
                expected = v_score * 0.7 + k_score * 0.3
                print(f"  Expected: {expected:.4f}")
                
                # 差分チェック
                diff = abs(h_score - expected)
                if diff > 0.001:
                    print(f"  ⚠️ 差分: {diff:.4f}")
                    
                    # 技術用語ブーストの可能性
                    if v_score > 0 and k_score == 0:
                        # キーワードスコアが0の場合、ブーストから逆算
                        boost_factor = h_score / (v_score * 0.7) - 1.0
                        if boost_factor > 0:
                            print(f"  💡 技術用語ブースト: {boost_factor:.3f} ({boost_factor*100:.1f}%)")
                else:
                    print(f"  ✅ スコア計算正常")
        else:
            print("❌ 検索結果なし")

if __name__ == "__main__":
    test_keyword_scores()