#!/usr/bin/env python3
"""
シンプルなキーワード検索デバッグ
"""

import sys
sys.path.insert(0, '/workspace')

from src.rag.retrieval.hybrid_search import KeywordSearchEngine
from loguru import logger

def test_keyword_search():
    """キーワード検索の基本動作を確認"""
    
    print("=" * 60)
    print("キーワード検索エンジンテスト")
    print("=" * 60)
    
    # テスト用の文書
    test_docs = [
        "設計速度80km/hの道路における最小曲線半径は280mです。",
        "横断勾配は道路の排水性能に重要な役割を果たします。",
        "インターチェンジの設計には交通量の予測が必要です。",
        "舗装構造は交通荷重と地盤条件により決定されます。",
        "道路構造令に基づいて設計速度を決定します。"
    ]
    
    test_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    # KeywordSearchEngineを初期化
    print("\n1. KeywordSearchEngineを初期化")
    keyword_engine = KeywordSearchEngine(min_df=1)  # min_dfを1に設定
    
    # 学習
    print("\n2. TF-IDFモデルを学習")
    keyword_engine.fit(test_docs, test_ids)
    print(f"   - 学習完了: {keyword_engine.is_fitted}")
    print(f"   - 文書数: {len(keyword_engine.corpus_texts)}")
    
    if keyword_engine.corpus_vectors is not None:
        print(f"   - TF-IDF行列の形状: {keyword_engine.corpus_vectors.shape}")
        print(f"   - 特徴数: {keyword_engine.corpus_vectors.shape[1]}")
    
    # 検索テスト
    print("\n3. 検索テスト")
    test_queries = ["設計速度", "曲線半径", "道路構造令", "交通"]
    
    for query in test_queries:
        print(f"\n   クエリ: '{query}'")
        results = keyword_engine.search(query, top_k=3, boost_technical_terms=False)
        
        if results:
            print("   結果:")
            for doc_id, score in results:
                doc_idx = test_ids.index(doc_id)
                print(f"     - ID: {doc_id}, スコア: {score:.3f}")
                print(f"       内容: {test_docs[doc_idx][:50]}...")
        else:
            print("   結果なし")
    
    # 語彙を確認
    print("\n4. TF-IDF語彙の確認")
    if hasattr(keyword_engine.vectorizer, 'vocabulary_'):
        vocab = keyword_engine.vectorizer.vocabulary_
        print(f"   語彙サイズ: {len(vocab)}")
        
        # 「設計」「速度」などが含まれているか確認
        important_terms = ["設計", "速度", "道路", "曲線", "半径"]
        print("\n   重要語彙の確認:")
        for term in important_terms:
            if term in vocab:
                print(f"     ✅ '{term}' -> index: {vocab[term]}")
            else:
                print(f"     ❌ '{term}' -> 語彙に含まれていません")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    test_keyword_search()