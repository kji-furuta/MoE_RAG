#!/usr/bin/env python3
"""
キーワード検索エンジンを再初期化するスクリプト
既存のRAGシステムのキーワード検索を有効化します
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.core.query_engine import QueryEngine
from src.rag.config.rag_config import RAGConfig
from loguru import logger
import time

def reinitialize_keyword_search():
    """キーワード検索エンジンを再初期化"""
    
    print("=" * 60)
    print("キーワード検索エンジン再初期化")
    print("=" * 60)
    
    try:
        # RAG設定を読み込み
        print("\n📚 RAG設定を読み込み中...")
        config = RAGConfig.from_yaml("configs/rag_config.yaml")
        
        # QueryEngineを初期化（これによりキーワード検索も初期化される）
        print("\n🔄 QueryEngineを初期化中...")
        query_engine = QueryEngine(config)
        
        # 初期化を実行
        print("\n⚙️ システムを初期化中...")
        start_time = time.time()
        query_engine.initialize()
        
        initialization_time = time.time() - start_time
        print(f"\n✅ 初期化完了 (所要時間: {initialization_time:.2f}秒)")
        
        # キーワード検索の状態を確認
        if hasattr(query_engine, 'hybrid_search') and query_engine.hybrid_search:
            if query_engine.hybrid_search.is_ready:
                print("\n📊 キーワード検索エンジンの状態:")
                print("  ✅ HybridSearchEngine: 準備完了")
                
                # キーワード検索エンジンの詳細を確認
                keyword_engine = query_engine.hybrid_search.keyword_engine
                if hasattr(keyword_engine, 'is_fitted') and keyword_engine.is_fitted:
                    print("  ✅ KeywordSearchEngine: 学習済み")
                    
                    if hasattr(keyword_engine, 'corpus_texts'):
                        corpus_size = len(keyword_engine.corpus_texts)
                        print(f"  📚 コーパスサイズ: {corpus_size} 文書")
                        
                        if corpus_size > 0:
                            # サンプル文書を表示
                            print("\n📝 コーパスのサンプル（最初の3件）:")
                            for i, text in enumerate(keyword_engine.corpus_texts[:3], 1):
                                preview = text[:100] + "..." if len(text) > 100 else text
                                print(f"  {i}. {preview}")
                else:
                    print("  ⚠️ KeywordSearchEngine: 未学習")
            else:
                print("\n⚠️ HybridSearchEngine: 未初期化")
        else:
            print("\n⚠️ HybridSearchEngineが見つかりません")
        
        # テストクエリを実行
        print("\n🧪 テストクエリを実行中...")
        test_query = "設計速度と曲線半径の関係"
        
        result = query_engine.query(
            query_text=test_query,
            top_k=3,
            search_type="hybrid"
        )
        
        print(f"\nクエリ: '{test_query}'")
        print("\n検索結果:")
        
        if result.sources:
            for i, source in enumerate(result.sources[:3], 1):
                print(f"\n  {i}. {source.get('title', '不明')}")
                
                # スコアの表示
                vector_score = source.get('vector_score', source.get('score', 0))
                keyword_score = source.get('keyword_score', 0)
                hybrid_score = source.get('hybrid_score', source.get('score', 0))
                
                print(f"     ベクトルスコア: {vector_score:.3f}")
                print(f"     キーワードスコア: {keyword_score:.3f}")
                print(f"     ハイブリッドスコア: {hybrid_score:.3f}")
                
                # キーワードスコアが0でない場合は成功
                if keyword_score > 0:
                    print("     ✅ キーワード検索が有効です！")
        else:
            print("  検索結果がありません")
            
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("再初期化プロセス完了")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = reinitialize_keyword_search()
    sys.exit(0 if success else 1)