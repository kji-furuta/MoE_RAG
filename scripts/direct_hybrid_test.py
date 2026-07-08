#!/usr/bin/env python3
"""
ハイブリッド検索を直接テスト
"""

import sys
sys.path.insert(0, '/workspace')

def direct_hybrid_test():
    """ハイブリッド検索を直接テスト"""
    
    print("=" * 60)
    print("ハイブリッド検索直接テスト")
    print("=" * 60)
    
    from src.rag.core.query_engine import RoadDesignQueryEngine
    from src.rag.retrieval.hybrid_search import SearchQuery
    
    # クエリエンジン初期化
    print("\n1. 初期化中...")
    qe = RoadDesignQueryEngine()
    qe.initialize()
    print("✅ 初期化完了")
    
    # ハイブリッド検索実行
    print("\n2. ハイブリッド検索実行...")
    query = SearchQuery(
        text="設計速度",
        search_type="hybrid"
    )
    
    results = qe.hybrid_search.search(query, top_k=3)
    
    print(f"\n3. 検索結果: {len(results)}件")
    for i, result in enumerate(results[:3], 1):
        print(f"\n結果 {i}:")
        print(f"  ID: {result.id[:50]}")
        print(f"  Vector Score: {result.vector_score:.4f}")
        print(f"  Keyword Score: {result.keyword_score:.4f}")
        print(f"  Hybrid Score: {result.hybrid_score:.4f}")
        
        # 計算確認
        expected = result.vector_score * 0.7 + result.keyword_score * 0.3
        print(f"  期待値 (0.7*v + 0.3*k): {expected:.4f}")
        
        diff = abs(result.hybrid_score - expected)
        if diff > 0.001:
            print(f"  💡 差分: {diff:.4f} (技術用語ブーストの可能性)")
        else:
            print(f"  ✅ 計算一致")
    
    # まとめ
    has_keyword = any(r.keyword_score > 0 for r in results)
    print("\n" + "=" * 60)
    if has_keyword:
        print("✅ キーワードスコアが正常に計算されています！")
        print("✅ 問題は解決しました。")
    else:
        print("⚠️ キーワードスコアがまだ0です。")
        print("⚠️ 追加の調査が必要です。")

if __name__ == "__main__":
    direct_hybrid_test()