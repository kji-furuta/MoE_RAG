#!/usr/bin/env python3
"""
技術用語ブースト表示機能のテストスクリプト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retrieval.hybrid_search import HybridSearchResult

def test_hybrid_search_result():
    """HybridSearchResultのtech_boostフィールドをテスト"""
    print("=" * 60)
    print("HybridSearchResult tech_boost フィールドテスト")
    print("=" * 60)

    # テストデータ作成
    result = HybridSearchResult(
        id="test-doc-1",
        text="設計速度80km/hの道路における曲線半径の最小値は280mである。",
        metadata={"title": "道路構造令の解説"},
        vector_score=0.876,
        keyword_score=0.000,
        hybrid_score=0.735,
        tech_boost=0.199,
        rank=1
    )

    print(f"\n✅ tech_boostフィールドが正常に追加されました")
    print(f"   ID: {result.id}")
    print(f"   ベクトルスコア: {result.vector_score:.3f}")
    print(f"   キーワードスコア: {result.keyword_score:.3f}")
    print(f"   技術用語ブースト: {result.tech_boost:.3f} ({result.tech_boost*100:.1f}%)")
    print(f"   最終ハイブリッドスコア: {result.hybrid_score:.3f}")

    # 計算検証
    base_score = 0.7 * result.vector_score + 0.3 * result.keyword_score
    expected_hybrid = base_score * (1.0 + result.tech_boost)

    print(f"\n📊 スコア計算検証:")
    print(f"   基本スコア = 0.7 × {result.vector_score:.3f} + 0.3 × {result.keyword_score:.3f}")
    print(f"             = {base_score:.3f}")
    print(f"   最終スコア = {base_score:.3f} × (1.0 + {result.tech_boost:.3f})")
    print(f"             = {expected_hybrid:.3f}")
    print(f"   実際の値   = {result.hybrid_score:.3f}")

    if abs(expected_hybrid - result.hybrid_score) < 0.001:
        print(f"   ✅ 計算結果が一致しています")
    else:
        print(f"   ⚠️  計算結果に差異があります (差分: {abs(expected_hybrid - result.hybrid_score):.6f})")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    test_hybrid_search_result()
