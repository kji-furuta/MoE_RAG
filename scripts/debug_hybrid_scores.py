#!/usr/bin/env python3
"""
ハイブリッドスコアの計算を詳細にデバッグ
"""

import requests
import json

def test_hybrid_scores():
    """ハイブリッドスコアの内訳を詳細に確認"""
    
    url = "http://localhost:8050/rag/query"
    
    test_query = {
        "query": "設計速度80km/hの道路の最小曲線半径は？",
        "top_k": 3,
        "search_type": "hybrid"
    }
    
    print("=" * 60)
    print("ハイブリッドスコア計算デバッグ")
    print("=" * 60)
    print(f"\n📝 クエリ: {test_query['query']}")
    
    try:
        response = requests.post(
            url,
            json=test_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            sources = data.get('sources', [])
            if sources:
                print(f"\n検索結果数: {len(sources)}")
                print("\n詳細なスコア分析:")
                print("-" * 60)
                
                for i, source in enumerate(sources[:3], 1):
                    print(f"\n結果 {i}:")
                    title = source.get('title', source.get('metadata', {}).get('title', '不明'))
                    print(f"  タイトル: {title[:50]}")
                    
                    # すべてのスコア関連フィールドを表示
                    print(f"  利用可能なフィールド: {list(source.keys())}")
                    
                    # スコアの取得と計算検証
                    vector_score = source.get('vector_score', source.get('score', 0))
                    keyword_score = source.get('keyword_score', 0)
                    hybrid_score = source.get('hybrid_score', source.get('score', 0))
                    
                    print(f"  ベクトルスコア: {vector_score:.3f}")
                    print(f"  キーワードスコア: {keyword_score:.3f}")
                    print(f"  ハイブリッドスコア: {hybrid_score:.3f}")
                    
                    # 期待される計算
                    expected_hybrid = vector_score * 0.7 + keyword_score * 0.3
                    print(f"  期待値 (0.7*V + 0.3*K): {expected_hybrid:.3f}")
                    
                    # 差分チェック
                    diff = abs(hybrid_score - expected_hybrid)
                    if diff > 0.001:
                        print(f"  ⚠️ 不整合検出! 差分: {diff:.3f}")
                        
                        # 逆算
                        if hybrid_score > 0 and vector_score > 0:
                            implied_keyword = (hybrid_score - vector_score * 0.7) / 0.3
                            print(f"  逆算されるキーワードスコア: {implied_keyword:.3f}")
                    else:
                        print(f"  ✅ スコア計算は正しい")
                        
        else:
            print(f"❌ エラー: HTTPステータス {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    test_hybrid_scores()