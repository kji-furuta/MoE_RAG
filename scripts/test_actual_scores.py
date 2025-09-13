#!/usr/bin/env python3
"""
実際のAPIレスポンスのスコアを詳細確認
"""

import requests
import json

def test_actual_scores():
    """実際のスコアの詳細を確認"""
    
    print("=" * 60)
    print("実際のスコア詳細確認")
    print("=" * 60)
    
    # テストクエリ
    queries = [
        "設計速度80km/hの道路の最小曲線半径",
        "道路の横断勾配",
        "インターチェンジの設計"
    ]
    
    for query_text in queries:
        print(f"\n📝 クエリ: {query_text}")
        print("-" * 40)
        
        response = requests.post(
            "http://localhost:8050/rag/query",
            json={
                "query": query_text,
                "top_k": 3,
                "search_type": "hybrid"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            sources = data.get('sources', [])
            
            for i, source in enumerate(sources[:3], 1):
                print(f"\n結果 {i}:")
                
                # スコアを取得
                v_score = source.get('vector_score', 0)
                k_score = source.get('keyword_score', 0) 
                h_score = source.get('hybrid_score', source.get('score', 0))
                
                print(f"  ベクトル: {v_score:.4f}")
                print(f"  キーワード: {k_score:.4f}")
                print(f"  ハイブリッド: {h_score:.4f}")
                
                # 期待値と比較
                expected = v_score * 0.7 + k_score * 0.3
                print(f"  期待値: {expected:.4f}")
                
                # 差分
                diff = abs(h_score - expected)
                if diff > 0.001:
                    print(f"  ⚠️ 不整合: 差分 {diff:.4f}")
                    
                    # 逆算
                    if v_score > 0:
                        implied_k = (h_score - v_score * 0.7) / 0.3
                        print(f"  💡 逆算キーワードスコア: {implied_k:.4f}")
                        
                        # 技術用語ブーストの可能性
                        if implied_k > k_score:
                            boost_factor = (h_score / expected) - 1
                            print(f"  🚀 ブースト係数: {boost_factor:.4f} ({boost_factor*100:.1f}%)")
                else:
                    print(f"  ✅ 計算一致")
        else:
            print(f"❌ エラー: HTTP {response.status_code}")
    
    print("\n" + "=" * 60)
    print("分析結果")
    print("=" * 60)
    print("キーワードスコアは内部で計算されているが、")
    print("APIレスポンスでは0.000として返されています。")
    print("ハイブリッドスコアには反映されているため、")
    print("スコアの伝達経路に問題があります。")

if __name__ == "__main__":
    test_actual_scores()