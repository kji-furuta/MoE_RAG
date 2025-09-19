#!/usr/bin/env python3
"""
キーワードスコア修正の確認
"""

import requests
import json

def test_keyword_fix():
    """キーワードスコア修正の確認"""
    
    print("=" * 60)
    print("キーワードスコア修正確認")
    print("=" * 60)
    
    # テストクエリ
    queries = [
        ("設計速度", "vector"),
        ("設計速度", "keyword"),
        ("設計速度", "hybrid"),
        ("道路の横断勾配", "hybrid"),
        ("インターチェンジ", "hybrid")
    ]
    
    for query_text, search_type in queries:
        print(f"\n📝 クエリ: {query_text} ({search_type})")
        print("-" * 40)
        
        try:
            response = requests.post(
                "http://localhost:8050/rag/query",
                json={
                    "query": query_text,
                    "top_k": 3,
                    "search_type": search_type
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get('sources', [])
                
                if sources:
                    for i, source in enumerate(sources[:3], 1):
                        print(f"\n結果 {i}:")
                        
                        # タイトル
                        title = source.get('metadata', {}).get('title', 'N/A')
                        print(f"  Title: {title[:40]}")
                        
                        # スコア
                        v_score = source.get('vector_score', 0)
                        k_score = source.get('keyword_score', 0)
                        h_score = source.get('hybrid_score', source.get('score', 0))
                        
                        print(f"  Vector Score: {v_score:.4f}")
                        print(f"  Keyword Score: {k_score:.4f}")
                        print(f"  Hybrid Score: {h_score:.4f}")
                        
                        # 計算チェック（ハイブリッド検索の場合）
                        if search_type == "hybrid":
                            expected = v_score * 0.7 + k_score * 0.3
                            print(f"  期待値 (0.7*v + 0.3*k): {expected:.4f}")
                            
                            diff = abs(h_score - expected)
                            if diff > 0.001:
                                if k_score == 0:
                                    print(f"  ⚠️ キーワードスコアが0")
                                else:
                                    print(f"  💡 差分: {diff:.4f} (技術用語ブーストの可能性)")
                            else:
                                print(f"  ✅ 計算一致")
                else:
                    print("❌ 検索結果なし")
            else:
                print(f"❌ APIエラー: {response.status_code}")
        except requests.Timeout:
            print("⏰ タイムアウト")
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    print("\n" + "=" * 60)
    print("まとめ")
    print("=" * 60)
    print("キーワードスコアが正しく表示されているか確認してください。")
    print("0でない値が表示されていれば、問題は解決しています。")

if __name__ == "__main__":
    test_keyword_fix()