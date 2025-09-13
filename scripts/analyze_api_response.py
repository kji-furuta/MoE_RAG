#!/usr/bin/env python3
"""
API応答のスコアを分析
"""

import requests
import json

def analyze_api_response():
    """API応答のスコア分析"""
    
    print("=" * 60)
    print("API応答スコア分析")
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
        
        # API呼び出し
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
            
            print(f"検索結果: {len(sources)}件")
            
            for i, source in enumerate(sources[:3], 1):
                print(f"\n結果 {i}:")
                
                # titleを短縮
                title = source.get('metadata', {}).get('title', 'N/A')
                print(f"  Title: {title[:50]}")
                
                # スコアフィールドを全て表示
                print(f"  利用可能なスコアフィールド:")
                for key, value in source.items():
                    if 'score' in key.lower():
                        print(f"    {key}: {value}")
                
                # metadataから追加スコアを探す
                metadata = source.get('metadata', {})
                for key, value in metadata.items():
                    if 'score' in key.lower():
                        print(f"    metadata.{key}: {value}")
                
                # 最上位のスコアを取得
                score = source.get('score', 0)
                vector_score = source.get('vector_score', 0)
                keyword_score = source.get('keyword_score', 0)
                hybrid_score = source.get('hybrid_score', score)
                
                print(f"\n  スコア表示:")
                print(f"    score: {score:.4f}")
                print(f"    vector_score: {vector_score:.4f}")
                print(f"    keyword_score: {keyword_score:.4f}")
                print(f"    hybrid_score: {hybrid_score:.4f}")
                
                # 計算確認
                if vector_score > 0:
                    expected = vector_score * 0.7 + keyword_score * 0.3
                    print(f"    期待値 (0.7*v + 0.3*k): {expected:.4f}")
                    
                    if keyword_score == 0 and hybrid_score > expected:
                        # 技術用語ブーストの可能性
                        boost = (hybrid_score / (vector_score * 0.7)) - 1.0
                        print(f"    💡 推定ブースト: {boost:.3f} ({boost*100:.1f}%)")
        else:
            print(f"❌ API エラー: {response.status_code}")

if __name__ == "__main__":
    analyze_api_response()