#!/usr/bin/env python3
"""RAG検索のテストスクリプト"""

import requests
import json

def test_rag_search():
    """RAG検索をテスト"""
    
    # テストクエリ
    queries = [
        "道路",
        "設計",
        "test",
        "JST",
        "document"
    ]
    
    base_url = "http://localhost:8050"
    
    print("=" * 60)
    print("RAG検索テスト")
    print("=" * 60)
    
    for query in queries:
        print(f"\n検索クエリ: '{query}'")
        print("-" * 40)
        
        # 検索リクエスト
        response = requests.post(
            f"{base_url}/rag/query",
            json={
                "query": query,
                "top_k": 3,
                "search_type": "hybrid"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 結果表示
            print(f"応答成功!")
            print(f"ソース数: {len(result.get('sources', []))}")
            print(f"信頼度スコア: {result.get('confidence_score', 0):.2f}")
            
            # ソースがある場合は表示
            if result.get('sources'):
                print("\nソース情報:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {source.get('title', 'N/A')} (スコア: {source.get('score', 0):.3f})")
                    text_preview = source.get('text', '')[:100] if source.get('text') else ''
                    if text_preview:
                        print(f"     テキスト: {text_preview}...")
            else:
                print("  ソースが見つかりませんでした")
                
            # 回答の最初の部分を表示
            answer = result.get('answer', '')
            if "関連する情報が見つかりませんでした" in answer:
                print("\n⚠️  関連情報なし")
            else:
                print(f"\n回答の冒頭: {answer[:200]}...")
        else:
            print(f"エラー: ステータスコード {response.status_code}")
            print(response.text)
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_search()