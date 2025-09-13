#!/usr/bin/env python3
"""
スコア表示のテストスクリプト
ベクトルスコアとキーワードスコアが正しく表示されることを確認
"""

import requests
import json
import time

def test_rag_query():
    """RAGクエリをテストしてスコアを確認"""
    
    # APIエンドポイント
    url = "http://localhost:8050/rag/query"
    
    # テストクエリ
    test_query = {
        "query": "設計速度80km/hの道路の最小曲線半径は？",
        "top_k": 5,
        "search_type": "hybrid"
    }
    
    print("=" * 60)
    print("RAGスコア表示テスト")
    print("=" * 60)
    print(f"\n📝 クエリ: {test_query['query']}")
    print(f"🔍 検索タイプ: {test_query['search_type']}")
    print(f"📊 取得件数: {test_query['top_k']}")
    
    try:
        # APIリクエスト送信
        print("\n⏳ APIリクエスト送信中...")
        response = requests.post(
            url,
            json=test_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ レスポンス受信成功")
            
            # 回答の表示
            print("\n📖 回答:")
            print("-" * 40)
            print(data.get('answer', 'No answer'))
            
            # スコアの確認
            print("\n📊 検索結果詳細:")
            print("-" * 40)
            
            sources = data.get('sources', [])
            if sources:
                print(f"検索結果数: {len(sources)}")
                print("\n{:<30} {:>12} {:>12} {:>12}".format(
                    "タイトル", "ベクトル", "キーワード", "ハイブリッド"
                ))
                print("-" * 70)
                
                for i, source in enumerate(sources[:5], 1):
                    title = source.get('title', source.get('metadata', {}).get('title', '不明'))
                    if len(title) > 27:
                        title = title[:27] + "..."
                    
                    # スコアの取得
                    vector_score = source.get('vector_score', source.get('score', 0))
                    keyword_score = source.get('keyword_score', 0)
                    hybrid_score = source.get('hybrid_score', source.get('score', 0))
                    
                    print("{:<30} {:>12.3f} {:>12.3f} {:>12.3f}".format(
                        title, vector_score, keyword_score, hybrid_score
                    ))
                    
                    # スコアタイプの確認
                    if i == 1:
                        print("\n📝 最初の結果の詳細:")
                        if 'vector_score' in source:
                            print(f"  ✅ vector_score: {source['vector_score']:.3f}")
                        else:
                            print(f"  ⚠️  vector_score: 存在しません (scoreを使用: {source.get('score', 0):.3f})")
                        
                        if 'keyword_score' in source:
                            print(f"  ✅ keyword_score: {source['keyword_score']:.3f}")
                        else:
                            print(f"  ⚠️  keyword_score: 存在しません")
                        
                        if 'hybrid_score' in source:
                            print(f"  ✅ hybrid_score: {source['hybrid_score']:.3f}")
                        else:
                            print(f"  ⚠️  hybrid_score: 存在しません")
                        print()
            else:
                print("⚠️ 検索結果がありません")
            
            # 引用の確認
            citations = data.get('citations', [])
            if citations:
                print(f"\n📚 引用数: {len(citations)}")
                cite = citations[0]
                print(f"最初の引用のスコア情報:")
                if 'vector_score' in cite:
                    print(f"  ✅ vector_score: {cite['vector_score']:.3f}")
                if 'keyword_score' in cite:
                    print(f"  ✅ keyword_score: {cite['keyword_score']:.3f}")
                if 'score' in cite:
                    print(f"  ℹ️  score: {cite['score']:.3f}")
                    
        else:
            print(f"\n❌ エラー: HTTPステータス {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 接続エラー: サーバーが起動していることを確認してください")
        print("起動コマンド: docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_query()