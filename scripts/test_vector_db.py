#!/usr/bin/env python3
"""
ベクトルデータベース(Qdrant)の性能テスト
"""

import sys
sys.path.insert(0, '/workspace')

import requests
import json
from loguru import logger

def test_qdrant_health():
    """Qdrantサーバーの健全性を確認"""
    print("=" * 60)
    print("1. Qdrantサーバー健全性チェック")
    print("=" * 60)
    
    try:
        # Qdrantのヘルスチェック
        response = requests.get("http://qdrant:6333/")
        if response.status_code == 200:
            print("✅ Qdrantサーバーは正常に稼働しています")
            data = response.json()
            print(f"   バージョン: {data.get('version', 'unknown')}")
        else:
            print(f"❌ Qdrantサーバーエラー: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant接続エラー: {e}")
        return False
    
    return True

def test_collection_info():
    """コレクション情報を確認"""
    print("\n" + "=" * 60)
    print("2. コレクション情報の確認")
    print("=" * 60)
    
    try:
        # コレクションリストを取得
        response = requests.get("http://qdrant:6333/collections")
        if response.status_code == 200:
            collections = response.json()
            print(f"✅ コレクション数: {len(collections.get('result', {}).get('collections', []))}")
            
            # road_design_docsコレクションの詳細情報
            response = requests.get("http://qdrant:6333/collections/road_design_docs")
            if response.status_code == 200:
                info = response.json()
                result = info.get('result', {})
                print(f"\n📦 road_design_docsコレクション:")
                print(f"   ベクトル数: {result.get('vectors_count', 0)}")
                print(f"   インデックス済みベクトル数: {result.get('indexed_vectors_count', 0)}")
                print(f"   ステータス: {result.get('status', 'unknown')}")
                
                config = result.get('config', {})
                params = config.get('params', {})
                print(f"   ベクトル次元数: {params.get('vectors', {}).get('size', 'unknown')}")
                print(f"   距離メトリック: {params.get('vectors', {}).get('distance', 'unknown')}")
                
                return result.get('vectors_count', 0)
            else:
                print(f"⚠️ road_design_docsコレクションが見つかりません")
                return 0
        else:
            print(f"❌ コレクション情報取得エラー: HTTP {response.status_code}")
            return 0
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 0

def test_vector_search():
    """ベクトル検索のテスト"""
    print("\n" + "=" * 60)
    print("3. ベクトル検索テスト")
    print("=" * 60)
    
    try:
        # ダミーベクトルで検索（1024次元のゼロベクトル）
        dummy_vector = [0.1] * 1024  # 少し値を入れたベクトル
        
        search_request = {
            "vector": dummy_vector,
            "limit": 5,
            "with_payload": True,
            "with_vector": False
        }
        
        response = requests.post(
            "http://qdrant:6333/collections/road_design_docs/points/search",
            json=search_request
        )
        
        if response.status_code == 200:
            results = response.json()
            points = results.get('result', [])
            print(f"✅ 検索成功: {len(points)}件の結果")
            
            for i, point in enumerate(points[:3], 1):
                print(f"\n結果 {i}:")
                print(f"   ID: {point.get('id')}")
                print(f"   スコア: {point.get('score', 0):.3f}")
                payload = point.get('payload', {})
                print(f"   タイトル: {payload.get('title', payload.get('metadata', {}).get('title', '不明'))[:50]}")
                text = payload.get('text', '')
                if text:
                    print(f"   テキスト: {text[:100]}...")
                else:
                    print(f"   テキスト: なし")
            
            return len(points) > 0
        else:
            print(f"❌ 検索エラー: HTTP {response.status_code}")
            print(f"   レスポンス: {response.text}")
            return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_rag_vector_search():
    """RAG APIを通じたベクトル検索テスト"""
    print("\n" + "=" * 60)
    print("4. RAG API経由のベクトル検索テスト")
    print("=" * 60)
    
    try:
        test_query = {
            "query": "設計速度",
            "top_k": 3,
            "search_type": "vector"  # ベクトル検索のみ
        }
        
        response = requests.post(
            "http://localhost:8050/rag/query",
            json=test_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            sources = data.get('sources', [])
            print(f"✅ RAG検索成功: {len(sources)}件の結果")
            
            for i, source in enumerate(sources[:3], 1):
                print(f"\n結果 {i}:")
                title = source.get('title', source.get('metadata', {}).get('title', '不明'))
                print(f"   タイトル: {title[:50]}")
                score = source.get('vector_score', source.get('score', 0))
                print(f"   ベクトルスコア: {score:.3f}")
                text = source.get('text', '')
                if text:
                    print(f"   テキスト: {text[:100]}...")
                    
            return len(sources) > 0
        else:
            print(f"❌ RAG検索エラー: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_document_retrieval():
    """文書の取得可能性を確認"""
    print("\n" + "=" * 60)
    print("5. 文書取得テスト（Scroll API）")
    print("=" * 60)
    
    try:
        # スクロール検索で文書を取得
        scroll_request = {
            "limit": 10,
            "with_payload": True,
            "with_vector": False
        }
        
        response = requests.post(
            "http://qdrant:6333/collections/road_design_docs/points/scroll",
            json=scroll_request
        )
        
        if response.status_code == 200:
            data = response.json()
            points = data.get('result', {}).get('points', [])
            next_page_offset = data.get('result', {}).get('next_page_offset')
            
            print(f"✅ スクロール検索成功: {len(points)}件の文書")
            print(f"   次ページオフセット: {next_page_offset}")
            
            doc_ids = []
            for point in points[:5]:
                payload = point.get('payload', {})
                doc_id = payload.get('original_id', payload.get('doc_id', point.get('id')))
                doc_ids.append(doc_id)
                title = payload.get('title', payload.get('metadata', {}).get('title', '不明'))
                print(f"   - ID: {doc_id}, タイトル: {title[:30]}")
                
            return len(points) > 0
        else:
            print(f"❌ スクロール検索エラー: HTTP {response.status_code}")
            print(f"   レスポンス: {response.text}")
            return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def main():
    print("=" * 60)
    print("ベクトルデータベース性能診断")
    print("=" * 60)
    
    # 各テストを実行
    results = {}
    
    # 1. Qdrantサーバー健全性
    results['qdrant_health'] = test_qdrant_health()
    
    # 2. コレクション情報
    vector_count = test_collection_info()
    results['collection'] = vector_count > 0
    
    # 3. ベクトル検索
    if vector_count > 0:
        results['vector_search'] = test_vector_search()
    else:
        print("\n⚠️ ベクトルが存在しないため、検索テストをスキップ")
        results['vector_search'] = False
    
    # 4. RAG API経由の検索
    results['rag_search'] = test_rag_vector_search()
    
    # 5. 文書取得
    if vector_count > 0:
        results['document_retrieval'] = test_document_retrieval()
    else:
        results['document_retrieval'] = False
    
    # 診断結果サマリー
    print("\n" + "=" * 60)
    print("診断結果サマリー")
    print("=" * 60)
    
    all_pass = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'正常' if passed else '異常'}")
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ベクトルデータベースは正常に動作しています")
    elif results['qdrant_health'] and not results['collection']:
        print("⚠️ Qdrantは稼働していますが、文書がインデックスされていません")
        print("→ 文書をアップロードしてください")
    else:
        print("❌ ベクトルデータベースに問題があります")
        print("→ 上記のエラーメッセージを確認してください")
    print("=" * 60)

if __name__ == "__main__":
    main()