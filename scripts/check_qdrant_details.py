#!/usr/bin/env python3
"""
Qdrantの詳細情報を確認
"""

import sys
sys.path.insert(0, '/workspace')
import requests
import json

def check_qdrant_details():
    """Qdrantの詳細確認"""
    
    print("=" * 60)
    print("Qdrant詳細確認")
    print("=" * 60)
    
    url = "http://qdrant:6333"
    
    # コレクション情報
    resp = requests.get(f"{url}/collections/road_design_docs")
    if resp.status_code == 200:
        data = resp.json()
        result = data.get("result", {})
        print("\nコレクション情報:")
        print(f"  Status: {result.get('status')}")
        print(f"  Points count: {result.get('points_count')}")
        print(f"  Vectors count: {result.get('vectors_count')}")
        print(f"  Indexed vectors: {result.get('indexed_vectors_count')}")
        print(f"  Segments count: {result.get('segments_count')}")
        
        # Config情報
        config = result.get('config', {})
        params = config.get('params', {})
        vectors = params.get('vectors', {})
        print(f"\nベクトル設定:")
        print(f"  Size: {vectors.get('size')}")
        print(f"  Distance: {vectors.get('distance')}")
    
    # サンプルポイント確認
    resp2 = requests.post(f"{url}/collections/road_design_docs/points/scroll", 
                         json={"limit": 3, "with_payload": True, "with_vector": True})
    if resp2.status_code == 200:
        points = resp2.json().get("result", {}).get("points", [])
        print(f"\nポイント数: {len(points)}")
        
        for i, point in enumerate(points[:3], 1):
            print(f"\nポイント {i}:")
            print(f"  ID: {point.get('id')}")
            
            # ベクトル確認
            vector = point.get('vector')
            if vector:
                if isinstance(vector, list):
                    print(f"  Vector: リスト形式, 長さ={len(vector)}")
                elif isinstance(vector, dict):
                    print(f"  Vector: 辞書形式")
                    for name, vec in vector.items():
                        if vec:
                            print(f"    {name}: 長さ={len(vec)}")
                        else:
                            print(f"    {name}: なし")
            else:
                print(f"  Vector: なし")
            
            # ペイロード確認
            payload = point.get('payload', {})
            if payload:
                print(f"  Payload keys: {list(payload.keys())[:5]}")

    # 検索テスト
    print("\n" + "=" * 60)
    print("検索テスト")
    print("=" * 60)
    
    # ランダムベクトルで検索
    import random
    test_vector = [random.random() for _ in range(1024)]
    
    search_req = {
        "vector": test_vector,
        "limit": 1,
        "with_payload": False
    }
    
    resp3 = requests.post(f"{url}/collections/road_design_docs/points/search", json=search_req)
    if resp3.status_code == 200:
        results = resp3.json().get("result", [])
        if results:
            print(f"✅ 検索成功: {len(results)}件")
            print(f"  スコア: {results[0].get('score', 0):.3f}")
        else:
            print("❌ 検索結果なし")
    else:
        print(f"❌ 検索エラー: {resp3.status_code}")
        print(f"  {resp3.text[:200]}")

if __name__ == "__main__":
    check_qdrant_details()