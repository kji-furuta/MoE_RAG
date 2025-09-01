#!/usr/bin/env python3
"""
サンプル文書をRAGシステムにインデックスするスクリプト
"""

import json
import requests
import time
from pathlib import Path

# サンプル文書データ
SAMPLE_DOCS = [
    {
        "title": "道路設計基準 - 設計速度と曲線半径",
        "content": """
        道路の設計速度と最小曲線半径の関係について説明します。
        
        設計速度別の最小曲線半径（標準値）：
        - 設計速度 120km/h: 最小曲線半径 710m
        - 設計速度 100km/h: 最小曲線半径 460m
        - 設計速度 80km/h: 最小曲線半径 280m
        - 設計速度 60km/h: 最小曲線半径 150m
        - 設計速度 50km/h: 最小曲線半径 100m
        - 設計速度 40km/h: 最小曲線半径 60m
        
        これらの値は、道路構造令に基づく標準的な値です。
        実際の設計では、地形条件や交通条件を考慮して決定します。
        """,
        "metadata": {
            "document_type": "design_standard",
            "version": "2024",
            "chapter": "3",
            "section": "3.2",
            "page": "45"
        }
    },
    {
        "title": "道路設計基準 - 横断勾配",
        "content": """
        道路の横断勾配に関する基準を説明します。
        
        標準横断勾配：
        - アスファルトコンクリート舗装: 1.5～2.0%
        - セメントコンクリート舗装: 1.5～2.0%
        - 簡易舗装: 3.0～5.0%
        
        片勾配の最大値：
        - 設計速度 80km/h以上: 最大6%
        - 設計速度 60km/h: 最大8%
        - 設計速度 50km/h以下: 最大10%
        
        積雪寒冷地域では、これらの値を調整する必要があります。
        """,
        "metadata": {
            "document_type": "design_standard",
            "version": "2024",
            "chapter": "4",
            "section": "4.1",
            "page": "67"
        }
    },
    {
        "title": "道路設計基準 - 縦断勾配",
        "content": """
        道路の縦断勾配に関する基準について説明します。
        
        最大縦断勾配：
        - 設計速度 120km/h: 最大3%
        - 設計速度 100km/h: 最大4%
        - 設計速度 80km/h: 最大5%
        - 設計速度 60km/h: 最大7%
        - 設計速度 50km/h: 最大8%
        - 設計速度 40km/h: 最大9%
        
        特例値として、地形の状況等により上記の値を1%増加できます。
        登坂車線を設ける場合は、さらに緩和することができます。
        """,
        "metadata": {
            "document_type": "design_standard",
            "version": "2024",
            "chapter": "5",
            "section": "5.3",
            "page": "89"
        }
    }
]

def index_documents():
    """文書をインデックス"""
    base_url = "http://localhost:8050"
    
    # ヘルスチェック
    try:
        response = requests.get(f"{base_url}/rag/health")
        if response.status_code != 200:
            print("RAGシステムが起動していません")
            return False
    except Exception as e:
        print(f"RAGシステムに接続できません: {e}")
        return False
    
    # 各文書をインデックス
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        print(f"\n文書 {i}/{len(SAMPLE_DOCS)} をインデックス中: {doc['title']}")
        
        # テキストファイルとして保存
        doc_path = Path(f"/tmp/sample_doc_{i}.txt")
        doc_path.write_text(doc['content'], encoding='utf-8')
        
        # メタデータJSONファイルを作成
        metadata_path = Path(f"/tmp/sample_doc_{i}_metadata.json")
        metadata_path.write_text(json.dumps(doc['metadata'], ensure_ascii=False), encoding='utf-8')
        
        # アップロード
        try:
            with open(doc_path, 'rb') as f:
                files = {'file': (f'{doc["title"]}.txt', f, 'text/plain')}
                data = {
                    'document_type': doc['metadata']['document_type'],
                    'metadata': json.dumps(doc['metadata'])
                }
                
                response = requests.post(
                    f"{base_url}/rag/upload-document",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✓ インデックス成功: {result.get('document_id', 'unknown')}")
                else:
                    print(f"  ✗ インデックス失敗: {response.status_code}")
                    print(f"    {response.text}")
                    
        except Exception as e:
            print(f"  ✗ エラー: {e}")
        
        # 一時ファイルを削除
        doc_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        
        # 少し待機
        time.sleep(1)
    
    print("\n\nインデックス完了！")
    
    # 統計情報を表示
    try:
        response = requests.get(f"{base_url}/rag/statistics")
        if response.status_code == 200:
            stats = response.json()
            print("\nRAGシステム統計:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"統計情報の取得に失敗: {e}")
    
    return True

if __name__ == "__main__":
    success = index_documents()
    
    if success:
        print("\n\nテストクエリを実行...")
        
        # テストクエリ
        test_query = {
            "query": "設計速度80km/hの道路の最小曲線半径は？",
            "top_k": 3,
            "search_type": "hybrid",
            "include_sources": True
        }
        
        try:
            response = requests.post(
                "http://localhost:8050/rag/query",
                json=test_query
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nクエリ結果:")
                print(f"質問: {result['query']}")
                print(f"回答: {result['answer'][:500]}...")
                print(f"\n出典情報:")
                for citation in result.get('citations', []):
                    print(f"  - {citation}")
                print(f"\nソース数: {len(result.get('sources', []))}")
                print(f"信頼度: {result.get('confidence_score', 0):.2f}")
            else:
                print(f"クエリ失敗: {response.status_code}")
                
        except Exception as e:
            print(f"テストクエリエラー: {e}")