#!/usr/bin/env python3
"""
文書アップロードとチェックボックス選択の統合テストスクリプト
"""

import requests
import json
import time
from pathlib import Path

# APIエンドポイント
BASE_URL = "http://localhost:8050"

def test_document_list():
    """文書リストを取得"""
    print("\n=== 文書リストを取得 ===")
    response = requests.get(f"{BASE_URL}/rag/documents")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 文書数: {len(data.get('documents', []))}")
        
        # 文書情報を表示
        for doc in data.get('documents', [])[:3]:  # 最初の3件のみ表示
            print(f"  - ID: {doc.get('id')}")
            print(f"    Title: {doc.get('title')}")
            print(f"    Filename: {doc.get('filename')}")
            print(f"    Category: {doc.get('category')}")
        
        return data.get('documents', [])
    else:
        print(f"✗ エラー: {response.status_code}")
        print(response.text)
        return []

def test_search_with_document_filter(documents):
    """文書フィルタ付き検索をテスト"""
    print("\n=== 文書フィルタ付き検索テスト ===")
    
    if not documents:
        print("✗ テスト可能な文書がありません")
        return
    
    # 最初の文書IDを使って検索
    selected_doc = documents[0]
    doc_id = selected_doc.get('id')
    doc_filename = selected_doc.get('filename')
    
    print(f"選択文書: {doc_filename} (ID: {doc_id})")
    
    # 検索リクエスト
    search_request = {
        "query": "道路設計の基準について",
        "search_type": "hybrid",
        "top_k": 5,
        "include_sources": True,
        "document_ids": [doc_id]  # 特定の文書のみを検索対象にする
    }
    
    print(f"検索リクエスト: {json.dumps(search_request, ensure_ascii=False, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/rag/query",
        headers={"Content-Type": "application/json"},
        json=search_request
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 検索成功")
        print(f"  回答: {data.get('answer', '')[:200]}...")
        
        # 引用元を確認
        citations = data.get('citations', [])
        print(f"  引用数: {len(citations)}")
        
        # 引用元のファイル名を確認
        for citation in citations[:3]:
            metadata = citation.get('metadata', {})
            print(f"    - 引用元: {metadata.get('filename', 'Unknown')}")
            
        # すべての引用が選択した文書からのものか確認
        all_from_selected = all(
            citation.get('metadata', {}).get('filename') == doc_filename
            for citation in citations
        )
        
        if all_from_selected:
            print(f"✓ すべての引用が選択した文書 ({doc_filename}) からのものです")
        else:
            print(f"⚠ 一部の引用が選択した文書以外からのものです")
            
    else:
        print(f"✗ 検索エラー: {response.status_code}")
        print(response.text)

def test_search_without_filter():
    """フィルタなし検索をテスト"""
    print("\n=== フィルタなし検索テスト ===")
    
    search_request = {
        "query": "道路設計の基準について",
        "search_type": "hybrid",
        "top_k": 5,
        "include_sources": True
        # document_idsを指定しない = すべての文書から検索
    }
    
    response = requests.post(
        f"{BASE_URL}/rag/query",
        headers={"Content-Type": "application/json"},
        json=search_request
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 検索成功")
        
        # 引用元の文書を集計
        citations = data.get('citations', [])
        unique_docs = set()
        for citation in citations:
            metadata = citation.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            unique_docs.add(filename)
        
        print(f"  引用元文書数: {len(unique_docs)}")
        for doc in list(unique_docs)[:5]:  # 最初の5件のみ表示
            print(f"    - {doc}")
            
    else:
        print(f"✗ 検索エラー: {response.status_code}")
        print(response.text)

def main():
    """メインテスト関数"""
    print("=" * 60)
    print("文書アップロードとチェックボックス選択の統合テスト")
    print("=" * 60)
    
    # 1. 文書リストを取得
    documents = test_document_list()
    
    # 2. 文書フィルタ付き検索をテスト
    if documents:
        test_search_with_document_filter(documents)
    
    # 3. フィルタなし検索をテスト
    test_search_without_filter()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    main()