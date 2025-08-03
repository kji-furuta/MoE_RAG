#!/usr/bin/env python3
"""
RAGデータ永続化テストスクリプト
Dockerコンテナ再起動後もデータが保持されることを確認
"""

import sys
import os
from pathlib import Path

def test_persistence():
    """永続化テスト"""
    print("RAG Data Persistence Test")
    print("=" * 50)
    
    # テスト対象のパス
    paths_to_check = {
        "metadata_db": Path("./metadata/metadata.db"),
        "processed_docs": Path("./outputs/rag_index/processed_documents"),
        "qdrant_data": Path("./qdrant_data"),
    }
    
    # 各パスの存在確認
    for name, path in paths_to_check.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"✅ {name}: 存在します (サイズ: {size:,} bytes)")
            else:
                # ディレクトリの場合はファイル数をカウント
                if path.is_dir():
                    files = list(path.glob("**/*"))
                    file_count = sum(1 for f in files if f.is_file())
                    print(f"✅ {name}: 存在します ({file_count} ファイル)")
        else:
            print(f"❌ {name}: 存在しません - {path}")
    
    print("\n" + "=" * 50)
    
    # メタデータの確認
    if paths_to_check["metadata_db"].exists():
        try:
            import sqlite3
            conn = sqlite3.connect(paths_to_check["metadata_db"])
            cursor = conn.cursor()
            
            # 文書数を確認
            cursor.execute("SELECT COUNT(*) FROM document_metadata")
            doc_count = cursor.fetchone()[0]
            print(f"📄 保存されている文書数: {doc_count}")
            
            if doc_count > 0:
                # 最新の文書を表示
                cursor.execute("""
                    SELECT id, title, created_at 
                    FROM document_metadata 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                print("\n最新の文書:")
                for doc in cursor.fetchall():
                    print(f"  - {doc[1]} (ID: {doc[0][:8]}..., 作成: {doc[2]})")
            
            conn.close()
        except Exception as e:
            print(f"⚠️ メタデータ読み取りエラー: {e}")
    
    # Qdrantデータの確認
    if paths_to_check["qdrant_data"].exists():
        print(f"\n🔍 Qdrantデータディレクトリ:")
        # コレクションディレクトリを探す
        collection_dirs = list(paths_to_check["qdrant_data"].glob("collections/*"))
        for cdir in collection_dirs:
            if cdir.is_dir():
                print(f"  - コレクション: {cdir.name}")
    
    print("\n" + "=" * 50)
    print("テスト完了")
    
    # Dockerボリュームの確認方法を表示
    print("\n📌 Dockerボリュームの確認コマンド:")
    print("  docker volume ls | grep ai-ft")
    print("  docker volume inspect ai-ft-3_rag_metadata")
    print("  docker volume inspect ai-ft-3_rag_processed")
    print("  docker volume inspect ai-ft-3_qdrant_storage")

if __name__ == "__main__":
    test_persistence()