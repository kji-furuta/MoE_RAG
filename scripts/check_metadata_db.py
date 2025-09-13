#!/usr/bin/env python3
"""
メタデータDBの内容を確認
"""

import sys
sys.path.insert(0, '/workspace')

import sqlite3
import json

def check_metadata_db():
    """メタデータDBの内容を確認"""
    
    print("=" * 60)
    print("メタデータDBの確認")
    print("=" * 60)
    
    db_path = "/workspace/metadata/metadata.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\nテーブル数: {len(tables)}")
        for table in tables:
            table_name = table[0]
            print(f"\nテーブル: {table_name}")
            
            # レコード数を確認
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  レコード数: {count}")
            
            if count > 0 and table_name == 'documents':
                # 最初の3件を表示
                cursor.execute(f"SELECT id, title, created_at FROM {table_name} LIMIT 3")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"  - ID: {row[0]}, タイトル: {row[1][:50]}, 作成日: {row[2]}")
            
            if count > 0 and table_name == 'chunks':
                # チャンク数を確認
                cursor.execute(f"SELECT COUNT(*), document_id FROM {table_name} GROUP BY document_id LIMIT 3")
                chunk_counts = cursor.fetchall()
                for cc in chunk_counts:
                    print(f"  - 文書ID {cc[1]}: {cc[0]}チャンク")
        
        conn.close()
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_metadata_db()