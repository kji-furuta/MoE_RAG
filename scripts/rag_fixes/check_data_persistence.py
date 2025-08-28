#!/usr/bin/env python3
"""
RAGシステムのデータ永続性を検証するスクリプト
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

def check_qdrant_persistence():
    """Qdrantのデータ永続性を確認"""
    print("\n1. Qdrantデータベースの確認")
    print("-" * 40)
    
    # Qdrantデータパスの確認
    qdrant_paths = [
        Path("data/qdrant"),
        Path("qdrant_data")
    ]
    
    for qdrant_path in qdrant_paths:
        if qdrant_path.exists():
            print(f"\n📁 {qdrant_path}:")
            
            # SQLiteファイルの確認
            sqlite_files = list(qdrant_path.glob("**/*.sqlite"))
            for sqlite_file in sqlite_files:
                size = sqlite_file.stat().st_size
                print(f"  📄 {sqlite_file.name}: {size:,} bytes")
                
                if size < 20000:  # 20KB未満は空とみなす
                    print(f"    ⚠️  ファイルサイズが小さすぎます（データなし）")
                    
                    # SQLiteの内容を確認
                    try:
                        conn = sqlite3.connect(sqlite_file)
                        cursor = conn.cursor()
                        
                        # テーブル一覧を取得
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        
                        print(f"    テーブル数: {len(tables)}")
                        
                        for table in tables:
                            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                            count = cursor.fetchone()[0]
                            print(f"      - {table[0]}: {count} レコード")
                        
                        conn.close()
                    except Exception as e:
                        print(f"    ❌ SQLite読み込みエラー: {e}")
            
            # meta.jsonの確認
            meta_file = qdrant_path / "meta.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    
                print(f"\n  📋 メタデータ:")
                for collection, config in meta.get('collections', {}).items():
                    print(f"    コレクション: {collection}")
                    if 'vectors' in config:
                        print(f"      次元: {config['vectors']['size']}")
                        print(f"      距離: {config['vectors']['distance']}")

def check_metadata_manager():
    """MetadataManagerのデータ永続性を確認"""
    print("\n2. MetadataManagerの確認")
    print("-" * 40)
    
    metadata_paths = [
        Path("metadata/metadata.db"),
        Path("data/metadata/metadata.db")
    ]
    
    found = False
    for db_path in metadata_paths:
        if db_path.exists():
            found = True
            size = db_path.stat().st_size
            print(f"  📄 {db_path}: {size:,} bytes")
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # document_metadataテーブルの確認
                cursor.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name='document_metadata'
                """)
                
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT COUNT(*) FROM document_metadata")
                    doc_count = cursor.fetchone()[0]
                    print(f"    文書数: {doc_count}")
                    
                    if doc_count > 0:
                        cursor.execute("""
                            SELECT id, title, file_path, created_at 
                            FROM document_metadata 
                            LIMIT 5
                        """)
                        docs = cursor.fetchall()
                        print(f"    最近の文書:")
                        for doc in docs:
                            print(f"      - {doc[1][:30]}... ({doc[2]})")
                else:
                    print(f"    ⚠️  document_metadataテーブルが存在しません")
                
                conn.close()
            except Exception as e:
                print(f"    ❌ データベース読み込みエラー: {e}")
    
    if not found:
        print(f"  ⚠️  MetadataManagerのデータベースが見つかりません")

def check_vector_store_data():
    """ベクトルストアの実際のデータを確認"""
    print("\n3. ベクトルストアのデータ確認")
    print("-" * 40)
    
    try:
        from src.rag.indexing.vector_store import QdrantVectorStore
        
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=1024,
            path="./data/qdrant"
        )
        
        # コレクション情報の取得
        info = vector_store.get_collection_info()
        
        print(f"  コレクション状態:")
        print(f"    ステータス: {info.get('status', 'unknown')}")
        print(f"    ベクトル数: {info.get('vectors_count', 0)}")
        print(f"    インデックス済み: {info.get('indexed_vectors_count', 0)}")
        
        vectors_count = info.get('vectors_count', 0)
        
        if vectors_count == 0:
            print(f"    ⚠️  データが保存されていません")
        else:
            print(f"    ✅ {vectors_count}件のデータが保存されています")
            
            # サンプルデータの取得を試みる
            try:
                # ダミーのクエリで検索
                import numpy as np
                dummy_embedding = np.random.randn(1024).astype(np.float32)
                results = vector_store.search(
                    query_embedding=dummy_embedding,
                    top_k=min(3, vectors_count)
                )
                
                print(f"\n    サンプルデータ:")
                for i, result in enumerate(results, 1):
                    text_preview = result.text[:50] if result.text else "N/A"
                    print(f"      {i}. {text_preview}...")
                    print(f"         メタデータ: {list(result.metadata.keys())}")
            except Exception as e:
                print(f"    ⚠️  サンプルデータの取得に失敗: {e}")
                
    except Exception as e:
        print(f"  ❌ ベクトルストアの確認に失敗: {e}")

def check_backup_system():
    """バックアップシステムの確認"""
    print("\n4. バックアップシステムの確認")
    print("-" * 40)
    
    backup_dir = Path("backups")
    
    if backup_dir.exists():
        backups = list(backup_dir.iterdir())
        print(f"  バックアップ数: {len(backups)}")
        
        if backups:
            print(f"  最近のバックアップ:")
            for backup in sorted(backups, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                size = sum(f.stat().st_size for f in backup.rglob('*') if f.is_file())
                print(f"    - {backup.name}: {size:,} bytes ({mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"  ⚠️  バックアップが存在しません")
    else:
        print(f"  ⚠️  バックアップディレクトリが存在しません")

def generate_report():
    """永続性の問題と推奨事項をまとめる"""
    print("\n" + "=" * 60)
    print("📊 データ永続性レポート")
    print("=" * 60)
    
    issues = []
    recommendations = []
    
    # Qdrantデータの確認
    qdrant_path = Path("data/qdrant/collection/road_design_docs/storage.sqlite")
    if qdrant_path.exists():
        size = qdrant_path.stat().st_size
        if size < 20000:
            issues.append("Qdrant SQLiteファイルがほぼ空（データ未保存）")
            recommendations.append("ドキュメントの再インデックスが必要")
    else:
        issues.append("Qdrant SQLiteファイルが存在しない")
        recommendations.append("Qdrantコレクションの初期化が必要")
    
    # MetadataManagerの確認
    metadata_db = Path("metadata/metadata.db")
    if not metadata_db.exists():
        issues.append("MetadataManagerのデータベースが存在しない")
        recommendations.append("MetadataManagerの初期化が必要")
    
    # バックアップの確認
    backup_dir = Path("backups")
    if not backup_dir.exists() or not list(backup_dir.iterdir()):
        issues.append("バックアップが存在しない")
        recommendations.append("定期バックアップシステムの実装が必要")
    
    # レポート出力
    if issues:
        print("\n❌ 発見された問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 推奨事項:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n🔧 修正スクリプト:")
        print("  python scripts/rag_fixes/fix_data_persistence.py")
    else:
        print("\n✅ データ永続性に問題はありません")

def main():
    """メイン処理"""
    print("=" * 60)
    print("🔍 RAGシステム データ永続性チェック")
    print("=" * 60)
    
    check_qdrant_persistence()
    check_metadata_manager()
    check_vector_store_data()
    check_backup_system()
    generate_report()

if __name__ == "__main__":
    main()
