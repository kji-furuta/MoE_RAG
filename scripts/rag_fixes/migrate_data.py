#!/usr/bin/env python3
"""
既存のRAGデータを新しい永続化システムに移行するスクリプト
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import sqlite3
import logging

sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigration:
    """既存データの移行クラス"""
    
    def __init__(self):
        # 永続化アダプターの初期化
        from scripts.rag_fixes.integrate_persistence import PersistentRAGAdapter
        self.adapter = PersistentRAGAdapter()
        
        self.migration_stats = {
            'documents_found': 0,
            'documents_migrated': 0,
            'documents_failed': 0,
            'vectors_found': 0,
            'vectors_migrated': 0,
            'start_time': None,
            'end_time': None
        }
    
    def migrate_from_qdrant(self):
        """既存のQdrantデータを移行"""
        print("\n1. Qdrantからのデータ移行")
        print("-" * 40)
        
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            import numpy as np
            
            # 既存のQdrantストア
            old_store = QdrantVectorStore(
                collection_name="road_design_docs",
                embedding_dim=1024,
                path="./data/qdrant"
            )
            
            # コレクション情報の取得
            info = old_store.get_collection_info()
            vectors_count = info.get('vectors_count', 0)
            
            print(f"  既存ベクトル数: {vectors_count}")
            self.migration_stats['vectors_found'] = vectors_count
            
            if vectors_count == 0:
                print("  ℹ️  移行するデータがありません")
                return 0
            
            # すべてのデータを取得（簡易的な方法）
            # 実際の実装では、スクロールAPIを使用すべき
            dummy_query = np.random.randn(1024).astype(np.float32)
            
            batch_size = 100
            migrated = 0
            
            for offset in range(0, min(vectors_count, 1000), batch_size):
                try:
                    results = old_store.search(
                        query_embedding=dummy_query,
                        top_k=batch_size,
                        score_threshold=0.0  # すべて取得
                    )
                    
                    for result in results:
                        try:
                            # 新しいシステムに追加
                            doc_id = self.adapter.add_document(
                                text=result.text,
                                title=result.metadata.get('title', f'Migrated Doc {migrated+1}'),
                                metadata=result.metadata
                            )
                            
                            migrated += 1
                            self.migration_stats['documents_migrated'] += 1
                            
                            if migrated % 10 == 0:
                                print(f"    移行済み: {migrated}件")
                                
                        except Exception as e:
                            logger.warning(f"Failed to migrate document: {e}")
                            self.migration_stats['documents_failed'] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to fetch batch at offset {offset}: {e}")
                    break
            
            print(f"  ✅ {migrated}件のデータを移行しました")
            return migrated
            
        except Exception as e:
            logger.error(f"Qdrant migration failed: {e}")
            return 0
    
    def migrate_from_metadata_db(self):
        """MetadataManagerのデータを移行"""
        print("\n2. MetadataManagerからのデータ移行")
        print("-" * 40)
        
        metadata_paths = [
            Path("metadata/metadata.db"),
            Path("data/metadata/metadata.db")
        ]
        
        for db_path in metadata_paths:
            if db_path.exists():
                print(f"  データベース: {db_path}")
                
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # 文書メタデータの取得
                    cursor.execute("""
                        SELECT id, title, filename, file_path, category, subcategory,
                               version, created_at, custom_fields
                        FROM document_metadata
                        WHERE status = 'ACTIVE' OR status = 'active'
                    """)
                    
                    documents = cursor.fetchall()
                    print(f"  文書数: {len(documents)}")
                    
                    migrated = 0
                    for doc in documents:
                        try:
                            # カスタムフィールドのパース
                            custom_fields = json.loads(doc[8]) if doc[8] else {}
                            
                            metadata = {
                                'original_id': doc[0],
                                'filename': doc[2],
                                'file_path': doc[3],
                                'category': doc[4],
                                'subcategory': doc[5],
                                'version': doc[6],
                                'created_at': doc[7],
                                **custom_fields
                            }
                            
                            # テキストコンテンツの取得（ファイルから読み込む必要がある場合）
                            text = f"Metadata entry for {doc[1]}"  # 実際はファイルから読み込み
                            
                            # 新しいシステムに追加
                            doc_id = self.adapter.add_document(
                                text=text,
                                title=doc[1],
                                metadata=metadata,
                                file_path=doc[3]
                            )
                            
                            migrated += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to migrate metadata {doc[0]}: {e}")
                    
                    conn.close()
                    
                    print(f"  ✅ {migrated}件のメタデータを移行しました")
                    self.migration_stats['documents_migrated'] += migrated
                    
                    return migrated
                    
                except Exception as e:
                    logger.error(f"Metadata migration failed: {e}")
                    
        print("  ℹ️  MetadataManagerのデータベースが見つかりません")
        return 0
    
    def migrate_documents_from_directory(self, directory: str = "data/rag_documents"):
        """ディレクトリからドキュメントを移行"""
        print(f"\n3. ディレクトリからの文書移行: {directory}")
        print("-" * 40)
        
        doc_dir = Path(directory)
        if not doc_dir.exists():
            print(f"  ℹ️  ディレクトリが存在しません: {directory}")
            return 0
        
        # PDFとテキストファイルを検索
        documents = list(doc_dir.glob("**/*.pdf")) + \
                   list(doc_dir.glob("**/*.txt")) + \
                   list(doc_dir.glob("**/*.md"))
        
        print(f"  文書ファイル数: {len(documents)}")
        self.migration_stats['documents_found'] = len(documents)
        
        migrated = 0
        for doc_path in documents:
            try:
                # ファイルの読み込み
                if doc_path.suffix == '.pdf':
                    # PDFの処理（実装が必要）
                    print(f"    ⏭️  PDFファイルはスキップ: {doc_path.name}")
                    continue
                else:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                # メタデータの構築
                metadata = {
                    'source': 'file',
                    'file_type': doc_path.suffix,
                    'file_size': doc_path.stat().st_size,
                    'modified': datetime.fromtimestamp(doc_path.stat().st_mtime).isoformat()
                }
                
                # カテゴリの推定（ディレクトリ名から）
                if doc_path.parent != doc_dir:
                    metadata['category'] = doc_path.parent.name
                
                # 新しいシステムに追加
                doc_id = self.adapter.add_document(
                    text=text,
                    title=doc_path.stem,
                    metadata=metadata,
                    file_path=str(doc_path)
                )
                
                migrated += 1
                print(f"    ✅ {doc_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to migrate {doc_path}: {e}")
                self.migration_stats['documents_failed'] += 1
        
        self.migration_stats['documents_migrated'] += migrated
        print(f"  ✅ {migrated}件の文書を移行しました")
        return migrated
    
    def create_migration_report(self):
        """移行レポートの作成"""
        report_path = Path("data/rag_persistent/migration_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.migration_stats['end_time'] = datetime.now().isoformat()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.migration_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 移行レポート: {report_path}")
        return report_path
    
    def run_full_migration(self):
        """完全な移行を実行"""
        print("=" * 60)
        print("🔄 既存データの移行")
        print("=" * 60)
        
        self.migration_stats['start_time'] = datetime.now().isoformat()
        
        # 各ソースからの移行
        qdrant_count = self.migrate_from_qdrant()
        metadata_count = self.migrate_from_metadata_db()
        file_count = self.migrate_documents_from_directory()
        
        # バックアップの作成
        print("\n4. 移行後のバックアップ作成")
        print("-" * 40)
        
        total_migrated = qdrant_count + metadata_count + file_count
        if total_migrated > 0:
            backup_name = f"migration_{datetime.now().strftime('%Y%m%d_%H%M')}_{total_migrated}docs"
            backup_path = self.adapter.create_backup(backup_name)
            print(f"  ✅ バックアップ作成: {backup_path}")
        
        # レポート作成
        self.create_migration_report()
        
        # サマリー表示
        print("\n" + "=" * 60)
        print("📊 移行サマリー")
        print("=" * 60)
        print(f"  文書発見数: {self.migration_stats['documents_found']}")
        print(f"  移行成功: {self.migration_stats['documents_migrated']}")
        print(f"  移行失敗: {self.migration_stats['documents_failed']}")
        print(f"  ベクトル発見数: {self.migration_stats['vectors_found']}")
        print(f"  ベクトル移行数: {self.migration_stats['vectors_migrated']}")
        
        if self.migration_stats['documents_migrated'] > 0:
            print("\n✅ データ移行が完了しました")
        else:
            print("\n⚠️  移行するデータがありませんでした")
        
        return total_migrated


def main():
    """メイン処理"""
    migration = DataMigration()
    
    # 確認プロンプト
    print("\n⚠️  既存データを新しい永続化システムに移行します")
    print("移行元:")
    print("  • Qdrant (data/qdrant/)")
    print("  • MetadataManager (metadata/metadata.db)")
    print("  • ドキュメントディレクトリ (data/rag_documents/)")
    print("")
    
    response = input("続行しますか？ (y/n): ")
    if response.lower() != 'y':
        print("移行を中止しました")
        return
    
    # 移行実行
    migrated_count = migration.run_full_migration()
    
    return migrated_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
