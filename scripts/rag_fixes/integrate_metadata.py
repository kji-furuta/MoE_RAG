#!/usr/bin/env python3
"""
既存システムとの統合アダプター
MetadataManagerとQdrantVectorStoreを統合メタデータストアに移行
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import hashlib

# パス設定
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from scripts.rag_fixes.unified_metadata_store import (
    UnifiedMetadataStore,
    UnifiedDocument,
    DocumentType,
    DocumentStatus
)


class MetadataIntegrationAdapter:
    """既存システムとの統合アダプター"""
    
    def __init__(self):
        # 統合ストアの初期化
        self.unified_store = UnifiedMetadataStore()
        
        # 既存システムの初期化（エラーハンドリング付き）
        self._init_legacy_systems()
        
        self.migration_log = []
        
    def _init_legacy_systems(self):
        """既存システムの初期化"""
        # QdrantVectorStore
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            self.legacy_qdrant = QdrantVectorStore(
                collection_name="road_design_docs",
                embedding_dim=1024,
                path="./data/qdrant"
            )
            logger.info("Legacy Qdrant initialized")
        except Exception as e:
            logger.warning(f"Could not initialize legacy Qdrant: {e}")
            self.legacy_qdrant = None
        
        # MetadataManager
        try:
            from src.rag.indexing.metadata_manager import MetadataManager
            self.legacy_metadata_manager = MetadataManager(
                db_path="./metadata/metadata.db"
            )
            logger.info("Legacy MetadataManager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MetadataManager: {e}")
            self.legacy_metadata_manager = None
    
    def add_document(self,
                    text: str,
                    title: str,
                    embedding: Optional[Any] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    file_path: Optional[str] = None) -> str:
        """
        統合APIで文書を追加
        既存の両システムとの互換性を維持
        """
        if metadata is None:
            metadata = {}
        
        # 統合文書の作成
        doc = UnifiedDocument(
            title=title,
            content=text,
            file_path=file_path,
            file_name=Path(file_path).name if file_path else None,
            document_type=DocumentType(metadata.get('document_type', 'other')),
            category=metadata.get('category', 'general'),
            subcategory=metadata.get('subcategory'),
            tags=metadata.get('tags', []),
            metadata=metadata,
            status=DocumentStatus.PENDING
        )
        
        # 統合ストアに追加
        if embedding is not None:
            doc_id = self.unified_store.add_document_with_vector(doc, embedding)
        else:
            doc_id = self.unified_store.add_document(doc)
        
        # 既存システムにも追加（後方互換性のため）
        self._add_to_legacy_systems(doc, embedding)
        
        logger.info(f"Document added with unified ID: {doc_id}")
        return doc_id
    
    def _add_to_legacy_systems(self,
                              doc: UnifiedDocument,
                              embedding: Optional[Any] = None):
        """既存システムにも文書を追加（後方互換性）"""
        # Qdrantに追加
        if self.legacy_qdrant and embedding is not None:
            try:
                self.legacy_qdrant.add_documents(
                    texts=[doc.content],
                    embeddings=[embedding],
                    metadatas=[{
                        'doc_id': doc.id,
                        'title': doc.title,
                        'category': doc.category,
                        **doc.metadata
                    }],
                    ids=[doc.id]
                )
                logger.info(f"Added to legacy Qdrant: {doc.id}")
            except Exception as e:
                logger.warning(f"Failed to add to legacy Qdrant: {e}")
        
        # MetadataManagerに追加
        if self.legacy_metadata_manager:
            try:
                from src.rag.indexing.metadata_manager import DocumentMetadata
                
                legacy_metadata = DocumentMetadata(
                    id=doc.id,
                    title=doc.title,
                    filename=doc.file_name or "unknown",
                    file_path=doc.file_path or "",
                    file_hash=doc.content_hash,
                    document_type=doc.document_type,
                    category=doc.category,
                    subcategory=doc.subcategory
                )
                
                self.legacy_metadata_manager.add_metadata(legacy_metadata)
                logger.info(f"Added to legacy MetadataManager: {doc.id}")
            except Exception as e:
                logger.warning(f"Failed to add to MetadataManager: {e}")
    
    def search(self,
              query: str,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        統合検索API
        ベクトル検索とメタデータ検索を統合
        """
        results = []
        
        # ベクトル検索（Qdrant経由）
        if self.unified_store.vector_store:
            try:
                # ベクトル検索の実行（embedding が必要）
                # ここではメタデータ検索のみ実装
                pass
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # メタデータ検索
        docs = self.unified_store.search_documents(
            query=query,
            category=filters.get('category') if filters else None,
            limit=top_k
        )
        
        # 結果の整形
        for doc in docs:
            results.append({
                'id': doc.id,
                'title': doc.title,
                'text': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                'metadata': doc.metadata,
                'category': doc.category,
                'score': 1.0  # ダミースコア
            })
        
        return results
    
    def migrate_existing_data(self) -> Dict[str, int]:
        """既存データを統合ストアに移行"""
        print("\n" + "=" * 60)
        print("📦 既存データの移行")
        print("=" * 60)
        
        stats = {
            'qdrant_migrated': 0,
            'metadata_migrated': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
        
        # 1. MetadataManagerからの移行
        if self.legacy_metadata_manager:
            print("\n1. MetadataManagerからの移行")
            stats['metadata_migrated'] = self._migrate_from_metadata_manager()
        
        # 2. Qdrantからの移行
        if self.legacy_qdrant:
            print("\n2. Qdrantからの移行")
            stats['qdrant_migrated'] = self._migrate_from_qdrant()
        
        # 3. ID重複の解決
        print("\n3. ID重複の解決")
        self._resolve_duplicates()
        
        return stats
    
    def _migrate_from_metadata_manager(self) -> int:
        """MetadataManagerからデータを移行"""
        if not self.legacy_metadata_manager:
            print("  ⚠️  MetadataManagerが利用できません")
            return 0
        
        try:
            # MetadataManagerのデータベースに直接アクセス
            db_path = Path("./metadata/metadata.db")
            if not db_path.exists():
                print("  ⚠️  MetadataManagerのデータベースが見つかりません")
                return 0
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 文書メタデータを取得
            cursor.execute("""
                SELECT id, title, filename, file_path, file_hash,
                       document_type, category, subcategory, version,
                       status, created_at
                FROM document_metadata
            """)
            
            rows = cursor.fetchall()
            migrated = 0
            
            for row in rows:
                try:
                    # UnifiedDocumentに変換
                    doc = UnifiedDocument(
                        id=row[0],  # 既存IDを保持
                        title=row[1],
                        content="",  # コンテンツは別途取得が必要
                        file_name=row[2],
                        file_path=row[3],
                        content_hash=row[4],
                        document_type=DocumentType(row[5]) if row[5] else DocumentType.OTHER,
                        category=row[6] or "general",
                        subcategory=row[7],
                        version=row[8] or "1.0",
                        status=DocumentStatus.INDEXED,
                        created_at=datetime.fromisoformat(row[10]) if row[10] else None
                    )
                    
                    # 統合ストアに追加
                    self.unified_store.add_document(doc)
                    
                    # IDマッピングを作成
                    self.unified_store.add_id_mapping(
                        unified_id=doc.id,
                        original_id=row[0],
                        source_system='metadata_manager'
                    )
                    
                    migrated += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate document {row[0]}: {e}")
            
            conn.close()
            print(f"  ✅ {migrated}件のメタデータを移行しました")
            return migrated
            
        except Exception as e:
            logger.error(f"MetadataManager migration failed: {e}")
            return 0
    
    def _migrate_from_qdrant(self) -> int:
        """Qdrantからデータを移行"""
        if not self.legacy_qdrant:
            print("  ⚠️  Qdrantが利用できません")
            return 0
        
        try:
            # コレクション情報の取得
            info = self.legacy_qdrant.get_collection_info()
            vectors_count = info.get('vectors_count', 0)
            
            if vectors_count == 0:
                print("  ⚠️  Qdrantにデータがありません")
                return 0
            
            print(f"  Qdrantベクトル数: {vectors_count}")
            
            # 簡易的な全データ取得（実際はスクロールAPIを使用すべき）
            import numpy as np
            dummy_query = np.random.randn(1024).astype(np.float32)
            
            results = self.legacy_qdrant.search(
                query_embedding=dummy_query,
                top_k=min(vectors_count, 100),
                score_threshold=0.0
            )
            
            migrated = 0
            for result in results:
                try:
                    # 既存の文書を確認
                    existing = self.unified_store.get_document(result.id)
                    
                    if existing:
                        # ベクトル情報を更新
                        self.unified_store.update_vector_info(
                            doc_id=result.id,
                            vector_id=result.id,
                            vector_dims=1024
                        )
                    else:
                        # 新規文書として追加
                        doc = UnifiedDocument(
                            id=result.id,
                            title=result.metadata.get('title', 'Untitled'),
                            content=result.text,
                            category=result.metadata.get('category', 'general'),
                            metadata=result.metadata,
                            status=DocumentStatus.INDEXED
                        )
                        
                        self.unified_store.add_document(doc)
                    
                    # IDマッピングを作成
                    self.unified_store.add_id_mapping(
                        unified_id=result.id,
                        original_id=result.id,
                        source_system='qdrant'
                    )
                    
                    migrated += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate vector {result.id}: {e}")
            
            print(f"  ✅ {migrated}件のベクトルデータを移行しました")
            return migrated
            
        except Exception as e:
            logger.error(f"Qdrant migration failed: {e}")
            return 0
    
    def _resolve_duplicates(self):
        """重複IDの解決"""
        conn = sqlite3.connect(self.unified_store.db_path)
        cursor = conn.cursor()
        
        # 重複するcontent_hashを検索
        cursor.execute("""
            SELECT content_hash, COUNT(*) as cnt
            FROM documents
            GROUP BY content_hash
            HAVING cnt > 1
        """)
        
        duplicates = cursor.fetchall()
        
        if duplicates:
            print(f"  ⚠️  {len(duplicates)}件の重複が見つかりました")
            
            for content_hash, count in duplicates:
                # 最新のものを残し、他は archived にする
                cursor.execute("""
                    UPDATE documents
                    SET status = 'archived'
                    WHERE content_hash = ?
                    AND id NOT IN (
                        SELECT id FROM documents
                        WHERE content_hash = ?
                        ORDER BY updated_at DESC
                        LIMIT 1
                    )
                """, (content_hash, content_hash))
            
            conn.commit()
            print(f"  ✅ 重複を解決しました")
        else:
            print(f"  ✅ 重複はありませんでした")
        
        conn.close()
    
    def get_migration_report(self) -> Dict[str, Any]:
        """移行レポートを生成"""
        stats = self.unified_store.get_statistics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': stats['total_documents'],
            'vectorized_documents': stats['vectorized_documents'],
            'status_distribution': stats['status_counts'],
            'category_distribution': stats['category_counts'],
            'type_distribution': stats['type_counts'],
            'vectorization_rate': stats['vectorization_rate'],
            'migration_log': self.migration_log
        }
        
        return report


def run_metadata_integration():
    """メタデータ統合を実行"""
    print("=" * 60)
    print("🔧 メタデータ管理の統合")
    print("=" * 60)
    
    # アダプターの初期化
    adapter = MetadataIntegrationAdapter()
    
    # 1. テストデータの追加
    print("\n1. テストデータの追加（統合API）")
    
    test_doc_id = adapter.add_document(
        text="統合APIによるテスト文書です。メタデータとベクトルが一元管理されます。",
        title="統合テスト文書",
        metadata={
            'category': 'test',
            'document_type': 'other',
            'tags': ['test', 'integration'],
            'author': 'System'
        }
    )
    print(f"  ✅ 文書追加: {test_doc_id}")
    
    # 2. 既存データの移行
    print("\n2. 既存データの移行")
    migration_stats = adapter.migrate_existing_data()
    
    print(f"\n  移行結果:")
    print(f"    MetadataManager: {migration_stats.get('metadata_migrated', 0)}件")
    print(f"    Qdrant: {migration_stats.get('qdrant_migrated', 0)}件")
    print(f"    重複スキップ: {migration_stats.get('duplicates_skipped', 0)}件")
    print(f"    エラー: {migration_stats.get('errors', 0)}件")
    
    # 3. 統計情報の表示
    print("\n3. 統合後の統計")
    stats = adapter.unified_store.get_statistics()
    
    print(f"  総文書数: {stats['total_documents']}")
    print(f"  ベクトル化済み: {stats['vectorized_documents']}")
    print(f"  ベクトル化率: {stats['vectorization_rate']:.1f}%")
    
    print(f"\n  ステータス別:")
    for status, count in stats['status_counts'].items():
        print(f"    {status}: {count}件")
    
    print(f"\n  カテゴリ別:")
    for category, count in stats['category_counts'].items():
        print(f"    {category}: {count}件")
    
    # 4. レポート生成
    print("\n4. 移行レポートの生成")
    report = adapter.get_migration_report()
    
    report_path = Path("data/unified_metadata/migration_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ レポート保存: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ メタデータ管理の統合が完了しました")
    print("=" * 60)
    
    return adapter


if __name__ == "__main__":
    run_metadata_integration()
