#!/usr/bin/env python3
"""
統合メタデータ管理システム
MetadataManagerとQdrantのメタデータを一元管理
"""

import os
import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import uuid

# パス設定
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JST タイムゾーン
JST = timezone(timedelta(hours=9))


class DocumentType(Enum):
    """文書タイプ"""
    ROAD_STANDARD = "road_standard"
    DESIGN_GUIDE = "design_guide"
    REGULATION = "regulation"
    TECHNICAL_MANUAL = "technical_manual"
    SPECIFICATION = "specification"
    OTHER = "other"


class DocumentStatus(Enum):
    """文書ステータス"""
    PENDING = "pending"          # 処理待ち
    INDEXING = "indexing"        # インデックス中
    INDEXED = "indexed"          # インデックス完了
    FAILED = "failed"            # 失敗
    ARCHIVED = "archived"        # アーカイブ


@dataclass
class UnifiedDocument:
    """統合文書モデル"""
    # 主キー（システム全体で一意）
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 基本情報
    title: str = ""
    content: str = ""
    content_hash: str = ""
    
    # ファイル情報
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    
    # 分類情報
    document_type: DocumentType = DocumentType.OTHER
    category: str = "general"
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # ベクトル情報
    vector_id: Optional[str] = None  # Qdrant内のID
    vector_dims: Optional[int] = None
    embedding_model: Optional[str] = None
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # バージョン管理
    version: str = "1.0"
    parent_id: Optional[str] = None  # 親文書のID
    
    # ステータス
    status: DocumentStatus = DocumentStatus.PENDING
    
    # タイムスタンプ
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if not self.created_at:
            self.created_at = datetime.now(JST)
        if not self.updated_at:
            self.updated_at = datetime.now(JST)
        if not self.content_hash and self.content:
            self.content_hash = self._calculate_hash(self.content)
    
    def _calculate_hash(self, text: str) -> str:
        """テキストのハッシュ値を計算"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        # Enumを文字列に変換
        data['document_type'] = self.document_type.value
        data['status'] = self.status.value
        # datetimeをISO形式に変換
        for key in ['created_at', 'updated_at', 'indexed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedDocument':
        """辞書から復元"""
        data = data.copy()
        # 文字列をEnumに変換
        if 'document_type' in data:
            data['document_type'] = DocumentType(data['document_type'])
        if 'status' in data:
            data['status'] = DocumentStatus(data['status'])
        # ISO形式をdatetimeに変換
        for key in ['created_at', 'updated_at', 'indexed_at']:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class UnifiedMetadataStore:
    """統合メタデータストア"""
    
    def __init__(self, base_path: str = "./data/unified_metadata"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # データベースパス
        self.db_path = self.base_path / "unified_store.db"
        
        # 初期化
        self._init_database()
        self._init_vector_store()
        
        logger.info(f"UnifiedMetadataStore initialized at {self.base_path}")
    
    def _init_database(self):
        """統合データベースの初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 統合文書テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                -- 主キー
                id TEXT PRIMARY KEY,
                
                -- 基本情報
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                
                -- ファイル情報
                file_path TEXT,
                file_name TEXT,
                file_size INTEGER,
                file_type TEXT,
                
                -- 分類情報
                document_type TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                tags TEXT,  -- JSON配列として保存
                
                -- ベクトル情報
                vector_id TEXT,
                vector_dims INTEGER,
                embedding_model TEXT,
                
                -- メタデータ（JSON）
                metadata TEXT,
                
                -- バージョン管理
                version TEXT DEFAULT '1.0',
                parent_id TEXT,
                
                -- ステータス
                status TEXT DEFAULT 'pending',
                
                -- タイムスタンプ
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed_at TIMESTAMP,
                
                -- 外部キー
                FOREIGN KEY (parent_id) REFERENCES documents(id)
            )
        ''')
        
        # ID マッピングテーブル（互換性のため）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS id_mappings (
                unified_id TEXT PRIMARY KEY,
                original_id TEXT,
                source_system TEXT,  -- 'qdrant', 'metadata_manager', 'legacy'
                mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (unified_id) REFERENCES documents(id)
            )
        ''')
        
        # インデックステーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                action TEXT,  -- 'add', 'update', 'delete', 'reindex'
                status TEXT,  -- 'success', 'failed'
                error_message TEXT,
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')
        
        # インデックス作成
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_category ON documents(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_vector ON documents(vector_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mapping_original ON id_mappings(original_id)')
        
        # トリガー: updated_atの自動更新
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_timestamp
            AFTER UPDATE ON documents
            FOR EACH ROW
            BEGIN
                UPDATE documents SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Unified database initialized")
    
    def _init_vector_store(self):
        """ベクトルストアの初期化"""
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            
            self.vector_store = QdrantVectorStore(
                collection_name="unified_docs",
                embedding_dim=1024,
                path=str(self.base_path / "vectors")
            )
            logger.info("Vector store initialized")
        except ImportError:
            logger.warning("QdrantVectorStore not available")
            self.vector_store = None
    
    def add_document(self, document: UnifiedDocument) -> str:
        """統合文書を追加"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 重複チェック（コンテンツハッシュベース）
            cursor.execute(
                "SELECT id FROM documents WHERE content_hash = ?",
                (document.content_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                logger.warning(f"Document already exists: {existing[0]}")
                return existing[0]
            
            # 文書をデータベースに追加
            cursor.execute('''
                INSERT INTO documents (
                    id, title, content, content_hash,
                    file_path, file_name, file_size, file_type,
                    document_type, category, subcategory, tags,
                    vector_id, vector_dims, embedding_model,
                    metadata, version, parent_id, status,
                    created_at, updated_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id,
                document.title,
                document.content,
                document.content_hash,
                document.file_path,
                document.file_name,
                document.file_size,
                document.file_type,
                document.document_type.value,
                document.category,
                document.subcategory,
                json.dumps(document.tags),
                document.vector_id,
                document.vector_dims,
                document.embedding_model,
                json.dumps(document.metadata),
                document.version,
                document.parent_id,
                document.status.value,
                document.created_at.isoformat() if document.created_at else datetime.now(JST).isoformat(),
                document.updated_at.isoformat() if document.updated_at else datetime.now(JST).isoformat(),
                document.indexed_at.isoformat() if document.indexed_at else None
            ))
            
            # インデックスログに記録
            cursor.execute('''
                INSERT INTO index_log (document_id, action, status)
                VALUES (?, ?, ?)
            ''', (document.id, 'add', 'success'))
            
            conn.commit()
            logger.info(f"Document added: {document.id} - {document.title}")
            return document.id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add document: {e}")
            
            # エラーログに記録
            cursor.execute('''
                INSERT INTO index_log (document_id, action, status, error_message)
                VALUES (?, ?, ?, ?)
            ''', (document.id, 'add', 'failed', str(e)))
            conn.commit()
            
            raise
        finally:
            conn.close()
    
    def add_document_with_vector(self,
                                 document: UnifiedDocument,
                                 embedding: Any) -> str:
        """文書とベクトルを同時に追加"""
        # 文書を追加
        doc_id = self.add_document(document)
        
        # ベクトルストアに追加
        if self.vector_store and embedding is not None:
            try:
                # ベクトルIDは文書IDと同じにする（一対一マッピング）
                self.vector_store.add_documents(
                    texts=[document.content],
                    embeddings=[embedding],
                    metadatas=[{
                        'doc_id': doc_id,
                        'title': document.title,
                        'category': document.category,
                        'document_type': document.document_type.value,
                        **document.metadata
                    }],
                    ids=[doc_id]  # 統一ID使用
                )
                
                # ベクトル情報を更新
                self.update_vector_info(
                    doc_id,
                    vector_id=doc_id,
                    vector_dims=len(embedding) if hasattr(embedding, '__len__') else None
                )
                
                logger.info(f"Vector added for document: {doc_id}")
                
            except Exception as e:
                logger.error(f"Failed to add vector: {e}")
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[UnifiedDocument]:
        """IDで文書を取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_document(row)
        return None
    
    def get_document_by_original_id(self, original_id: str) -> Optional[UnifiedDocument]:
        """元のIDで文書を取得（互換性のため）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # IDマッピングから検索
        cursor.execute('''
            SELECT d.* FROM documents d
            JOIN id_mappings m ON d.id = m.unified_id
            WHERE m.original_id = ?
        ''', (original_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_document(row)
        return None
    
    def search_documents(self,
                        query: Optional[str] = None,
                        category: Optional[str] = None,
                        document_type: Optional[DocumentType] = None,
                        status: Optional[DocumentStatus] = None,
                        limit: int = 100) -> List[UnifiedDocument]:
        """文書を検索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # クエリ構築
        conditions = []
        params = []
        
        if query:
            conditions.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if document_type:
            conditions.append("document_type = ?")
            params.append(document_type.value)
        
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        
        # SQL実行
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM documents
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_document(row) for row in rows]
    
    def update_document(self, document: UnifiedDocument) -> bool:
        """文書を更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            document.updated_at = datetime.now(JST)
            
            cursor.execute('''
                UPDATE documents SET
                    title = ?, content = ?, content_hash = ?,
                    file_path = ?, file_name = ?, file_size = ?, file_type = ?,
                    document_type = ?, category = ?, subcategory = ?, tags = ?,
                    vector_id = ?, vector_dims = ?, embedding_model = ?,
                    metadata = ?, version = ?, parent_id = ?, status = ?,
                    updated_at = ?, indexed_at = ?
                WHERE id = ?
            ''', (
                document.title,
                document.content,
                document.content_hash,
                document.file_path,
                document.file_name,
                document.file_size,
                document.file_type,
                document.document_type.value,
                document.category,
                document.subcategory,
                json.dumps(document.tags),
                document.vector_id,
                document.vector_dims,
                document.embedding_model,
                json.dumps(document.metadata),
                document.version,
                document.parent_id,
                document.status.value,
                document.updated_at.isoformat(),
                document.indexed_at.isoformat() if document.indexed_at else None,
                document.id
            ))
            
            # ログ記録
            cursor.execute('''
                INSERT INTO index_log (document_id, action, status)
                VALUES (?, ?, ?)
            ''', (document.id, 'update', 'success'))
            
            conn.commit()
            logger.info(f"Document updated: {document.id}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update document: {e}")
            return False
        finally:
            conn.close()
    
    def update_vector_info(self,
                          doc_id: str,
                          vector_id: str,
                          vector_dims: Optional[int] = None,
                          embedding_model: Optional[str] = None) -> bool:
        """ベクトル情報を更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE documents SET
                    vector_id = ?,
                    vector_dims = ?,
                    embedding_model = ?,
                    indexed_at = CURRENT_TIMESTAMP,
                    status = ?
                WHERE id = ?
            ''', (
                vector_id,
                vector_dims,
                embedding_model,
                DocumentStatus.INDEXED.value,
                doc_id
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update vector info: {e}")
            return False
        finally:
            conn.close()
    
    def add_id_mapping(self,
                      unified_id: str,
                      original_id: str,
                      source_system: str) -> bool:
        """IDマッピングを追加（互換性のため）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO id_mappings
                (unified_id, original_id, source_system)
                VALUES (?, ?, ?)
            ''', (unified_id, original_id, source_system))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add ID mapping: {e}")
            return False
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 総文書数
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        # ステータス別
        cursor.execute("""
            SELECT status, COUNT(*) FROM documents
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # カテゴリ別
        cursor.execute("""
            SELECT category, COUNT(*) FROM documents
            GROUP BY category
        """)
        category_counts = dict(cursor.fetchall())
        
        # 文書タイプ別
        cursor.execute("""
            SELECT document_type, COUNT(*) FROM documents
            GROUP BY document_type
        """)
        type_counts = dict(cursor.fetchall())
        
        # ベクトル化済み
        cursor.execute("SELECT COUNT(*) FROM documents WHERE vector_id IS NOT NULL")
        vectorized_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': total_docs,
            'status_counts': status_counts,
            'category_counts': category_counts,
            'type_counts': type_counts,
            'vectorized_documents': vectorized_count,
            'vectorization_rate': (vectorized_count / total_docs * 100) if total_docs > 0 else 0
        }
    
    def _row_to_document(self, row: Tuple) -> UnifiedDocument:
        """データベース行を文書オブジェクトに変換"""
        columns = [
            'id', 'title', 'content', 'content_hash',
            'file_path', 'file_name', 'file_size', 'file_type',
            'document_type', 'category', 'subcategory', 'tags',
            'vector_id', 'vector_dims', 'embedding_model',
            'metadata', 'version', 'parent_id', 'status',
            'created_at', 'updated_at', 'indexed_at'
        ]
        
        data = dict(zip(columns, row))
        
        # JSON フィールドをパース
        if data['tags']:
            data['tags'] = json.loads(data['tags'])
        else:
            data['tags'] = []
        
        if data['metadata']:
            data['metadata'] = json.loads(data['metadata'])
        else:
            data['metadata'] = {}
        
        return UnifiedDocument.from_dict(data)
    
    def migrate_from_separate_systems(self) -> Dict[str, int]:
        """既存の分離システムからデータを移行"""
        migration_stats = {
            'qdrant_migrated': 0,
            'metadata_manager_migrated': 0,
            'conflicts_resolved': 0
        }
        
        # 1. MetadataManagerからの移行
        try:
            from src.rag.indexing.metadata_manager import MetadataManager
            
            old_mm = MetadataManager()
            # 実装は MetadataManager の実際のAPIに依存
            logger.info("Migrating from MetadataManager...")
            
        except Exception as e:
            logger.warning(f"Could not migrate from MetadataManager: {e}")
        
        # 2. 既存Qdrantからの移行
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            
            old_qdrant = QdrantVectorStore(
                collection_name="road_design_docs",
                embedding_dim=1024,
                path="./data/qdrant"
            )
            
            logger.info("Migrating from Qdrant...")
            # 実装は実際のQdrant APIに依存
            
        except Exception as e:
            logger.warning(f"Could not migrate from Qdrant: {e}")
        
        return migration_stats


def test_unified_store():
    """統合ストアのテスト"""
    print("=" * 60)
    print("🔧 統合メタデータストアのテスト")
    print("=" * 60)
    
    # ストアの初期化
    store = UnifiedMetadataStore()
    
    # テスト文書の作成
    test_docs = [
        UnifiedDocument(
            title="道路設計基準 第1章",
            content="道路設計における最小曲線半径の基準について説明します。",
            document_type=DocumentType.ROAD_STANDARD,
            category="設計基準",
            subcategory="幾何構造",
            tags=["道路", "設計", "曲線半径"],
            metadata={"author": "国土交通省", "year": 2024}
        ),
        UnifiedDocument(
            title="施工管理マニュアル",
            content="道路工事の施工管理について詳細に解説します。",
            document_type=DocumentType.TECHNICAL_MANUAL,
            category="施工",
            subcategory="管理",
            tags=["施工", "管理", "品質"],
            metadata={"version": "2.0", "department": "施工部"}
        )
    ]
    
    print("\n1. 文書の追加")
    for doc in test_docs:
        doc_id = store.add_document(doc)
        print(f"  ✅ 追加: {doc.title} (ID: {doc_id[:8]}...)")
        
        # IDマッピングの追加（互換性テスト）
        original_id = f"legacy_{doc_id[:8]}"
        store.add_id_mapping(doc_id, original_id, "legacy")
        print(f"    マッピング: {original_id} -> {doc_id[:8]}...")
    
    print("\n2. 文書の検索")
    results = store.search_documents(query="道路", limit=5)
    print(f"  検索結果: {len(results)}件")
    for result in results:
        print(f"    - {result.title} ({result.status.value})")
    
    print("\n3. 統計情報")
    stats = store.get_statistics()
    print(f"  総文書数: {stats['total_documents']}")
    print(f"  ベクトル化済み: {stats['vectorized_documents']}")
    print(f"  ベクトル化率: {stats['vectorization_rate']:.1f}%")
    print(f"  カテゴリ別:")
    for category, count in stats['category_counts'].items():
        print(f"    - {category}: {count}件")
    
    print("\n✅ テスト完了")
    return store


if __name__ == "__main__":
    test_unified_store()
