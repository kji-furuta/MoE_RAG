"""
メタデータ管理モジュール
文書のバージョン管理、分類、検索用メタデータの管理
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from loguru import logger
from enum import Enum


class DocumentType(Enum):
    """文書タイプ"""
    ROAD_STANDARD = "road_standard"  # 道路構造令
    DESIGN_GUIDE = "design_guide"    # 設計指針
    REGULATION = "regulation"        # 規則・規定
    TECHNICAL_MANUAL = "technical_manual"  # 技術マニュアル
    SPECIFICATION = "specification"  # 仕様書
    OTHER = "other"


class DocumentStatus(Enum):
    """文書ステータス"""
    ACTIVE = "active"      # 有効
    SUPERSEDED = "superseded"  # 廃止
    DRAFT = "draft"        # 草案
    ARCHIVED = "archived"  # アーカイブ


@dataclass
class DocumentMetadata:
    """文書メタデータ"""
    id: str
    title: str
    filename: str
    file_path: str
    file_hash: str
    
    # 分類情報
    document_type: DocumentType
    category: str
    subcategory: Optional[str] = None
    
    # バージョン情報
    version: str = "1.0"
    revision_date: Optional[str] = None
    effective_date: Optional[str] = None
    supersedes: Optional[str] = None  # 前バージョンのID
    
    # ステータス
    status: DocumentStatus = DocumentStatus.ACTIVE
    
    # 発行情報
    publisher: Optional[str] = None
    issued_by: Optional[str] = None
    approval_authority: Optional[str] = None
    
    # 技術情報
    applicable_standards: List[str] = None
    related_documents: List[str] = None
    keywords: List[str] = None
    
    # 処理情報
    processing_timestamp: Optional[str] = None
    processing_version: str = "1.0"
    
    # 統計情報
    page_count: Optional[int] = None
    section_count: Optional[int] = None
    table_count: Optional[int] = None
    figure_count: Optional[int] = None
    
    # カスタムメタデータ
    custom_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.applicable_standards is None:
            self.applicable_standards = []
        if self.related_documents is None:
            self.related_documents = []
        if self.keywords is None:
            self.keywords = []
        if self.custom_fields is None:
            self.custom_fields = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data['document_type'] = self.document_type.value
        data['status'] = self.status.value
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """辞書から復元"""
        data = data.copy()
        data['document_type'] = DocumentType(data['document_type'])
        data['status'] = DocumentStatus(data['status'])
        return cls(**data)


class MetadataManager:
    """メタデータ管理クラス"""
    
    def __init__(self, db_path: str = "./metadata/metadata.db"):
        """
        Args:
            db_path: SQLiteデータベースファイルのパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データベースの初期化
        self._init_database()
        
        logger.info(f"MetadataManager initialized with database: {self.db_path}")
        
    def _init_database(self):
        """データベーステーブルの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # メインのメタデータテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    version TEXT NOT NULL,
                    revision_date TEXT,
                    effective_date TEXT,
                    supersedes TEXT,
                    status TEXT NOT NULL,
                    publisher TEXT,
                    issued_by TEXT,
                    approval_authority TEXT,
                    processing_timestamp TEXT,
                    processing_version TEXT,
                    page_count INTEGER,
                    section_count INTEGER,
                    table_count INTEGER,
                    figure_count INTEGER,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 関連文書テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_doc_id TEXT NOT NULL,
                    target_doc_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_doc_id) REFERENCES document_metadata (id),
                    FOREIGN KEY (target_doc_id) REFERENCES document_metadata (id),
                    UNIQUE(source_doc_id, target_doc_id, relation_type)
                )
            ''')
            
            # キーワードテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES document_metadata (id),
                    UNIQUE(doc_id, keyword)
                )
            ''')
            
            # バージョン履歴テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    change_description TEXT,
                    changed_by TEXT,
                    change_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES document_metadata (id)
                )
            ''')
            
            # インデックスの作成
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_type ON document_metadata (document_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON document_metadata (category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON document_metadata (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_version ON document_metadata (version)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON document_metadata (file_hash)')
            
            conn.commit()
            
    def add_document(self, metadata: DocumentMetadata) -> bool:
        """文書メタデータを追加"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # メタデータをJSONとして保存
                metadata_json = json.dumps({
                    'applicable_standards': metadata.applicable_standards,
                    'custom_fields': metadata.custom_fields
                }, ensure_ascii=False)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO document_metadata (
                        id, title, filename, file_path, file_hash,
                        document_type, category, subcategory, version,
                        revision_date, effective_date, supersedes, status,
                        publisher, issued_by, approval_authority,
                        processing_timestamp, processing_version,
                        page_count, section_count, table_count, figure_count,
                        metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.id, metadata.title, metadata.filename, metadata.file_path,
                    metadata.file_hash, metadata.document_type.value, metadata.category,
                    metadata.subcategory, metadata.version, metadata.revision_date,
                    metadata.effective_date, metadata.supersedes, metadata.status.value,
                    metadata.publisher, metadata.issued_by, metadata.approval_authority,
                    metadata.processing_timestamp, metadata.processing_version,
                    metadata.page_count, metadata.section_count, metadata.table_count,
                    metadata.figure_count, metadata_json, datetime.now().isoformat()
                ))
                
                # 関連文書を追加
                if metadata.related_documents:
                    for related_id in metadata.related_documents:
                        cursor.execute('''
                            INSERT OR IGNORE INTO document_relations 
                            (source_doc_id, target_doc_id, relation_type)
                            VALUES (?, ?, ?)
                        ''', (metadata.id, related_id, 'related'))
                        
                # キーワードを追加
                if metadata.keywords:
                    for keyword in metadata.keywords:
                        cursor.execute('''
                            INSERT OR IGNORE INTO document_keywords (doc_id, keyword)
                            VALUES (?, ?)
                        ''', (metadata.id, keyword))
                        
                # バージョン履歴を記録
                cursor.execute('''
                    INSERT INTO version_history 
                    (doc_id, version, change_type, change_description)
                    VALUES (?, ?, ?, ?)
                ''', (metadata.id, metadata.version, 'create', 'Document added to system'))
                
                conn.commit()
                
            logger.info(f"Document metadata added: {metadata.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document metadata: {e}")
            return False
            
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """文書メタデータを取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM document_metadata WHERE id = ?
                ''', (doc_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                # 関連文書を取得
                cursor.execute('''
                    SELECT target_doc_id FROM document_relations 
                    WHERE source_doc_id = ? AND relation_type = 'related'
                ''', (doc_id,))
                related_docs = [r[0] for r in cursor.fetchall()]
                
                # キーワードを取得
                cursor.execute('''
                    SELECT keyword FROM document_keywords WHERE doc_id = ?
                ''', (doc_id,))
                keywords = [r[0] for r in cursor.fetchall()]
                
                # メタデータオブジェクトを構築
                metadata_json = json.loads(row['metadata_json'] or '{}')
                
                metadata = DocumentMetadata(
                    id=row['id'],
                    title=row['title'],
                    filename=row['filename'],
                    file_path=row['file_path'],
                    file_hash=row['file_hash'],
                    document_type=DocumentType(row['document_type']),
                    category=row['category'],
                    subcategory=row['subcategory'],
                    version=row['version'],
                    revision_date=row['revision_date'],
                    effective_date=row['effective_date'],
                    supersedes=row['supersedes'],
                    status=DocumentStatus(row['status']),
                    publisher=row['publisher'],
                    issued_by=row['issued_by'],
                    approval_authority=row['approval_authority'],
                    processing_timestamp=row['processing_timestamp'],
                    processing_version=row['processing_version'],
                    page_count=row['page_count'] if row['page_count'] is not None else 0,
                    section_count=row['section_count'] if row['section_count'] is not None else 0,
                    table_count=row['table_count'] if row['table_count'] is not None else 0,
                    figure_count=row['figure_count'] if row['figure_count'] is not None else 0,
                    applicable_standards=metadata_json.get('applicable_standards', []),
                    related_documents=related_docs,
                    keywords=keywords,
                    custom_fields=metadata_json.get('custom_fields', {})
                )
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return None
            
    def search_documents(self,
                        query: Optional[str] = None,
                        document_type: Optional[DocumentType] = None,
                        category: Optional[str] = None,
                        status: Optional[DocumentStatus] = None,
                        keywords: Optional[List[str]] = None,
                        version: Optional[str] = None) -> List[DocumentMetadata]:
        """文書を検索"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # ベースクエリ
                base_query = "SELECT DISTINCT dm.* FROM document_metadata dm"
                conditions = []
                params = []
                
                # キーワード検索の場合はJOIN
                if keywords:
                    base_query += " JOIN document_keywords dk ON dm.id = dk.doc_id"
                    
                # 条件を構築
                if query:
                    conditions.append("(dm.title LIKE ? OR dm.filename LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                    
                if document_type:
                    conditions.append("dm.document_type = ?")
                    params.append(document_type.value)
                    
                if category:
                    conditions.append("dm.category = ?")
                    params.append(category)
                    
                if status:
                    conditions.append("dm.status = ?")
                    params.append(status.value)
                    
                if version:
                    conditions.append("dm.version = ?")
                    params.append(version)
                    
                if keywords:
                    keyword_conditions = ["dk.keyword = ?" for _ in keywords]
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
                    params.extend(keywords)
                    
                # クエリを組み立て
                if conditions:
                    final_query = f"{base_query} WHERE {' AND '.join(conditions)}"
                else:
                    final_query = base_query
                    
                final_query += " ORDER BY dm.updated_at DESC"
                
                cursor.execute(final_query, params)
                rows = cursor.fetchall()
                
                # メタデータオブジェクトのリストを作成
                documents = []
                for row in rows:
                    metadata = self.get_document(row['id'])
                    if metadata:
                        documents.append(metadata)
                        
                return documents
                
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
            
    def delete_document(self, doc_id: str) -> bool:
        """文書を削除"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 関連データを削除（外部キー制約がないため手動で削除）
                cursor.execute('DELETE FROM document_keywords WHERE doc_id = ?', (doc_id,))
                cursor.execute('DELETE FROM document_relations WHERE source_doc_id = ? OR target_doc_id = ?', 
                             (doc_id, doc_id))
                cursor.execute('DELETE FROM version_history WHERE doc_id = ?', (doc_id,))
                
                # メインの文書を削除
                cursor.execute('DELETE FROM document_metadata WHERE id = ?', (doc_id,))
                
                conn.commit()
                
            logger.info(f"Document deleted: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def update_document_status(self, doc_id: str, status: DocumentStatus, 
                              changed_by: Optional[str] = None) -> bool:
        """文書ステータスを更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ステータスを更新
                cursor.execute('''
                    UPDATE document_metadata 
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                ''', (status.value, datetime.now().isoformat(), doc_id))
                
                # バージョン履歴を記録
                cursor.execute('''
                    INSERT INTO version_history 
                    (doc_id, version, change_type, change_description, changed_by)
                    SELECT version, ?, ?, ?
                    FROM document_metadata WHERE id = ?
                ''', ('status_change', f'Status changed to {status.value}', 
                     changed_by, doc_id))
                
                conn.commit()
                
            logger.info(f"Document status updated: {doc_id} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False
            
    def get_version_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """バージョン履歴を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM version_history 
                    WHERE doc_id = ?
                    ORDER BY change_timestamp DESC
                ''', (doc_id,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []
            
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # 総文書数
                cursor.execute("SELECT COUNT(*) FROM document_metadata")
                stats['total_documents'] = cursor.fetchone()[0]
                
                # 文書タイプ別統計
                cursor.execute('''
                    SELECT document_type, COUNT(*) 
                    FROM document_metadata 
                    GROUP BY document_type
                ''')
                stats['by_type'] = dict(cursor.fetchall())
                
                # ステータス別統計
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM document_metadata 
                    GROUP BY status
                ''')
                stats['by_status'] = dict(cursor.fetchall())
                
                # カテゴリ別統計
                cursor.execute('''
                    SELECT category, COUNT(*) 
                    FROM document_metadata 
                    GROUP BY category
                ''')
                stats['by_category'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
            
    def export_metadata(self, output_file: str) -> bool:
        """メタデータをJSONファイルにエクスポート"""
        try:
            documents = self.search_documents()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_documents': len(documents),
                'documents': [doc.to_dict() for doc in documents]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Metadata exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False
            
    def import_metadata(self, input_file: str) -> bool:
        """JSONファイルからメタデータをインポート"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            documents = data.get('documents', [])
            success_count = 0
            
            for doc_data in documents:
                try:
                    metadata = DocumentMetadata.from_dict(doc_data)
                    if self.add_document(metadata):
                        success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to import document {doc_data.get('id', 'unknown')}: {e}")
                    
            logger.info(f"Imported {success_count}/{len(documents)} documents")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to import metadata: {e}")
            return False


# 便利な関数
def create_road_design_metadata(filename: str,
                               title: str,
                               category: str,
                               **kwargs) -> DocumentMetadata:
    """道路設計文書用のメタデータを作成"""
    doc_id = str(uuid.uuid4())
    
    return DocumentMetadata(
        id=doc_id,
        title=title,
        filename=filename,
        file_path=kwargs.get('file_path', ''),
        file_hash=kwargs.get('file_hash', ''),
        document_type=DocumentType.ROAD_STANDARD,
        category=category,
        processing_timestamp=datetime.now().isoformat(),
        **kwargs
    )