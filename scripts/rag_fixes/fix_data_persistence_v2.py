#!/usr/bin/env python3
"""
RAGシステムのデータ永続性を改善する包括的なソリューション（修正版）
"""

import os
import sys
import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib
import logging

# パスの設定を修正
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

# 環境に応じたパス設定
if os.path.exists("/workspace"):
    sys.path.insert(0, "/workspace")
elif os.path.exists("/home/kjifu/MoE_RAG"):
    sys.path.insert(0, "/home/kjifu/MoE_RAG")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistentVectorStore:
    """永続性を強化したベクトルストア"""
    
    def __init__(self, base_path: str = "./data/rag_persistent"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # ディレクトリ構造の初期化
        self.vector_path = self.base_path / "vectors"
        self.metadata_path = self.base_path / "metadata"
        self.backup_path = self.base_path / "backups"
        self.checkpoint_path = self.base_path / "checkpoints"
        
        for path in [self.vector_path, self.metadata_path, 
                    self.backup_path, self.checkpoint_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # 永続化DB初期化
        self.db_path = self.metadata_path / "persistent_store.db"
        self._init_database()
        
        # Qdrantストアの初期化
        self._init_vector_store()
        
        logger.info(f"PersistentVectorStore initialized at {self.base_path}")
    
    def _init_database(self):
        """永続化用データベースの初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ドキュメントテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_path TEXT,
                file_hash TEXT UNIQUE,
                embedding_id TEXT,
                vector_dims INTEGER,
                category TEXT,
                subcategory TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                metadata TEXT
            )
        ''')
        
        # ベクトルテーブル（バックアップ用）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                vector_data BLOB,
                dimensions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')
        
        # インデックステーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_name TEXT,
                total_documents INTEGER,
                indexed_documents INTEGER,
                failed_documents INTEGER,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                error_log TEXT
            )
        ''')
        
        # バックアップ履歴テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_name TEXT UNIQUE,
                backup_path TEXT,
                backup_size INTEGER,
                document_count INTEGER,
                vector_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                restore_count INTEGER DEFAULT 0
            )
        ''')
        
        # インデックス作成
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_category ON documents(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vec_doc_id ON vectors(document_id)')
        
        conn.commit()
        conn.close()
        
        logger.info("Persistent database initialized")
    
    def _init_vector_store(self):
        """Qdrantベクトルストアの初期化（エラーハンドリング付き）"""
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            
            self.vector_store = QdrantVectorStore(
                collection_name="persistent_docs",
                embedding_dim=1024,
                path=str(self.vector_path)
            )
            
            # コレクションの状態を確認
            try:
                info = self.vector_store.get_collection_info()
                logger.info(f"Vector store initialized: {info.get('vectors_count', 0)} vectors")
            except Exception as e:
                logger.warning(f"Vector store initialization warning: {e}")
                self.vector_store._ensure_collection()
                
        except ImportError as e:
            logger.warning(f"Could not import QdrantVectorStore, using fallback: {e}")
            # フォールバック実装
            self.vector_store = None
    
    def _calculate_hash(self, text: str) -> str:
        """テキストのハッシュ値を計算"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def add_document_with_persistence(self, 
                                     text: str,
                                     title: str,
                                     embedding: Any,
                                     metadata: Dict[str, Any],
                                     file_path: Optional[str] = None) -> str:
        """文書を永続的に追加"""
        import uuid
        import pickle
        
        # NumPyのインポート（オプショナル）
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False
            logger.warning("NumPy not available, using list for embeddings")
        
        # 文書のハッシュ値計算（重複チェック用）
        doc_hash = self._calculate_hash(text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 重複チェック
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (doc_hash,))
            existing = cursor.fetchone()
            
            if existing:
                logger.warning(f"Document already exists: {existing[0]}")
                return existing[0]
            
            # 新規文書ID生成
            doc_id = str(uuid.uuid4())
            
            # データベースに文書を保存
            cursor.execute('''
                INSERT INTO documents 
                (id, title, content, file_path, file_hash, category, subcategory, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                title,
                text,
                file_path,
                doc_hash,
                metadata.get('category', 'general'),
                metadata.get('subcategory'),
                json.dumps(metadata),
                'indexing'
            ))
            
            # ベクトルデータを保存（バックアップ用）
            if has_numpy and isinstance(embedding, np.ndarray):
                vector_data = pickle.dumps(embedding)
                dimensions = embedding.shape[-1]
            else:
                # リストとして処理
                if hasattr(embedding, '__len__'):
                    vector_data = pickle.dumps(list(embedding))
                    dimensions = len(embedding)
                else:
                    vector_data = pickle.dumps([embedding])
                    dimensions = 1
            
            cursor.execute('''
                INSERT INTO vectors (id, document_id, vector_data, dimensions)
                VALUES (?, ?, ?, ?)
            ''', (str(uuid.uuid4()), doc_id, vector_data, dimensions))
            
            # Qdrantにも追加（利用可能な場合）
            if self.vector_store:
                try:
                    self.vector_store.add_documents(
                        texts=[text],
                        embeddings=[embedding],
                        metadatas=[{**metadata, 'doc_id': doc_id, 'title': title}],
                        ids=[doc_id]
                    )
                except Exception as e:
                    logger.warning(f"Failed to add to vector store: {e}")
            
            # ステータス更新
            cursor.execute('''
                UPDATE documents 
                SET status = 'indexed', indexed_at = CURRENT_TIMESTAMP, embedding_id = ?
                WHERE id = ?
            ''', (doc_id, doc_id))
            
            conn.commit()
            logger.info(f"Document persisted: {doc_id} - {title}")
            
            # 自動バックアップ（10文書ごと）
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'")
            count = cursor.fetchone()[0]
            if count % 10 == 0:
                self.create_backup(f"auto_backup_{count}")
            
            return doc_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to persist document: {e}")
            raise
        finally:
            conn.close()
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """完全バックアップの作成"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self.backup_path / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup: {backup_name}")
        
        try:
            # 1. SQLiteデータベースのバックアップ
            shutil.copy2(self.db_path, backup_dir / "persistent_store.db")
            
            # 2. Qdrantデータのバックアップ（存在する場合）
            qdrant_backup = backup_dir / "qdrant_data"
            if self.vector_path.exists() and any(self.vector_path.iterdir()):
                shutil.copytree(self.vector_path, qdrant_backup)
            
            # 3. メタデータのエクスポート
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 文書一覧をJSON形式で保存
            cursor.execute('''
                SELECT id, title, category, file_path, created_at, status 
                FROM documents
            ''')
            documents = cursor.fetchall()
            
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'document_count': len(documents),
                'documents': [
                    {
                        'id': doc[0],
                        'title': doc[1],
                        'category': doc[2],
                        'file_path': doc[3],
                        'created_at': doc[4],
                        'status': doc[5]
                    }
                    for doc in documents
                ]
            }
            
            with open(backup_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 4. バックアップ情報を記録
            backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
            
            cursor.execute('''
                INSERT OR REPLACE INTO backup_history 
                (backup_name, backup_path, backup_size, document_count, vector_count, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                backup_name,
                str(backup_dir),
                backup_size,
                len(documents),
                len(documents),  # ベクトル数は文書数と同じと仮定
                'completed'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Backup completed: {backup_name} ({backup_size:,} bytes)")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """バックアップからリストア"""
        backup_dir = self.backup_path / backup_name
        
        if not backup_dir.exists():
            logger.error(f"Backup not found: {backup_name}")
            return False
        
        logger.info(f"Restoring from backup: {backup_name}")
        
        try:
            # 1. 現在のデータをバックアップ（安全のため）
            self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 2. SQLiteデータベースのリストア
            backup_db = backup_dir / "persistent_store.db"
            if backup_db.exists():
                shutil.copy2(backup_db, self.db_path)
            
            # 3. Qdrantデータのリストア
            qdrant_backup = backup_dir / "qdrant_data"
            if qdrant_backup.exists():
                if self.vector_path.exists():
                    shutil.rmtree(self.vector_path)
                shutil.copytree(qdrant_backup, self.vector_path)
            
            # 4. ベクトルストアの再初期化
            self._init_vector_store()
            
            # 5. リストア履歴を更新
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE backup_history 
                SET restore_count = restore_count + 1 
                WHERE backup_name = ?
            ''', (backup_name,))
            conn.commit()
            conn.close()
            
            logger.info(f"Restore completed: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def verify_persistence(self) -> Dict[str, Any]:
        """永続性の検証"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 統計情報の収集
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'")
        indexed_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vectors")
        total_vectors = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM backup_history")
        total_backups = cursor.fetchone()[0]
        
        # Qdrantの確認
        if self.vector_store:
            try:
                vector_info = self.vector_store.get_collection_info()
                qdrant_vectors = vector_info.get('vectors_count', 0)
            except:
                qdrant_vectors = 0
        else:
            qdrant_vectors = 0
        
        conn.close()
        
        return {
            'total_documents': total_docs,
            'indexed_documents': indexed_docs,
            'total_vectors': total_vectors,
            'total_backups': total_backups,
            'qdrant_vectors': qdrant_vectors,
            'persistence_status': 'healthy' if indexed_docs > 0 else 'empty',
            'backup_status': 'available' if total_backups > 0 else 'no_backups'
        }
    
    def cleanup_old_backups(self, keep_days: int = 7):
        """古いバックアップの削除"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 古いバックアップを取得
        cursor.execute('''
            SELECT backup_name, backup_path 
            FROM backup_history 
            WHERE created_at < ?
        ''', (cutoff_date.isoformat(),))
        
        old_backups = cursor.fetchall()
        
        for backup_name, backup_path in old_backups:
            backup_dir = Path(backup_path)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                logger.info(f"Deleted old backup: {backup_name}")
            
            # 履歴から削除
            cursor.execute("DELETE FROM backup_history WHERE backup_name = ?", (backup_name,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {len(old_backups)} old backups")


def implement_persistence_improvements():
    """永続性改善の実装"""
    print("=" * 60)
    print("🔧 データ永続性の改善実装")
    print("=" * 60)
    
    # 1. PersistentVectorStoreの初期化
    print("\n1. 永続化ストアの初期化")
    persistent_store = PersistentVectorStore()
    
    # 2. テストデータの追加
    print("\n2. テストデータの追加")
    
    # NumPyが利用できない場合のフォールバック
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
        print("  ⚠️  NumPyが利用できません。リストを使用します。")
    
    test_documents = [
        {
            'title': '道路設計基準書 第1章',
            'text': '道路設計における最小曲線半径は、設計速度に応じて決定される。設計速度60km/hの場合、最小曲線半径は150mとする。',
            'category': '設計基準',
            'subcategory': '幾何構造'
        },
        {
            'title': '縦断勾配の制限',
            'text': '縦断勾配は原則として5%以下とする。ただし、地形の状況によりやむを得ない場合は8%まで許容される。',
            'category': '設計基準',
            'subcategory': '縦断設計'
        },
        {
            'title': '横断勾配の基準',
            'text': '横断勾配は、排水を考慮して片勾配2%を標準とする。曲線部においては、超高を設ける。',
            'category': '設計基準',
            'subcategory': '横断設計'
        }
    ]
    
    for doc in test_documents:
        # ダミーの埋め込みベクトル
        if use_numpy:
            import numpy as np
            embedding = np.random.randn(1024).astype(np.float32)
        else:
            import random
            embedding = [random.random() for _ in range(1024)]
        
        doc_id = persistent_store.add_document_with_persistence(
            text=doc['text'],
            title=doc['title'],
            embedding=embedding,
            metadata={
                'category': doc['category'],
                'subcategory': doc['subcategory']
            }
        )
        print(f"  ✅ 追加: {doc['title']} (ID: {doc_id[:8]}...)")
    
    # 3. バックアップの作成
    print("\n3. バックアップの作成")
    backup_path = persistent_store.create_backup("initial_backup")
    print(f"  ✅ バックアップ作成: {backup_path}")
    
    # 4. 永続性の検証
    print("\n4. 永続性の検証")
    verification = persistent_store.verify_persistence()
    
    print(f"  📊 検証結果:")
    print(f"    総文書数: {verification['total_documents']}")
    print(f"    インデックス済み: {verification['indexed_documents']}")
    print(f"    ベクトル数: {verification['total_vectors']}")
    print(f"    バックアップ数: {verification['total_backups']}")
    print(f"    Qdrantベクトル数: {verification['qdrant_vectors']}")
    print(f"    永続化状態: {verification['persistence_status']}")
    
    print("\n✅ データ永続性の改善が完了しました")
    
    return persistent_store


if __name__ == "__main__":
    try:
        implement_persistence_improvements()
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
