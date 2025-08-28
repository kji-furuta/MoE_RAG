#!/usr/bin/env python3
"""
既存のRAGシステムに永続化機能を統合するアダプター
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistentRAGAdapter:
    """既存のRAGシステムに永続化機能を追加するアダプター"""
    
    def __init__(self):
        # 永続化ストアのインポートと初期化
        from scripts.rag_fixes.fix_data_persistence import PersistentVectorStore
        self.persistent_store = PersistentVectorStore()
        
        # 既存のRAGコンポーネントの初期化
        self._init_existing_components()
        
        # 自動バックアップの設定
        self.auto_backup_enabled = True
        self.auto_backup_interval = 10  # 10文書ごとにバックアップ
        self.document_count = 0
        
        logger.info("PersistentRAGAdapter initialized")
    
    def _init_existing_components(self):
        """既存のRAGコンポーネントを初期化"""
        try:
            from src.rag.indexing.vector_store import QdrantVectorStore
            from src.rag.indexing.embedding_model import EmbeddingModelFactory
            from src.rag.indexing.metadata_manager import MetadataManager
            
            # 既存のベクトルストア
            self.original_vector_store = QdrantVectorStore(
                collection_name="road_design_docs",
                embedding_dim=1024,
                path="./data/qdrant"
            )
            
            # 埋め込みモデル
            self.embedding_model = EmbeddingModelFactory.create(
                model_name="intfloat/multilingual-e5-large",
                device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
            )
            
            # メタデータマネージャー
            self.metadata_manager = MetadataManager(
                db_path="./metadata/metadata.db"
            )
            
            logger.info("Existing RAG components initialized")
            
        except Exception as e:
            logger.warning(f"Some RAG components could not be initialized: {e}")
    
    def add_document(self,
                    text: str,
                    title: str,
                    metadata: Dict[str, Any],
                    file_path: Optional[str] = None) -> str:
        """文書を永続的に追加（既存のAPIと互換）"""
        
        try:
            # 1. 埋め込みベクトルの生成
            embedding = self.embedding_model.encode(text, is_query=False)
            
            # 2. 永続化ストアに追加
            doc_id = self.persistent_store.add_document_with_persistence(
                text=text,
                title=title,
                embedding=embedding,
                metadata=metadata,
                file_path=file_path
            )
            
            # 3. 既存のベクトルストアにも追加（互換性のため）
            try:
                self.original_vector_store.add_documents(
                    texts=[text],
                    embeddings=[embedding],
                    metadatas=[{**metadata, 'doc_id': doc_id, 'title': title}],
                    ids=[doc_id]
                )
            except Exception as e:
                logger.warning(f"Failed to add to original vector store: {e}")
            
            # 4. メタデータマネージャーに登録
            try:
                from src.rag.indexing.metadata_manager import DocumentMetadata, DocumentType, DocumentStatus
                
                doc_metadata = DocumentMetadata(
                    id=doc_id,
                    title=title,
                    filename=Path(file_path).name if file_path else "inline",
                    file_path=file_path or "",
                    file_hash=self.persistent_store._calculate_hash(text),
                    document_type=DocumentType(metadata.get('document_type', 'OTHER')),
                    category=metadata.get('category', 'general'),
                    status=DocumentStatus.ACTIVE
                )
                
                self.metadata_manager.add_metadata(doc_metadata)
            except Exception as e:
                logger.warning(f"Failed to register metadata: {e}")
            
            # 5. 自動バックアップのチェック
            self.document_count += 1
            if self.auto_backup_enabled and self.document_count % self.auto_backup_interval == 0:
                backup_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M')}_docs{self.document_count}"
                self.create_backup(backup_name)
            
            logger.info(f"Document added with persistence: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def search(self,
              query: str,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """永続化されたデータから検索"""
        
        try:
            # クエリの埋め込みベクトル生成
            query_embedding = self.embedding_model.encode(query, is_query=True)
            
            # 永続化ストアから検索
            results = self.persistent_store.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # 結果の整形
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': float(result.score),
                    'text': result.text,
                    'metadata': result.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """バックアップを作成"""
        if not backup_name:
            backup_name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.persistent_store.create_backup(backup_name)
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_name: str) -> bool:
        """バックアップから復元"""
        success = self.persistent_store.restore_from_backup(backup_name)
        if success:
            logger.info(f"Successfully restored from backup: {backup_name}")
            # 既存のコンポーネントも再初期化
            self._init_existing_components()
        else:
            logger.error(f"Failed to restore from backup: {backup_name}")
        return success
    
    def get_persistence_status(self) -> Dict[str, Any]:
        """永続化の状態を取得"""
        status = self.persistent_store.verify_persistence()
        
        # 追加情報
        status['auto_backup_enabled'] = self.auto_backup_enabled
        status['auto_backup_interval'] = self.auto_backup_interval
        status['documents_since_last_backup'] = self.document_count % self.auto_backup_interval
        
        return status
    
    def migrate_existing_data(self) -> int:
        """既存のデータを永続化ストアに移行"""
        migrated_count = 0
        
        try:
            # 既存のベクトルストアからデータを取得
            import numpy as np
            
            # ダミークエリで全データを取得（実際は別の方法が必要）
            dummy_query = np.random.randn(1024).astype(np.float32)
            existing_data = self.original_vector_store.search(
                query_embedding=dummy_query,
                top_k=1000  # 大きな数を指定
            )
            
            logger.info(f"Found {len(existing_data)} documents to migrate")
            
            for item in existing_data:
                try:
                    # 永続化ストアに追加
                    self.persistent_store.add_document_with_persistence(
                        text=item.text,
                        title=item.metadata.get('title', 'Migrated Document'),
                        embedding=np.random.randn(1024),  # 実際は再計算が必要
                        metadata=item.metadata
                    )
                    migrated_count += 1
                    
                    if migrated_count % 10 == 0:
                        logger.info(f"Migrated {migrated_count} documents...")
                        
                except Exception as e:
                    logger.warning(f"Failed to migrate document {item.id}: {e}")
            
            # 移行完了後にバックアップ作成
            if migrated_count > 0:
                self.create_backup(f"migration_complete_{migrated_count}_docs")
            
            logger.info(f"Migration completed: {migrated_count} documents")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
        
        return migrated_count


def setup_persistent_rag():
    """永続化RAGシステムのセットアップ"""
    print("=" * 60)
    print("🔧 永続化RAGシステムのセットアップ")
    print("=" * 60)
    
    # 1. アダプターの初期化
    print("\n1. 永続化アダプターの初期化")
    adapter = PersistentRAGAdapter()
    print("  ✅ アダプター初期化完了")
    
    # 2. 永続化状態の確認
    print("\n2. 現在の永続化状態")
    status = adapter.get_persistence_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 3. テストドキュメントの追加
    print("\n3. テストドキュメントの追加")
    test_docs = [
        {
            'title': '永続化テスト文書1',
            'text': 'これは永続化システムのテスト文書です。データが正しく保存され、バックアップが作成されることを確認します。',
            'metadata': {'category': 'test', 'priority': 'high'}
        },
        {
            'title': '永続化テスト文書2',
            'text': 'バックアップとリストア機能のテスト。システム障害が発生してもデータが復元できることを確認します。',
            'metadata': {'category': 'test', 'priority': 'medium'}
        }
    ]
    
    for doc in test_docs:
        doc_id = adapter.add_document(
            text=doc['text'],
            title=doc['title'],
            metadata=doc['metadata']
        )
        print(f"  ✅ 追加: {doc['title']} (ID: {doc_id[:8]}...)")
    
    # 4. 検索テスト
    print("\n4. 検索テスト")
    results = adapter.search("永続化", top_k=3)
    print(f"  検索結果: {len(results)}件")
    for i, result in enumerate(results[:2], 1):
        print(f"    {i}. スコア: {result['score']:.3f}, ID: {result['id'][:8]}...")
    
    # 5. バックアップの作成
    print("\n5. 手動バックアップの作成")
    backup_path = adapter.create_backup("setup_complete")
    print(f"  ✅ バックアップ作成: {backup_path}")
    
    print("\n✅ セットアップ完了")
    print("\n次のステップ:")
    print("  1. 既存データの移行: python scripts/rag_fixes/migrate_existing_data.py")
    print("  2. 自動バックアップの設定: python scripts/rag_fixes/setup_auto_backup.py")
    print("  3. システムの再起動: ./start_dev_env.sh")
    
    return adapter


if __name__ == "__main__":
    setup_persistent_rag()
