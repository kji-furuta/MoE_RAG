"""
ベクトルストア管理モジュール
Qdrantをメインに、ChromaDBやFAISSにも対応
"""

import os
import json
import shutil
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

import qdrant_client
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HnswConfigDiff
)


@dataclass
class SearchResult:
    """検索結果を格納するデータクラス"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    
    
class VectorStore:
    """ベクトルストアの基底クラス"""
    
    def __init__(self, collection_name: str, embedding_dim: int = 1024):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
    def add_documents(self, texts: List[str], embeddings: List[np.ndarray], 
                     metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """文書を追加"""
        raise NotImplementedError
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """類似検索を実行"""
        raise NotImplementedError
        
    def delete(self, ids: List[str]) -> None:
        """文書を削除"""
        raise NotImplementedError
        
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> None:
        """メタデータを更新"""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrantベースのベクトルストア実装"""
    
    def __init__(self, 
                 collection_name: str = "road_design_docs",
                 embedding_dim: int = 1024,
                 path: str = "./qdrant_data",
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 prefer_grpc: bool = True,
                 client: Optional[qdrant_client.QdrantClient] = None):
        """
        Args:
            collection_name: コレクション名
            embedding_dim: 埋め込みベクトルの次元数
            path: ローカルストレージパス（urlが指定されていない場合）
            url: Qdrantサーバーのurl（リモート使用時）
            api_key: APIキー（リモート使用時）
            prefer_grpc: gRPC接続を優先するか
            client: 既存のQdrantクライアント（共有使用時）
        """
        super().__init__(collection_name, embedding_dim)
        
        # Qdrantクライアントの初期化
        try:
            if client:
                # 既存のクライアントを使用
                self.client = client
                logger.info("Using provided Qdrant client")
            elif url:
                self.client = qdrant_client.QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc
                )
            else:
                # ローカルモード
                self.client = qdrant_client.QdrantClient(
                    path=path,
                    prefer_grpc=False  # ローカルではgRPC使用不可
                )
                
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            # 既存のクライアントをクリーンアップして再試行
            try:
                import shutil
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path, exist_ok=True)
                    
                self.client = qdrant_client.QdrantClient(
                    path=path,
                    prefer_grpc=False
                )
                self._ensure_collection()
            except Exception as retry_e:
                logger.error(f"Failed to reinitialize Qdrant client: {retry_e}")
                raise
        
    def _ensure_collection(self):
        """コレクションが存在することを確認し、なければ作成"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )
            
            # インデックスを作成
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_type",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="version",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chapter",
                field_schema="integer"
            )
            
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: List[np.ndarray], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str]) -> None:
        """文書をベクトルストアに追加"""
        
        logger.info(f"Adding {len(texts)} documents to vector store")
        logger.info(f"Embedding dimensions: {[emb.shape for emb in embeddings[:3]]}")
        
        points = []
        for i, (text, embedding, metadata, doc_id) in enumerate(
            zip(texts, embeddings, metadatas, ids)
        ):
            # メタデータにテキストを追加
            payload = metadata.copy()
            payload["text"] = text
            
            # embeddingが正しい形式かチェック
            if isinstance(embedding, np.ndarray):
                vector_list = embedding.tolist()
            else:
                vector_list = list(embedding)
            
            # UUIDを生成してQdrant互換のIDにする
            # 元のIDはメタデータに保存
            payload["original_id"] = doc_id
            qdrant_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=qdrant_id,
                vector=vector_list,
                payload=payload
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} points for upsert")
        
        # バッチで追加
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Batch {i//batch_size + 1}: upserted {len(batch)} points, operation_id: {operation_info.operation_id}")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully added {len(texts)} documents to {self.collection_name}")
        
    def search(self, 
              query_embedding: np.ndarray, 
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None,
              score_threshold: float = 0.0) -> List[SearchResult]:
        """類似検索を実行"""
        
        # フィルタの構築
        qdrant_filter = None
        if filters:
            must_conditions = []
            should_conditions = []
            
            for key, value in filters.items():
                if key == "document_ids" and isinstance(value, list) and len(value) > 0:
                    # document_idsは文書のIDリストを示す（MetadataManagerのUUID）
                    for doc_id in value:
                        # doc_idでマッチング（MetadataManagerのID）
                        doc_id_condition = FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                        should_conditions.append(doc_id_condition)
                        
                        # 後方互換性のためoriginal_idでもマッチング
                        original_id_condition = FieldCondition(
                            key="original_id", 
                            match=MatchValue(value=doc_id)
                        )
                        should_conditions.append(original_id_condition)
                else:
                    # その他のフィルタは通常通り処理
                    condition = FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    must_conditions.append(condition)
            
            # フィルタを構築
            if should_conditions and must_conditions:
                qdrant_filter = Filter(must=must_conditions, should=should_conditions)
            elif should_conditions:
                qdrant_filter = Filter(should=should_conditions)
            elif must_conditions:
                qdrant_filter = Filter(must=must_conditions)
                
        # 検索実行
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold
        )
        
        # 結果を整形
        results = []
        for hit in search_result:
            result = SearchResult(
                id=str(hit.id),
                score=hit.score,
                text=hit.payload.get("text", ""),
                metadata={k: v for k, v in hit.payload.items() if k != "text"}
            )
            results.append(result)
            
        return results
        
    def hybrid_search(self,
                     query_embedding: np.ndarray,
                     keywords: List[str],
                     top_k: int = 5,
                     vector_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """ベクトル検索とキーワード検索のハイブリッド検索"""
        
        # ベクトル検索
        vector_results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # リランキングのため多めに取得
            filters=filters
        )
        
        # キーワード検索（テキストフィールドに対して）
        keyword_conditions = []
        for keyword in keywords:
            # 各キーワードがテキストに含まれているかチェック
            condition = FieldCondition(
                key="text",
                match=MatchValue(value=keyword)
            )
            keyword_conditions.append(condition)
            
        if keyword_conditions and filters:
            # 既存のフィルタと組み合わせ
            filter_conditions = []
            for key, value in filters.items():
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                filter_conditions.append(condition)
                
            combined_filter = Filter(
                must=filter_conditions,
                should=keyword_conditions
            )
        elif keyword_conditions:
            combined_filter = Filter(should=keyword_conditions)
        else:
            combined_filter = None
            
        # Qdrantではキーワード検索が限定的なので、
        # ここでは簡易的な実装とする
        # 本格的な実装では、別途全文検索エンジンと組み合わせる
        
        # スコアの再計算
        result_dict = {}
        for result in vector_results:
            # キーワードマッチング数をカウント
            keyword_matches = sum(
                1 for keyword in keywords 
                if keyword.lower() in result.text.lower()
            )
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            
            # ハイブリッドスコアを計算
            hybrid_score = (
                vector_weight * result.score + 
                keyword_weight * keyword_score
            )
            
            result_dict[result.id] = (result, hybrid_score)
            
        # スコアでソートして上位k件を返す
        sorted_results = sorted(
            result_dict.values(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [result for result, _ in sorted_results]
        
    def delete(self, ids: List[str]) -> None:
        """文書を削除"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        logger.info(f"Deleted {len(ids)} documents from collection {self.collection_name}")
        
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> None:
        """メタデータを更新"""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[id]
        )
        logger.info(f"Updated metadata for document {id}")
        
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "config": "config_unavailable"  # pydantic validation errorを避けるため
            }
        except Exception as e:
            logger.warning(f"Failed to get detailed collection info: {e}")
            # フォールバック: リスト取得で基本情報のみ取得
            try:
                collections = self.client.get_collections().collections
                for collection in collections:
                    if collection.name == self.collection_name:
                        return {
                            "vectors_count": getattr(collection, 'vectors_count', 0),
                            "indexed_vectors_count": getattr(collection, 'vectors_count', 0),
                            "status": "available",
                            "config": "config_unavailable"
                        }
                return {
                    "vectors_count": 0,
                    "indexed_vectors_count": 0,
                    "status": "not_found",
                    "config": "config_unavailable"
                }
            except Exception as fallback_e:
                logger.warning(f"Failed to get collection info via fallback: {fallback_e}")
                return {
                    "vectors_count": 0,
                    "indexed_vectors_count": 0,
                    "status": "error",
                    "config": "config_unavailable"
                }
        
    def clear_collection(self) -> None:
        """コレクションをクリア（開発用）"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.warning(f"Cleared collection {self.collection_name}")