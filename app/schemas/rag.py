"""RAG関連のPydanticデータモデル定義."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    description: str
    size: str
    status: str


class QueryRequest(BaseModel):
    """RAGクエリリクエスト"""
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(5, description="取得する結果数", ge=1, le=20)
    search_type: str = Field("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
    include_sources: bool = Field(True, description="ソース情報を含めるか")
    filters: Optional[Dict[str, Any]] = Field(None, description="検索フィルター")
    document_ids: Optional[List[str]] = Field(None, description="検索対象文書IDリスト")
    model: Optional[str] = Field(None, description="使用するLLMモデル (例: ollama:deepseek-32b-rag)")


class QueryResponse(BaseModel):
    """RAGクエリレスポンス"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class BatchQueryRequest(BaseModel):
    """バッチクエリリクエスト"""
    queries: List[str] = Field(..., description="クエリリスト")
    top_k: int = Field(5, description="取得する結果数", ge=1, le=20)
    search_type: str = Field("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")


class SystemInfoResponse(BaseModel):
    """システム情報レスポンス"""
    status: str
    system_info: Dict[str, Any]
    timestamp: str


class DocumentUploadResponse(BaseModel):
    """文書アップロードレスポンス"""
    status: str
    message: str
    document_id: Optional[str] = None
    processing_status: str
    metadata: Optional[Dict[str, Any]] = None


class SavedSearchResult(BaseModel):
    """保存された検索結果"""
    id: str
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    search_type: str
    top_k: int
    saved_at: str
    metadata: Optional[Dict[str, Any]] = None


class SaveSearchRequest(BaseModel):
    """検索結果保存リクエスト"""
    query_response: QueryResponse
    name: Optional[str] = Field(None, description="保存名")
    tags: Optional[List[str]] = Field(None, description="タグ")


class SearchHistoryResponse(BaseModel):
    """検索履歴レスポンス"""
    total: int
    results: List[SavedSearchResult]
    page: int
    limit: int
