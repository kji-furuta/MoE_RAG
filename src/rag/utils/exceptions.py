"""
RAGシステム用の具体的な例外クラス定義
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """RAGシステムの基底例外クラス"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        parts = [self.message]
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " - ".join(parts)


# 既存のエラークラス（後方互換性のため維持）
class ConfigurationError(RAGException):
    """設定関連のエラー"""
    def __init__(self, message: str, config_file: Optional[str] = None, invalid_fields: Optional[list] = None):
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            details={"config_file": config_file, "invalid_fields": invalid_fields}
        )


class DocumentProcessingError(RAGException):
    """ドキュメント処理エラー"""
    def __init__(self, message: str, file_path: Optional[str] = None, error_type: Optional[str] = None):
        super().__init__(
            message,
            error_code="DOC_PROCESSING",
            details={"file_path": file_path, "error_type": error_type}
        )


class SearchError(RAGException):
    """検索処理エラー"""
    def __init__(self, message: str, search_type: Optional[str] = None, query: Optional[str] = None):
        super().__init__(
            message,
            error_code="SEARCH_ERROR",
            details={"search_type": search_type, "query": query[:100] if query else None}
        )


class GenerationError(RAGException):
    """テキスト生成エラー"""
    def __init__(self, message: str, model_name: Optional[str] = None, fallback_available: bool = False):
        super().__init__(
            message,
            error_code="GENERATION_ERROR",
            details={"model_name": model_name, "fallback_available": fallback_available}
        )


class VectorStoreError(RAGException):
    """ベクトルストア関連エラー"""
    def __init__(self, message: str, operation: Optional[str] = None, collection: Optional[str] = None):
        super().__init__(
            message,
            error_code="VECTOR_STORE",
            details={"operation": operation, "collection": collection}
        )


class ModelLoadError(RAGException):
    """モデルロードエラー"""
    def __init__(self, message: str, model_name: Optional[str] = None, memory_required: Optional[int] = None):
        super().__init__(
            message,
            error_code="MODEL_LOAD",
            details={"model_name": model_name, "memory_required_gb": memory_required}
        )


class ValidationError(RAGException):
    """バリデーションエラー"""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            error_code="VALIDATION",
            details={"field": field, "value": str(value)[:100] if value else None}
        )


# 新規追加の具体的な例外クラス
class VectorStoreConnectionError(VectorStoreError):
    """ベクトルストア接続エラー"""
    def __init__(self, message: str, url: Optional[str] = None, retry_count: int = 0):
        super().__init__(message, operation="connection")
        self.error_code = "VECTOR_STORE_CONNECTION"
        self.details.update({"url": url, "retry_count": retry_count})


class EmbeddingDimensionError(VectorStoreError):
    """埋め込み次元エラー"""
    def __init__(self, message: str, expected_dim: int, actual_dim: int):
        super().__init__(message, operation="embedding")
        self.error_code = "EMBEDDING_DIMENSION"
        self.details.update({"expected_dimension": expected_dim, "actual_dimension": actual_dim})


class QueryTimeoutError(SearchError):
    """クエリタイムアウトエラー"""
    def __init__(self, message: str, timeout_seconds: float, query: str):
        super().__init__(message, query=query)
        self.error_code = "QUERY_TIMEOUT"
        self.details.update({"timeout_seconds": timeout_seconds})


class LLMMemoryError(GenerationError):
    """LLMメモリエラー"""
    def __init__(self, message: str, required_memory: int, available_memory: int):
        super().__init__(message)
        self.error_code = "LLM_MEMORY"
        self.details.update({"required_memory_gb": required_memory, "available_memory_gb": available_memory})


class AuthenticationError(RAGException):
    """認証エラー"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTH_ERROR")


class RateLimitError(RAGException):
    """レート制限エラー"""
    def __init__(self, message: str, limit: int, window_seconds: int):
        super().__init__(
            message,
            error_code="RATE_LIMIT",
            details={"limit": limit, "window_seconds": window_seconds}
        )