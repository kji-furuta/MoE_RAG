"""
統一化された例外クラス定義
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """RAGシステムの基底例外クラス"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(RAGException):
    """設定関連のエラー"""
    pass


class DocumentProcessingError(RAGException):
    """ドキュメント処理エラー"""
    pass


class SearchError(RAGException):
    """検索処理エラー"""
    pass


class GenerationError(RAGException):
    """テキスト生成エラー"""
    pass


class VectorStoreError(RAGException):
    """ベクトルストア関連エラー"""
    pass


class ModelLoadError(RAGException):
    """モデルロードエラー"""
    pass


class ValidationError(RAGException):
    """バリデーションエラー"""
    pass