"""
バリデーションユーティリティ
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re

from .exceptions import ValidationError


def validate_query(
    query: str,
    min_length: int = 1,
    max_length: int = 2000,
    allow_empty: bool = False
) -> str:
    """
    クエリテキストのバリデーション
    
    Args:
        query: クエリテキスト
        min_length: 最小文字数
        max_length: 最大文字数
        allow_empty: 空文字列を許可するか
    
    Returns:
        正規化されたクエリ
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    if not isinstance(query, str):
        raise ValidationError(f"Query must be string, got {type(query)}")
    
    # 空白の正規化
    query = query.strip()
    
    if not allow_empty and not query:
        raise ValidationError("Query cannot be empty")
    
    if len(query) < min_length:
        raise ValidationError(
            f"Query too short (min: {min_length} chars)"
        )
    
    if len(query) > max_length:
        raise ValidationError(
            f"Query too long (max: {max_length} chars)"
        )
    
    return query


def validate_config(config: Dict[str, Any], required_fields: List[str]) -> None:
    """
    設定のバリデーション
    
    Args:
        config: 設定辞書
        required_fields: 必須フィールドのリスト
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    if not isinstance(config, dict):
        raise ValidationError("Config must be a dictionary")
    
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required config fields: {', '.join(missing_fields)}"
        )


def validate_document(
    document: Union[str, Path, Dict[str, Any]],
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    ドキュメントパスのバリデーション
    
    Args:
        document: ドキュメントパスまたは情報
        allowed_extensions: 許可される拡張子リスト
    
    Returns:
        検証済みのパス
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    if isinstance(document, dict):
        if 'path' not in document:
            raise ValidationError("Document dict must contain 'path' key")
        document = document['path']
    
    if isinstance(document, str):
        document = Path(document)
    
    if not isinstance(document, Path):
        raise ValidationError(f"Invalid document type: {type(document)}")
    
    if not document.exists():
        raise ValidationError(f"Document not found: {document}")
    
    if not document.is_file():
        raise ValidationError(f"Not a file: {document}")
    
    if allowed_extensions:
        if document.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension: {document.suffix}. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
    
    return document


def validate_top_k(top_k: int, min_val: int = 1, max_val: int = 100) -> int:
    """
    top_k パラメータのバリデーション
    
    Args:
        top_k: 検証する値
        min_val: 最小値
        max_val: 最大値
    
    Returns:
        検証済みの値
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            raise ValidationError(f"top_k must be integer, got {type(top_k)}")
    
    if top_k < min_val:
        raise ValidationError(f"top_k must be >= {min_val}, got {top_k}")
    
    if top_k > max_val:
        raise ValidationError(f"top_k must be <= {max_val}, got {top_k}")
    
    return top_k


def validate_search_type(search_type: str) -> str:
    """
    検索タイプのバリデーション
    
    Args:
        search_type: 検索タイプ
    
    Returns:
        検証済みの検索タイプ
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    valid_types = ['vector', 'keyword', 'hybrid']
    
    search_type = search_type.lower().strip()
    
    if search_type not in valid_types:
        raise ValidationError(
            f"Invalid search type: {search_type}. "
            f"Valid types: {', '.join(valid_types)}"
        )
    
    return search_type


def validate_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    検索フィルターのバリデーション
    
    Args:
        filters: フィルター辞書
    
    Returns:
        検証済みのフィルター
    
    Raises:
        ValidationError: バリデーション失敗時
    """
    if filters is None:
        return None
    
    if not isinstance(filters, dict):
        raise ValidationError("Filters must be a dictionary")
    
    # 特定のフィールドの検証
    if 'date_from' in filters or 'date_to' in filters:
        # 日付フォーマットの検証
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        
        for field in ['date_from', 'date_to']:
            if field in filters:
                if not date_pattern.match(str(filters[field])):
                    raise ValidationError(
                        f"Invalid date format for {field}: {filters[field]}. "
                        "Expected: YYYY-MM-DD"
                    )
    
    return filters