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
    クエリテキストのバリデーションとサニタイゼーション
    
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
    import re
    import html
    
    if not isinstance(query, str):
        raise ValidationError(f"Query must be string, got {type(query)}", field="query", value=query)
    
    # HTMLエスケープ処理
    query = html.escape(query, quote=True)
    
    # 危険な文字パターンの検出と除去
    # SQLインジェクション対策
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        r"(--|#|/\*|\*/)",  # SQLコメント
        r"(\bOR\b.*=.*)",  # OR 1=1 パターン
        r"(;.*\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)",  # 複数ステートメント
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            # 危険なパターンを検出した場合、除去またはエスケープ
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)
    
    # スクリプトインジェクション対策
    script_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # onload=, onclick= など
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    for pattern in script_patterns:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE)
    
    # 制御文字の除去
    query = re.sub(r'[\x00-\x1F\x7F]', '', query)
    
    # 連続する空白を単一スペースに正規化
    query = re.sub(r'\s+', ' ', query)
    
    # 空白の正規化
    query = query.strip()
    
    # 文字数制限のチェック
    if not allow_empty and not query:
        raise ValidationError("Query cannot be empty", field="query", value="")
    
    if len(query) < min_length:
        raise ValidationError(
            f"Query too short (min: {min_length} chars)",
            field="query",
            value=query
        )
    
    if len(query) > max_length:
        raise ValidationError(
            f"Query too long (max: {max_length} chars)",
            field="query",
            value=query[:100] + "..."
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

def sanitize_html(text: str) -> str:
    """
    HTML/JavaScriptの危険な要素を除去
    
    Args:
        text: サニタイズ対象のテキスト
    
    Returns:
        サニタイズされたテキスト
    """
    import html
    import re
    
    # HTMLエスケープ
    text = html.escape(text, quote=True)
    
    # 危険なHTMLタグとJavaScriptを除去
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'<iframe[^>]*>.*?</iframe>',
        r'javascript:',
        r'on\w+\s*=',
        r'<img[^>]*onerror[^>]*>',
        r'<svg[^>]*onload[^>]*>',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text


def validate_document_id(doc_id: str) -> str:
    """
    ドキュメントIDの検証
    
    Args:
        doc_id: ドキュメントID
    
    Returns:
        検証済みのドキュメントID
    
    Raises:
        ValidationError: 無効なIDの場合
    """
    import re
    
    if not isinstance(doc_id, str):
        raise ValidationError("Document ID must be string", field="doc_id", value=doc_id)
    
    # UUID形式またはファイル名形式を許可
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    filename_pattern = r'^[a-zA-Z0-9_\-\.]+$'
    
    if not (re.match(uuid_pattern, doc_id, re.IGNORECASE) or re.match(filename_pattern, doc_id)):
        raise ValidationError(
            "Invalid document ID format",
            field="doc_id",
            value=doc_id
        )
    
    return doc_id


def validate_file_upload(
    file_content: bytes,
    filename: str,
    allowed_extensions: List[str] = None,
    max_size_mb: int = 50
) -> Dict[str, Any]:
    """
    ファイルアップロードの検証
    
    Args:
        file_content: ファイル内容
        filename: ファイル名
        allowed_extensions: 許可する拡張子リスト
        max_size_mb: 最大ファイルサイズ（MB）
    
    Returns:
        検証結果の辞書
    
    Raises:
        ValidationError: 検証失敗時
    """
    import os
    import magic
    
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.txt', '.md', '.docx', '.json']
    
    # ファイル名の検証
    filename = sanitize_html(filename)
    _, ext = os.path.splitext(filename.lower())
    
    if ext not in allowed_extensions:
        raise ValidationError(
            f"File type not allowed: {ext}",
            field="filename",
            value=filename
        )
    
    # ファイルサイズの検証
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValidationError(
            f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)",
            field="file_size",
            value=file_size_mb
        )
    
    # MIMEタイプの検証（マジックナンバー）
    try:
        mime = magic.from_buffer(file_content[:2048], mime=True)
        
        # 許可するMIMEタイプ
        allowed_mimes = {
            '.pdf': ['application/pdf'],
            '.txt': ['text/plain'],
            '.md': ['text/plain', 'text/markdown'],
            '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            '.json': ['application/json', 'text/plain']
        }
        
        if ext in allowed_mimes and mime not in allowed_mimes[ext]:
            raise ValidationError(
                f"File content does not match extension: {mime} != {ext}",
                field="mime_type",
                value=mime
            )
    except Exception as e:
        logger.warning(f"Could not verify MIME type: {e}")
    
    return {
        'filename': filename,
        'extension': ext,
        'size_mb': file_size_mb,
        'mime_type': mime if 'mime' in locals() else 'unknown'
    }


def validate_api_key(api_key: str) -> str:
    """
    APIキーの検証
    
    Args:
        api_key: APIキー
    
    Returns:
        検証済みのAPIキー
    
    Raises:
        ValidationError: 無効なキーの場合
    """
    import re
    
    if not isinstance(api_key, str):
        raise ValidationError("API key must be string", field="api_key", value="")
    
    # APIキーの形式チェック（英数字とハイフン、アンダースコア）
    if not re.match(r'^[a-zA-Z0-9_\-]{32,128}$', api_key):
        raise ValidationError(
            "Invalid API key format",
            field="api_key",
            value="[REDACTED]"
        )
    
    return api_key
