"""
データフォーマッティングユーティリティ
"""

from typing import Any, Dict, List, Optional, Union
import json
from datetime import datetime


def format_citations(
    citations: List[Union[Dict[str, Any], Any]],
    max_length: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    引用情報のフォーマット
    
    Args:
        citations: 引用リスト
        max_length: 最大文字数
    
    Returns:
        フォーマット済み引用リスト
    """
    formatted = []
    
    for cite in citations:
        if isinstance(cite, dict):
            formatted_cite = cite.copy()
        elif hasattr(cite, '__dict__'):
            formatted_cite = cite.__dict__.copy()
        else:
            formatted_cite = {'text': str(cite)}
        
        # テキストの切り詰め
        if max_length and 'text' in formatted_cite:
            text = formatted_cite['text']
            if len(text) > max_length:
                formatted_cite['text'] = text[:max_length] + '...'
        
        # デフォルト値の設定
        formatted_cite.setdefault('source', 'Unknown')
        formatted_cite.setdefault('page', None)
        
        formatted.append(formatted_cite)
    
    return formatted


def format_sources(
    sources: List[Union[Dict[str, Any], Any]],
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    ソース情報のフォーマット
    
    Args:
        sources: ソースリスト
        include_metadata: メタデータを含めるか
    
    Returns:
        フォーマット済みソースリスト
    """
    formatted = []
    
    for source in sources:
        if isinstance(source, dict):
            formatted_source = source.copy()
        elif hasattr(source, '__dict__'):
            formatted_source = source.__dict__.copy()
        else:
            formatted_source = {'text': str(source)}
        
        # スコアの正規化（0-1の範囲）
        if 'score' in formatted_source:
            score = formatted_source['score']
            if isinstance(score, (int, float)):
                formatted_source['score'] = min(1.0, max(0.0, float(score)))
            else:
                formatted_source['score'] = 0.0
        else:
            formatted_source['score'] = 0.0
        
        # タイトルの設定
        if 'title' not in formatted_source:
            if 'metadata' in formatted_source and isinstance(formatted_source['metadata'], dict):
                formatted_source['title'] = formatted_source['metadata'].get(
                    'title', 'Untitled Document'
                )
            else:
                formatted_source['title'] = 'Untitled Document'
        
        # メタデータの削除（オプション）
        if not include_metadata and 'metadata' in formatted_source:
            del formatted_source['metadata']
        
        formatted.append(formatted_source)
    
    return formatted


def format_metadata(
    metadata: Dict[str, Any],
    include_timestamps: bool = True,
    readable_format: bool = True
) -> Dict[str, Any]:
    """
    メタデータのフォーマット
    
    Args:
        metadata: メタデータ辞書
        include_timestamps: タイムスタンプを含めるか
        readable_format: 人間が読みやすい形式にするか
    
    Returns:
        フォーマット済みメタデータ
    """
    formatted = metadata.copy() if isinstance(metadata, dict) else {}
    
    # タイムスタンプの追加
    if include_timestamps and 'timestamp' not in formatted:
        formatted['timestamp'] = datetime.now().isoformat()
    
    # 数値の丸め
    for key, value in formatted.items():
        if isinstance(value, float):
            formatted[key] = round(value, 3)
    
    # 読みやすい形式への変換
    if readable_format:
        # バイト数を人間が読みやすい形式に
        if 'memory_usage' in formatted:
            formatted['memory_usage'] = format_bytes(formatted['memory_usage'])
        
        # 時間を人間が読みやすい形式に
        if 'processing_time' in formatted:
            formatted['processing_time'] = format_duration(formatted['processing_time'])
    
    return formatted


def format_bytes(bytes_value: Union[int, float]) -> str:
    """
    バイト数を人間が読みやすい形式に変換
    
    Args:
        bytes_value: バイト数
    
    Returns:
        フォーマット済み文字列
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: Union[int, float]) -> str:
    """
    秒数を人間が読みやすい形式に変換
    
    Args:
        seconds: 秒数
    
    Returns:
        フォーマット済み文字列
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(
    text: str,
    max_length: int = 200,
    suffix: str = "..."
) -> str:
    """
    テキストを指定長で切り詰め
    
    Args:
        text: 対象テキスト
        max_length: 最大文字数
        suffix: 末尾に追加する文字列
    
    Returns:
        切り詰められたテキスト
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_json(
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False
) -> str:
    """
    JSON形式で整形
    
    Args:
        data: データ
        indent: インデント幅
        ensure_ascii: ASCII文字のみ使用するか
    
    Returns:
        JSON文字列
    """
    return json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=str
    )