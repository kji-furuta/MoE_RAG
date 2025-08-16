"""
ロギングユーティリティ
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional
import json
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    統一されたロガー設定
    
    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルパス
        format_string: ログフォーマット文字列
    
    Returns:
        設定済みのロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # デフォルトフォーマット
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # コンソールハンドラー
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # ファイルハンドラー
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_performance(logger: Optional[logging.Logger] = None):
    """
    パフォーマンス測定デコレータ
    
    Args:
        logger: 使用するロガー（Noneの場合はデフォルト）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = None
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                elapsed_time = time.time() - start_time
                
                log_data = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'elapsed_time': f'{elapsed_time:.3f}s',
                    'success': error is None
                }
                
                if error:
                    log_data['error'] = str(error)
                
                # ロガーの選択
                _logger = logger or logging.getLogger(func.__module__)
                
                if error:
                    _logger.error(f"Performance: {json.dumps(log_data)}")
                else:
                    _logger.info(f"Performance: {json.dumps(log_data)}")
        
        return wrapper
    return decorator


class ContextLogger:
    """コンテキストマネージャー型のロガー"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Failed {self.operation} after {elapsed:.3f}s: {exc_val}"
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {elapsed:.3f}s"
            )
        
        return False