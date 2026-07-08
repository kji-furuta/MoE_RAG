# コーディング規約とスタイルガイド

## Python コーディング規約

### 基本設定
- **Pythonバージョン**: 3.8以上（本番環境: 3.12.3）
- **フォーマッター**: Black (line-length: 88)
- **インポート順序**: isort (profile: black)
- **リンター**: flake8

### 命名規則
- **クラス名**: PascalCase（例: `QueryEngine`, `RAGProcessor`）
- **関数/メソッド名**: snake_case（例: `process_document`, `get_embeddings`）
- **定数**: UPPER_SNAKE_CASE（例: `MAX_TOKENS`, `DEFAULT_MODEL`）
- **プライベートメソッド**: アンダースコア接頭辞（例: `_internal_process`）

### 型ヒント
```python
# 必須: 関数シグネチャには型ヒントを付ける
from typing import List, Dict, Optional, Union, Tuple

def process_query(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Union[str, float]]]:
    """クエリを処理して結果を返す"""
    pass
```

### ドキュメンテーション
```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    関数の簡潔な説明。
    
    Args:
        param1 (str): パラメータ1の説明
        param2 (int): パラメータ2の説明
        
    Returns:
        Dict[str, Any]: 戻り値の説明
        
    Raises:
        ValueError: エラー条件の説明
    """
    pass
```

### クラス構造
```python
class ExampleClass:
    """クラスの説明"""
    
    def __init__(self, config: Dict[str, Any]):
        """初期化メソッド"""
        self.config = config
        self._private_var = None
        
    @property
    def public_property(self) -> str:
        """プロパティの説明"""
        return self._private_var
        
    def public_method(self) -> None:
        """パブリックメソッド"""
        pass
        
    def _private_method(self) -> None:
        """プライベートメソッド"""
        pass
```

## ファイル構造規約

### モジュール構成
- 1ファイル1クラス/1機能を基本とする
- `__init__.py`で公開APIを明示的に定義
- utilsは`src/utils/`に集約

### インポート順序
1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルモジュール

```python
# 標準ライブラリ
import os
import sys
from pathlib import Path

# サードパーティ
import torch
import numpy as np
from transformers import AutoModel

# ローカル
from src.rag.core import QueryEngine
from src.utils.logger import get_logger
```

## エラーハンドリング

```python
# 明示的なエラーハンドリング
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error occurred")
    return default_value
```

## ロギング

```python
from loguru import logger

# 設定
logger.add("logs/app.log", rotation="10 MB")

# 使用
logger.info("Processing started")
logger.debug(f"Input data: {data}")
logger.error(f"Error occurred: {error}")
logger.success("Task completed successfully")
```

## 非同期処理

```python
# FastAPIエンドポイント
from fastapi import FastAPI, BackgroundTasks

@app.post("/api/process")
async def process_endpoint(
    data: RequestModel,
    background_tasks: BackgroundTasks
):
    # 非同期処理
    background_tasks.add_task(long_running_task, data)
    return {"status": "processing"}
```

## テスト規約

```python
# pytestを使用
import pytest

def test_function_name():
    """テスト関数の説明"""
    # Arrange
    input_data = prepare_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

## コメント規約

```python
# TODO: 実装が必要な箇所
# FIXME: 修正が必要なバグ
# NOTE: 重要な説明
# HACK: 一時的な回避策
# WARNING: 注意が必要な処理
```

## セキュリティ考慮事項

- APIキーや認証情報は環境変数で管理
- SQLインジェクション対策（パラメータ化クエリ使用）
- ファイルアップロードの検証（タイプ、サイズ）
- CORSの適切な設定
- センシティブ情報のログ出力禁止