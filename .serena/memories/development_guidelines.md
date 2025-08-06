# 開発ガイドラインとベストプラクティス

## コーディング規約

### Python スタイルガイド
```python
# 1. インポート順序（isortで自動整理）
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from src.training import LoRATrainer
from src.utils import logger

# 2. クラス定義
class ModelTrainer:
    """モデル訓練を管理するクラス
    
    Args:
        model_name: 使用するモデル名
        config: 訓練設定
        
    Example:
        >>> trainer = ModelTrainer("gpt2", config)
        >>> trainer.train(dataset)
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self._model: Optional[torch.nn.Module] = None
    
    @property
    def model(self) -> torch.nn.Module:
        """遅延読み込みプロパティ"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

# 3. 関数定義
def prepare_dataset(
    data_path: Path,
    tokenizer: Any,
    max_length: int = 512
) -> Dataset:
    """データセットを準備する
    
    Args:
        data_path: データファイルのパス
        tokenizer: 使用するトークナイザー
        max_length: 最大トークン長
        
    Returns:
        準備されたデータセット
        
    Raises:
        FileNotFoundError: データファイルが見つからない場合
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # 処理実装
    return dataset
```

### 命名規則
```python
# 変数名: snake_case
model_name = "gpt2"
batch_size = 32

# 定数: UPPER_SNAKE_CASE
MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-4

# クラス名: PascalCase
class DataProcessor:
    pass

# プライベートメソッド/変数: アンダースコアプレフィックス
class Model:
    def __init__(self):
        self._internal_state = {}
    
    def _process_internal(self):
        pass

# 型エイリアス: PascalCase
ModelOutput = Dict[str, torch.Tensor]
```

## エラーハンドリング

### 適切な例外処理
```python
# ❌ 悪い例: 全ての例外をキャッチ
try:
    model = load_model(path)
except Exception:
    pass  # エラーを握りつぶす

# ✅ 良い例: 具体的な例外を処理
try:
    model = load_model(path)
except FileNotFoundError:
    logger.error(f"Model file not found: {path}")
    raise ModelLoadError(f"Cannot load model from {path}")
except torch.cuda.OutOfMemoryError:
    logger.warning("GPU memory insufficient, falling back to CPU")
    model = load_model(path, device="cpu")
except Exception as e:
    logger.exception(f"Unexpected error loading model: {e}")
    raise
```

### カスタム例外
```python
class AIFTException(Exception):
    """プロジェクト基底例外"""
    pass

class ModelError(AIFTException):
    """モデル関連エラー"""
    pass

class DataError(AIFTException):
    """データ関連エラー"""
    pass

class ConfigError(AIFTException):
    """設定関連エラー"""
    pass
```

## ログ記録

### ログレベルの使い分け
```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: 詳細なデバッグ情報
logger.debug(f"Model parameters: {model.num_parameters()}")

# INFO: 通常の動作情報
logger.info(f"Training started with {len(dataset)} samples")

# WARNING: 注意が必要だが続行可能
logger.warning("GPU memory low, performance may be affected")

# ERROR: エラーだが復旧可能
logger.error(f"Failed to save checkpoint: {e}")

# CRITICAL: 致命的エラー
logger.critical("CUDA not available, cannot continue")
```

### 構造化ログ
```python
# コンテキスト情報を含める
logger.info(
    "Training completed",
    extra={
        "task_id": task_id,
        "duration": time.time() - start_time,
        "final_loss": final_loss,
        "model_name": model_name
    }
)
```

## テスト作成

### ユニットテスト
```python
import pytest
from unittest.mock import Mock, patch

class TestModelLoader:
    @pytest.fixture
    def mock_model(self):
        """モックモデルのフィクスチャ"""
        model = Mock()
        model.num_parameters.return_value = 1000
        return model
    
    def test_load_model_success(self, mock_model):
        """正常系: モデル読み込み成功"""
        with patch("app.model_loader.AutoModel.from_pretrained") as mock_load:
            mock_load.return_value = mock_model
            
            loader = ModelLoader()
            model = loader.load("gpt2")
            
            assert model is not None
            mock_load.assert_called_once_with("gpt2")
    
    def test_load_model_not_found(self):
        """異常系: モデルが見つからない"""
        loader = ModelLoader()
        
        with pytest.raises(ModelNotFoundError):
            loader.load("non_existent_model")
```

### 統合テスト
```python
@pytest.mark.integration
async def test_training_api_flow(client: TestClient):
    """トレーニングAPIの統合テスト"""
    # 1. データアップロード
    with open("test_data.jsonl", "rb") as f:
        response = await client.post(
            "/api/upload-data",
            files={"file": f}
        )
    assert response.status_code == 200
    
    # 2. トレーニング開始
    response = await client.post(
        "/api/train",
        json={
            "model_name": "distilgpt2",
            "training_data": ["test_data.jsonl"],
            "training_method": "lora"
        }
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    
    # 3. ステータス確認
    response = await client.get(f"/api/training-status/{task_id}")
    assert response.json()["status"] in ["running", "completed"]
```

## パフォーマンス考慮事項

### メモリ効率
```python
# ❌ 悪い例: 全データをメモリに読み込む
data = []
with open("large_file.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

# ✅ 良い例: ジェネレーターを使用
def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)
```

### GPU メモリ管理
```python
# メモリクリア
torch.cuda.empty_cache()

# グラデーション累積
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

## ドキュメント作成

### Docstring形式（Google Style）
```python
def process_documents(
    documents: List[str],
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """ドキュメントを処理してチャンクに分割する
    
    Args:
        documents: 処理対象のドキュメントリスト
        chunk_size: 各チャンクの最大サイズ
        overlap: チャンク間のオーバーラップサイズ
    
    Returns:
        処理されたチャンクのリスト。各チャンクは以下の形式:
        {
            "text": str,
            "metadata": dict,
            "chunk_id": int
        }
    
    Raises:
        ValueError: chunk_size <= overlap の場合
        
    Example:
        >>> docs = ["長いテキスト1", "長いテキスト2"]
        >>> chunks = process_documents(docs, chunk_size=100)
        >>> print(len(chunks))
        5
    """
```

## Git コミットメッセージ
```bash
# 形式: <type>(<scope>): <subject>

feat(training): LoRA訓練の動的バッチサイズ調整を追加
fix(rag): Qdrant接続タイムアウトの修正
docs(api): RAGエンドポイントのドキュメント更新
refactor(model): モデルローダーの責任分離
test(integration): 大規模モデルの統合テスト追加
chore(deps): transformers を v4.36.0 に更新

# type:
# - feat: 新機能
# - fix: バグ修正
# - docs: ドキュメント
# - refactor: リファクタリング
# - test: テスト
# - chore: ビルド、補助ツール
```