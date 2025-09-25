# コード規約とスタイル (2025年9月18日更新)

## Python コーディング規約

### 命名規則
- **クラス**: PascalCase (例: `QueryEngine`, `LoRATrainer`)
- **関数・メソッド**: snake_case (例: `load_model`, `run_training_task`)
- **定数**: UPPER_SNAKE_CASE (例: `MAX_TOKEN_LENGTH`, `RAG_AVAILABLE`)
- **変数**: snake_case (例: `model_path`, `training_config`)
- **プライベート**: アンダースコア接頭辞 (例: `_internal_method`)

### 型ヒント
```python
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

def load_model(
    model_name: str,
    cache_dir: Optional[Path] = None,
    quantization: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    ...
```

### Docstring形式
```python
def process_document(file_path: str, config: Dict[str, Any]) -> str:
    """
    ドキュメントを処理してテキストを抽出
    
    Args:
        file_path: 処理するファイルのパス
        config: 処理設定の辞書
        
    Returns:
        抽出されたテキスト文字列
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
```

### インポート順序
1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルモジュール

```python
import os
import json
from pathlib import Path
from typing import Optional, Dict

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig

from src.training.lora_finetuning import LoRATrainer
from app.model_utils import load_model_and_tokenizer
```

## ロギング規約
```python
import logging
logger = logging.getLogger(__name__)

# レベル別使用
logger.debug(f"詳細情報: {variable}")
logger.info(f"処理開始: {task_name}")
logger.warning(f"警告: {warning_message}")
logger.error(f"エラー発生: {error}")
```

## エラーハンドリング
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"特定のエラー: {e}")
    # 具体的な対処
except Exception as e:
    logger.error(f"予期しないエラー: {e}")
    raise  # 再スロー or フォールバック
finally:
    cleanup_resources()
```

## ファイル構成
- 1ファイル1責務の原則
- 500行を超える場合は分割検討
- 関連機能はサブモジュールに整理

## コメント規約
- 日本語コメント推奨（技術用語は英語可）
- WHYを説明（WHATはコードが語る）
- TODO/FIXME/HACKにはチケット番号付与

## Git コミットメッセージ
```
<type>(<scope>): <subject>

<body>

<footer>
```

タイプ:
- feat: 新機能
- fix: バグ修正
- docs: ドキュメント
- refactor: リファクタリング
- test: テスト
- chore: その他