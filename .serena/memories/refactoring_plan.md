# AI_FT_3 serenaMCPリファクタリング計画

## 概要
main_unified.py（1600行以上）を中心とした大規模リファクタリング。serenaMCPツールを活用して安全かつ段階的に実施。

## 現状分析
- **main_unified.py**: 1660行以上、16のクラス、50以上の関数
- **問題点**:
  - 単一ファイルに多くの責任が集中
  - ファインチューニングとRAGの機能が混在
  - 長大な関数（run_training_task: 300行）
  - モデルクラスとAPIエンドポイントが同一ファイル

## リファクタリング手順

### Phase 1: 構造分析と準備（serenaツール活用）
```bash
# 1. 依存関係の分析
serena:find_referencing_symbols "TrainingRequest" "app/main_unified.py"
serena:find_referencing_symbols "RAGApplication" "app/main_unified.py"

# 2. 重複コードの検出
serena:search_for_pattern "model.*=.*AutoModel" 
serena:search_for_pattern "try:.*except.*Exception"

# 3. 現在の構造を記録
serena:write_memory "original_structure" "[現在の構造分析結果]"
```

### Phase 2: ディレクトリ構造の作成
```python
# 新しい構造を作成
app/
├── api/
│   ├── __init__.py
│   ├── training.py      # トレーニング関連エンドポイント
│   ├── rag.py          # RAG関連エンドポイント
│   ├── models.py       # モデル管理エンドポイント
│   └── admin.py        # 管理系エンドポイント
├── services/
│   ├── __init__.py
│   ├── training_service.py
│   ├── model_service.py
│   ├── rag_service.py
│   └── cache_service.py
├── models/
│   ├── __init__.py
│   ├── request_models.py  # Pydanticモデル
│   └── response_models.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── dependencies.py
└── utils/
    ├── __init__.py
    ├── exceptions.py
    └── validators.py
```

### Phase 3: Pydanticモデルの分離
```python
# 1. モデルクラスを特定
serena:find_symbol "BaseModel" "app/main_unified.py"

# 2. 新しいファイルを作成
filesystem:write_file "app/models/request_models.py" """
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TrainingRequest(BaseModel):
    model_name: str
    training_data: List[str]
    training_method: str = "lora"
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]
"""

# 3. 既存コードから削除
serena:replace_symbol_body "TrainingRequest" "app/main_unified.py" "# Moved to models/request_models.py"
```

### Phase 4: サービス層の実装
```python
# トレーニングサービスの作成
serena:find_symbol "run_training_task" "app/main_unified.py"

# 関数を分割して移動
filesystem:write_file "app/services/training_service.py" """
import logging
from pathlib import Path
from typing import Dict, Any
from app.models.request_models import TrainingRequest

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.training_tasks = {}
        
    async def run_training(self, task_id: str, request: TrainingRequest):
        # run_training_taskの内容を移動
        pass
"""
```

### Phase 5: APIルーターの分離
```python
# 1. エンドポイントの移動
serena:find_symbol "/api/train" "app/main_unified.py"

# 2. 新しいルーターファイル
filesystem:write_file "app/api/training.py" """
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.models.request_models import TrainingRequest
from app.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])
training_service = TrainingService()

@router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    # 既存のロジックを移動
    pass
"""

# 3. main_unified.pyで統合
serena:replace_regex "app/main_unified.py" 
  "@app.post\\(\"/api/train\"\\).*?return.*?\\}" 
  "# Moved to api/training.py"
```

### Phase 6: 共通処理の抽出
```python
# エラーハンドリングの統一
filesystem:write_file "app/utils/exceptions.py" """
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    pass

def handle_model_loading_error(error: Exception, model_name: str, task_id: str) -> str:
    if "CUDA out of memory" in str(error):
        return f"GPU memory insufficient for {model_name}"
    return str(error)
"""

# 設定値の処理を統一
serena:find_symbol "get_config_value" "app/main_unified.py"
# → app/core/config.pyへ移動
```

### Phase 7: テストの作成
```python
# 各サービスのテスト
filesystem:write_file "tests/test_training_service.py" """
import pytest
from app.services.training_service import TrainingService
from app.models.request_models import TrainingRequest

@pytest.fixture
def training_service():
    return TrainingService()

async def test_run_training(training_service):
    request = TrainingRequest(
        model_name="distilgpt2",
        training_data=["test.jsonl"],
        training_method="lora",
        lora_config={},
        training_config={}
    )
    # テスト実装
"""
```

### Phase 8: 段階的な移行
```python
# 1. 新旧の並行稼働
serena:insert_before_symbol "app" "app/main_unified.py" """
# 新しいルーターをインポート
from app.api import training, rag, models, admin

# ルーターを登録
app.include_router(training.router)
app.include_router(rag.router)
app.include_router(models.router)
app.include_router(admin.router)
"""

# 2. 動作確認後、旧コードを削除
serena:search_for_pattern "@app.post\\(\"/api/"
# 各エンドポイントを確認して削除
```

### Phase 9: クリーンアップ
```python
# 1. 不要なインポートの削除
serena:search_for_pattern "^import.*$" "app/main_unified.py"

# 2. コメントアウトされたコードの削除
serena:search_for_pattern "^\\s*#.*Moved to"

# 3. 最終的な構造の記録
serena:write_memory "refactored_structure" "[新しい構造]"
```

## serenaツール活用のベストプラクティス

1. **分析ツール**
   - `find_symbol`: 特定のシンボルを探す
   - `find_referencing_symbols`: 依存関係を調べる
   - `search_for_pattern`: パターンマッチング

2. **リファクタリングツール**
   - `replace_symbol_body`: 関数/クラスの本体を置換
   - `replace_regex`: 正規表現による置換
   - `insert_before_symbol`: シンボル前に挿入

3. **進捗管理**
   - `write_memory`: 進捗を記録
   - `think_about_task_adherence`: タスクの妥当性確認
   - `think_about_collected_information`: 情報の整理

## 実行順序と注意点

1. **必ずバックアップを作成**
2. **小さな変更から開始**
3. **各ステップでテスト実行**
4. **メモリに進捗を記録**
5. **問題があれば即座にロールバック**

## 期待される成果

- コードの可読性向上（各ファイル200行以下）
- 責任の明確な分離
- テストしやすい構造
- 保守性の向上
- 新機能追加の容易化