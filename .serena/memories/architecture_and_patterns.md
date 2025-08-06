# プロジェクトアーキテクチャと設計パターン

## アーキテクチャ概要

### レイヤード・アーキテクチャ
```
┌─────────────────────────────────────────┐
│      Presentation Layer (Web UI)         │
│         - FastAPI (port 8050)           │
│         - WebSocket (real-time)         │
│         - Bootstrap UI                  │
├─────────────────────────────────────────┤
│        Application Layer (API)           │
│    - Training endpoints                 │
│    - RAG endpoints                      │
│    - Model management                   │
├─────────────────────────────────────────┤
│        Business Logic Layer              │
│    - Training services                  │
│    - RAG query engine                   │
│    - Model loaders                      │
├─────────────────────────────────────────┤
│         Data Access Layer                │
│    - Qdrant vector store                │
│    - File system (models/data)          │
│    - Cache management                   │
└─────────────────────────────────────────┘
```

## 主要設計パターン

### 1. Dependency Injection (DI)
```python
# FastAPIの依存性注入システムを活用
async def get_rag_app():
    return rag_app

@app.get("/rag/query")
async def query(
    request: QueryRequest,
    rag: RAGApplication = Depends(get_rag_app)
):
    return await rag.query(request)
```

### 2. Repository Pattern
```python
# データアクセスの抽象化
class ModelRepository:
    def load_model(self, model_path: str) -> Any
    def save_model(self, model: Any, path: str) -> None
    def list_models(self) -> List[ModelInfo]
```

### 3. Factory Pattern
```python
# モデルローダーのファクトリー
def create_model_loader(model_type: str) -> BaseModelLoader:
    if model_type == "lora":
        return LoRAModelLoader()
    elif model_type == "full":
        return FullModelLoader()
```

### 4. Strategy Pattern
```python
# トレーニング戦略の切り替え
class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, model, dataset, config): pass

class LoRATrainingStrategy(TrainingStrategy):
    def train(self, model, dataset, config):
        # LoRA specific training

class FullTrainingStrategy(TrainingStrategy):
    def train(self, model, dataset, config):
        # Full finetuning
```

### 5. Observer Pattern (WebSocket)
```python
# リアルタイム進捗通知
@app.websocket("/ws/training/{task_id}")
async def training_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    while True:
        status = get_training_status(task_id)
        await websocket.send_json(status)
```

## コード組織原則

### 1. Single Responsibility Principle (SRP)
- 各クラス/関数は単一の責任を持つ
- 例: `ModelLoader`はモデル読み込みのみ、`Trainer`は訓練のみ

### 2. Open/Closed Principle (OCP)
- 拡張に対して開き、修正に対して閉じている
- 新しいモデルタイプの追加は既存コードを変更せずに可能

### 3. Dependency Inversion Principle (DIP)
- 高レベルモジュールは低レベルモジュールに依存しない
- 抽象に依存する（インターフェース/プロトコル使用）

## ディレクトリ構造の意味

```
src/
├── training/          # ドメイン: モデル訓練
│   ├── lora_*.py     # LoRA関連機能
│   ├── full_*.py     # フル訓練機能
│   └── ewc_*.py      # 継続学習機能
├── rag/              # ドメイン: 情報検索
│   ├── core/         # RAGコア機能
│   ├── indexing/     # インデックス管理
│   └── retrieval/    # 検索機能
├── utils/            # 横断的関心事
│   ├── logger.py     # ロギング
│   └── gpu_utils.py  # GPU管理
└── data/             # データ処理
```

## 非同期処理パターン

### 1. バックグラウンドタスク
```python
@app.post("/api/train")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_training_task, task_id, request)
    return {"task_id": task_id}
```

### 2. 非同期コンテキストマネージャー
```python
async with rag_app.get_session() as session:
    result = await session.query(prompt)
```

## エラーハンドリング戦略

### 1. 階層的例外処理
```python
class AppException(Exception): pass
class ModelLoadError(AppException): pass
class TrainingError(AppException): pass
class RAGError(AppException): pass
```

### 2. グローバルエラーハンドラー
```python
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )
```

## パフォーマンス最適化

### 1. モデルキャッシング
```python
model_cache = {}  # グローバルキャッシュ

def get_cached_model(model_path: str):
    if model_path not in model_cache:
        model_cache[model_path] = load_model(model_path)
    return model_cache[model_path]
```

### 2. GPU メモリ管理
```python
# 動的バッチサイズ調整
def get_optimal_batch_size(model_size: int, available_memory: int):
    # メモリに基づいてバッチサイズを計算
```

### 3. 非同期I/O
```python
# ファイル操作の非同期化
async def save_results_async(results: dict, path: str):
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(results))
```

## セキュリティ考慮事項

### 1. 入力検証
- Pydantic モデルによる自動検証
- ファイルアップロードのサイズ制限
- パス・トラバーサル攻撃の防止

### 2. 認証・認可
- 環境変数による API キー管理
- HuggingFace トークンの安全な使用

### 3. リソース制限
- 同時実行タスク数の制限
- メモリ使用量の監視
- タイムアウト設定