# アーキテクチャとデザインパターン

## システムアーキテクチャ

### レイヤードアーキテクチャ
```
┌─────────────────────────────────────┐
│     Presentation Layer (app/)       │
│  - FastAPI エンドポイント            │
│  - WebSocket ハンドラー              │
│  - 静的ファイル配信                  │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     Business Logic Layer (src/)     │
│  - MoE エキスパートルーティング      │
│  - RAG クエリ処理                   │
│  - 訓練パイプライン                  │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│      Data Access Layer              │
│  - Qdrant ベクトルストア            │
│  - モデルローダー                    │
│  - ファイルシステムアクセス          │
└─────────────────────────────────────┘
```

## 主要デザインパターン

### 1. Mixture of Experts (MoE) パターン
```python
class MoERouter:
    """エキスパート選択ルーター"""
    def route(self, query: str) -> List[Expert]:
        # ゲーティングネットワークによる専門家選択
        scores = self.gating_network(query)
        return self.select_top_k_experts(scores)
```

### 2. Strategy パターン（訓練戦略）
```python
class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, model, data): pass

class LoRATraining(TrainingStrategy):
    def train(self, model, data):
        # LoRA特有の訓練ロジック

class FullTraining(TrainingStrategy):
    def train(self, model, data):
        # フル訓練ロジック
```

### 3. Factory パターン（モデル生成）
```python
class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig) -> BaseModel:
        if config.type == "lora":
            return LoRAModel(config)
        elif config.type == "full":
            return FullModel(config)
```

### 4. Singleton パターン（リソース管理）
```python
class QdrantClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 5. Pipeline パターン（RAG処理）
```python
class RAGPipeline:
    def __init__(self):
        self.stages = [
            DocumentRetrieval(),
            Reranking(),
            ResponseGeneration()
        ]
    
    def process(self, query):
        result = query
        for stage in self.stages:
            result = stage.process(result)
        return result
```

## 非同期処理パターン

### AsyncIO + FastAPI
```python
@app.post("/api/query")
async def process_query(request: QueryRequest):
    # 非同期でMoEとRAGを並列実行
    moe_task = asyncio.create_task(moe_process(request))
    rag_task = asyncio.create_task(rag_process(request))
    
    moe_result, rag_result = await asyncio.gather(moe_task, rag_task)
    return fusion_results(moe_result, rag_result)
```

### WebSocketストリーミング
```python
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for chunk in generate_stream():
        await websocket.send_json(chunk)
```

## メモリ最適化パターン

### 動的量子化
```python
def load_model_optimized(model_path: str, model_size: int):
    if model_size > 20_000:  # 20B以上
        return load_4bit_quantized(model_path)
    elif model_size > 7_000:  # 7B以上
        return load_8bit_quantized(model_path)
    else:
        return load_full_precision(model_path)
```

### Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
# メモリ使用量を削減（計算時間は増加）
```

## エラーハンドリングパターン

### Circuit Breaker
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        if self.is_open:
            raise ServiceUnavailableError()
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.is_open = True
            raise
```

## データフローパターン

### Pub/Sub（訓練進捗通知）
```python
class TrainingEventBus:
    def __init__(self):
        self.subscribers = []
    
    def subscribe(self, callback):
        self.subscribers.append(callback)
    
    def publish(self, event):
        for subscriber in self.subscribers:
            subscriber(event)
```

## セキュリティパターン

### 環境変数による設定管理
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_token: str = Field(..., env="HF_TOKEN")
    api_key: str = Field(..., env="API_KEY")
    
    class Config:
        env_file = ".env"
```

## キャッシング戦略

### LRUキャッシュ（モデル推論）
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_inference(model_id: str, input_text: str):
    return model.generate(input_text)
```

### Redisキャッシュ（RAG結果）
```python
async def get_rag_result(query: str):
    # Redisから確認
    cached = await redis.get(f"rag:{query}")
    if cached:
        return json.loads(cached)
    
    # 新規計算
    result = await compute_rag(query)
    await redis.setex(f"rag:{query}", 3600, json.dumps(result))
    return result
```