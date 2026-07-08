# MoE-RAG アーキテクチャ詳細

## システム全体構成

```
┌─────────────────────────────────────────────────────────┐
│                   Web Browser (Client)                   │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/WebSocket
┌─────────────────▼───────────────────────────────────────┐
│              FastAPI Server (Port 8050)                 │
│                 app/main_unified.py                     │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Fine-tune  │  │   RAG    │  │ Continual Learn  │   │
│  │   Routes   │  │  Routes  │  │     Routes       │   │
│  └──────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
└─────────┼──────────────┼─────────────────┼─────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼─────────────┐
│                  Core Services Layer                    │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐    │
│  │  Training   │  │    RAG     │  │     MoE      │    │
│  │   Engine    │  │   Engine   │  │   Manager    │    │
│  └─────────────┘  └────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼─────────────┐
│              Infrastructure Layer                       │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐    │
│  │   Models    │  │   Qdrant   │  │    Ollama    │    │
│  │  Storage    │  │   Vector   │  │   Service    │    │
│  │             │  │     DB     │  │ (Port 11434) │    │
│  └─────────────┘  └────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────┘
```

## レイヤー別詳細

### 1. プレゼンテーション層
**場所**: `templates/`, `app/static/`

- **HTMLテンプレート**: Jinja2ベース
  - `base.html` - 基本レイアウト
  - `index.html` - ダッシュボード
  - `finetune.html` - ファインチューニングUI
  - `rag.html` - RAG検索UI
  - `continual_learning.html` - 継続学習UI

- **静的リソース**:
  - Bootstrap CSS/JS
  - カスタムJavaScript
  - ロゴ・画像アセット

### 2. APIルーティング層
**場所**: `app/main_unified.py`, `app/routers/`

- **メインルーター**: FastAPIアプリケーション
- **エンドポイントグループ**:
  - `/api/*` - ファインチューニングAPI
  - `/rag/*` - RAG API
  - `/api/continual/*` - 継続学習API
  - `/api/moe/*` - MoE API

### 3. ビジネスロジック層
**場所**: `src/`

#### ファインチューニングサービス (`src/training/`)
```python
# 主要クラス
- LoRATrainer
- DoRAImplementation
- EWCContinualLearner
- MultiGPUTrainingManager
```

#### RAGサービス (`src/rag/`)
```python
# コアコンポーネント
- QueryEngine (core/query_engine.py)
- VectorStore (indexing/vector_store.py)
- HybridSearch (retrieval/hybrid_search.py)
- DocumentProcessor (document_processing/)
```

#### 推論サービス (`src/inference/`)
```python
# 推論エンジン
- VLLMIntegration
- AWQQuantizer
- ModelLoader
```

#### MoEサービス (`src/moe/`)
```python
# MoEコンポーネント
- ExpertRouter
- MoEModel
- DatasetManager
```

### 4. データアクセス層
**場所**: Various

#### ベクトルストア
- **Qdrant**: ポート6333
- **接続管理**: `src/rag/indexing/vector_store.py`
- **スキーマ**: UUID-based points

#### モデルストレージ
- **ローカル**: `outputs/`ディレクトリ
- **HuggingFace Hub**: API統合
- **Ollama**: ローカルモデル管理

#### メタデータ管理
- **タスク状態**: `data/continual_learning/tasks_state.json`
- **EWCデータ**: `outputs/ewc_data/`
- **検索履歴**: SQLite/JSON

### 5. インフラストラクチャ層
**場所**: `docker/`, システム設定

#### Dockerコンテナ
```yaml
services:
  ai-ft-container:
    - FastAPIサーバー
    - 全サービス統合
    - GPU対応
  
  qdrant:
    - ベクトルDB
    - ポート6333
  
  ollama:
    - LLMサービス
    - ポート11434
```

#### リソース管理
- **GPU割り当て**: NVIDIA Container Toolkit
- **メモリ管理**: 動的量子化、CPU offloading
- **プロセス管理**: Uvicorn workers

## データフロー

### 1. ファインチューニングフロー
```
User Request → API → Training Manager → Model Loader
    ↓                                        ↓
Response ← Progress Updates ← Task Queue ← GPU Training
```

### 2. RAGクエリフロー
```
Query → Embedding → Vector Search → Hybrid Ranking
   ↓        ↓            ↓              ↓
Response ← Generation ← Context ← Retrieved Docs
```

### 3. 継続学習フロー
```
New Task → EWC Calculator → Fisher Matrix
    ↓           ↓               ↓
Model Update ← Training ← Regularization
```

## 設定管理

### 環境変数
```bash
HF_TOKEN          # HuggingFace認証
WANDB_API_KEY     # Weights & Biases
CUDA_VISIBLE_DEVICES  # GPU選択
MODEL_CACHE_DIR   # モデルキャッシュ
```

### 設定ファイル
- `config/model_config.yaml` - モデル設定
- `src/rag/config/rag_config.yaml` - RAG設定
- `configs/` - 各種設定テンプレート

## セキュリティアーキテクチャ

### 認証・認可
- JWT準備中（現在は基本認証なし）
- APIキー管理
- CORS制限: localhost:8050のみ

### データ保護
- 入力検証: Pydantic
- ファイルアップロード制限
- SQLインジェクション対策

## スケーラビリティ

### 水平スケーリング
- マルチワーカー対応
- 負荷分散準備
- キャッシュ戦略

### 垂直スケーリング
- マルチGPU対応
- メモリ最適化
- バッチ処理最適化

## 監視・ログ

### ログ管理
- Python標準ログ
- ローテーション設定
- エラートラッキング

### メトリクス
- `/metrics`エンドポイント
- GPU使用率監視
- レスポンスタイム計測

## デプロイメントパターン

### 開発環境
```bash
./scripts/docker_build_rag.sh --no-cache
docker exec ai-ft-container python -m uvicorn app.main_unified:app --reload
```

### 本番環境
```bash
docker-compose up -d
# Gunicorn/複数ワーカー
# ロードバランサー統合
```

## 今後の拡張ポイント

1. **マイクロサービス化**: 各コンポーネントの独立デプロイ
2. **Kubernetes対応**: オーケストレーション
3. **API Gateway**: 統一エントリーポイント
4. **メッセージキュー**: 非同期処理強化
5. **分散キャッシュ**: Redis統合