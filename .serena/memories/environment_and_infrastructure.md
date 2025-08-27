# 環境とインフラストラクチャ

## システム環境

### ホスト環境
- **OS**: WSL2 (Windows Subsystem for Linux 2)
- **ディストリビューション**: Ubuntu
- **Python**: 3.12.3
- **Docker**: 28.3.2
- **Docker Compose**: v2

### コンテナ環境
- **ベースイメージ**: pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
- **Python**: 3.11.13 (コンテナ内)
- **CUDA**: 12.6
- **cuDNN**: 9
- **ユーザー**: ai-user (UID: 1000)

## ネットワーク構成

### ポートマッピング
```yaml
ports:
  - "8050:8050"  # 統合Webインターフェース
  - "8051:8051"  # RAG API（開発用）
  - "6333:6333"  # Qdrant Web UI
  - "6334:6334"  # Qdrant gRPC
  - "8888:8888"  # Jupyter Lab
  - "6006:6006"  # TensorBoard
  - "11434:11434" # Ollama API
```

### サービス構成
```yaml
services:
  ai-ft:        # メインアプリケーションコンテナ
  qdrant:       # ベクトルデータベース
  tensorboard:  # モニタリング（オプション）
  jupyter:      # 開発環境（オプション）
```

## ストレージ構成

### ボリュームマッピング
```yaml
volumes:
  # アプリケーションコード
  - ../src:/workspace/src
  - ../app:/workspace/app
  - ../templates:/workspace/templates
  - ../scripts:/workspace/scripts
  
  # データとモデル
  - ../data:/workspace/data
  - ../models:/workspace/models
  - ../outputs:/workspace/outputs
  
  # RAG関連
  - ../qdrant_data:/workspace/qdrant_data
  - ../temp_uploads:/workspace/temp_uploads
  - ../docs:/workspace/docs
  
  # キャッシュ
  - ~/.cache/huggingface:/home/ai-user/.cache/huggingface
  - ~/.wandb:/home/ai-user/.wandb
  
  # 永続化ボリューム
  - ai_ft_rag_metadata:/workspace/metadata
  - ai_ft_rag_processed:/workspace/outputs/rag_index
  - ollama_models:/usr/share/ollama/.ollama/models
```

### ディレクトリ権限
```bash
/workspace/
├── data/          [775, ai-user:ai-user]
├── logs/          [777, ai-user:ai-user]
├── models/        [775, ai-user:ai-user]
├── outputs/       [777, ai-user:ai-user]
├── qdrant_data/   [777, ai-user:ai-user]
├── temp_uploads/  [777, ai-user:ai-user]
└── metadata/      [777, ai-user:ai-user]
```

## 環境変数

### 必須環境変数
```bash
# Hugging Face
HUGGINGFACE_TOKEN=hf_xxxxx

# OpenAI (RAGで使用)
OPENAI_API_KEY=sk-xxxxx

# システム設定
PYTHONPATH=/workspace
CUDA_VISIBLE_DEVICES=0,1
```

### オプション環境変数
```bash
# Weights & Biases
WANDB_API_KEY=xxxxx
WANDB_PROJECT=ai-ft-7

# MLflow
MLFLOW_TRACKING_URI=mlruns

# ログレベル
LOG_LEVEL=INFO
```

## GPU設定

### NVIDIA Docker設定
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### メモリ設定
- **共有メモリ**: 32GB (`shm_size: '32gb'`)
- **ulimits設定**: 
  - memlock: unlimited
  - stack: 64MB

## セキュリティ設定

### CORS設定
```python
# app/main_unified.py
allow_origins = [
    "http://localhost:8050",
    "http://127.0.0.1:8050"
]
```

### ファイルアップロード制限
- **最大サイズ**: 100MB
- **許可される形式**: PDF, TXT, JSON, CSV
- **一時保存先**: `/workspace/temp_uploads/`

### 認証・認可
- 現在は基本認証なし（開発環境）
- 本番環境では JWT トークン認証を推奨

## モニタリング

### ログ設定
```python
# Loguru設定
logger.add("logs/app.log", rotation="10 MB", retention="7 days")
```

### メトリクス収集
- TensorBoard: トレーニングメトリクス
- Weights & Biases: 実験管理
- Prometheus: システムメトリクス（オプション）

## バックアップとリカバリ

### 重要データのバックアップ対象
```bash
# モデル
/workspace/outputs/
/workspace/models/checkpoints/

# RAGインデックス
/workspace/qdrant_data/
/workspace/outputs/rag_index/

# 設定
/workspace/config/
```

### バックアップコマンド
```bash
# Dockerボリュームバックアップ
docker run --rm -v ai_ft_rag_metadata:/data -v $(pwd):/backup \
  alpine tar czf /backup/metadata_backup.tar.gz -C /data .
```

## 災害復旧

### コンテナ再構築
```bash
# 完全クリーンビルド
cd docker
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### データリストア
```bash
# ボリュームリストア
docker run --rm -v ai_ft_rag_metadata:/data -v $(pwd):/backup \
  alpine tar xzf /backup/metadata_backup.tar.gz -C /data
```