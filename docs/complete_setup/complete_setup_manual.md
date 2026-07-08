# AI_FT_3 完全環境構築マニュアル

生成日時: 2025-08-04T11:34:43.705302

## 1. システム要件

### ハードウェア要件
- GPU: NVIDIA GPU (CUDA 12.6対応)
- メモリ: 32GB以上推奨
- ディスク: 100GB以上の空き容量

### ソフトウェア要件
- Docker Desktop (WSL2バックエンド有効)
- WSL2 (Ubuntu 20.04以降)
- NVIDIA Docker Runtime

## 2. プロジェクト構造

### コンテナ内部構造 (/workspace)
```
Error: find: paths must precede expression: `|'

...
```

### 重要ファイルの確認結果
#### configs- ✗ `/workspace/config/rag_config.yaml`- ✗ `/workspace/configs/available_models.json`- ✗ `/workspace/docker/.env.example`#### scripts- ✗ `/workspace/scripts/rag/index_documents.py`- ✗ `/workspace/app/main_unified.py`- ✗ `/workspace/start_server.sh`#### data_dirs- ✗ `/workspace/data/rag_documents`- ✗ `/workspace/temp_uploads`- ✗ `/workspace/qdrant_data`## 3. Python環境Error:   File "<string>", line 1
    "import
    ^
SyntaxError: unterminated string literal (detected at line 1)
### 主要パッケージ- accelerate: 1.9.0- fastapi: 0.116.1- langchain: 0.1.7- langchain-community: 0.0.20- langchain-core: 0.1.23- opentelemetry-instrumentation-fastapi: 0.57b0- peft: 0.16.0- qdrant-client: 1.7.3- sentence-transformers: 2.3.1- torch: 2.5.1- torchaudio: 2.7.1+cu126- torchelastic: 0.2.2- torchvision: 0.20.1- transformers: 4.54.1## 4. セットアップ手順### 1. リポジトリのクローン```bashgit clone <repository_url>
cd AI_FT_3```### 2. 必要なディレクトリの作成```bashmkdir -p data/{raw,processed,uploaded,rag_documents}
mkdir -p outputs/rag_index/processed_documents
mkdir -p temp_uploads qdrant_data logs docker/logs
mkdir -p models/checkpoints```### 3. 環境変数の設定```bashcp docker/.env.example docker/.env
# docker/.env を編集して HF_TOKEN を設定```### 4. Dockerイメージのビルド```bashcd docker
docker-compose build```### 5. コンテナの起動```bashdocker-compose up -d```### 6. 初期設定の実行```bash# RAGインデックスの作成
docker exec -it ai-ft-container python scripts/rag/index_documents.py```## 5. トラブルシューティング

### コンテナが起動しない場合
1. Docker Desktop が起動していることを確認
2. WSL2 が有効になっていることを確認
3. GPU ドライバーが最新であることを確認

### ファイルが見つからないエラー
1. 必要なディレクトリがすべて作成されているか確認
2. ファイルの権限を確認: `chmod -R 755 .`

### GPU が認識されない場合
```bash
docker exec ai-ft-container python -c "import torch; print(torch.cuda.is_available())"
```

## 6. 動作確認

### Web UI へのアクセス
- http://localhost:8050 - メインインターフェース
- http://localhost:8888 - Jupyter Lab

### コンテナ内での作業
```bash
docker exec -it ai-ft-container bash
```

### ログの確認
```bash
docker logs ai-ft-container
tail -f docker/logs/indexing.log
```
