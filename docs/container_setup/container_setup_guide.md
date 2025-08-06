# AI_FT_3 コンテナ環境セットアップガイド

生成日時: 2025-08-04T11:54:08.461136

## 1. コンテナ内ファイル構造

### ワークスペースルート (/workspace)
```
total 88
drwxr-xr-x  1 ai-user ai-user 4096 Aug  3 00:53 .
drwxr-xr-x  1 root    root    4096 Aug  4 01:57 ..
-rwxr-xr-x  1 ai-user ai-user    0 Aug  2 13:37 =2.0.0
drwxr-xr-x  4 ai-user ai-user 4096 Aug  2 00:55 app
drwxr-xr-x  2 ai-user ai-user 4096 Jul 24 14:30 config
drwxr-xr-x  7 ai-user ai-user 4096 Aug  1 23:29 data
drwxr-xr-x  4 ai-user ai-user 4096 Aug  4 02:54 docs
drwxr-xr-x  2 ai-user ai-user 4096 Jul 24 14:30 examples
drwxr-xr-x  2 ai-user ai-user 4096 Jul 27 01:27 logs
drwxrwxrwx  2 ai-user ai-user 4096 Jul 30 13:54 metadata
drwxr-xr-x  2 ai-user ai-user 4096 Jul 24 14:35 models
drwxr-xr-x  2 ai-user ai-user 4096 Jul 24 14:35 notebooks
drwxr-xr-x 16 ai-user ai-user 4096 Aug  3 21:31 outputs
-rwxr-xr-x  1 ai-user ai-user  843 Jul 24 14:30 pyproject.toml
drwxr-xr-x  2 ai-user ai-user 4096 Jul 27 13:56 qdrant_data
-rwxr-xr-x  1 ai-user ai-user  618 Jul 24 14:30 requirements.txt
-rwxr-xr-x  1 ai-user ai-user 1462 Jul 27 05:31 requirements_rag.txt
drwxr-xr-x  7 ai-user ai-user 4096 Jul 31 08:06 scripts
drwxr-xr-x  8 ai-user ai-user 4096 Jul 31 06:33 src
-rwxr-xr-x  1 ai-user ai-user  339 Aug  1 01:54 start_server.sh
drwxr-xr-x  2 ai-user ai-user 4096 Aug  1 00:21 temp_uploads
drwxr-xr-x  1 ai-user ai-user 4096 Aug  3 00:53 templates
drwxr-xr-x  2 ai-user ai-user 4096 Jul 24 14:35 tests
```

### 存在するディレクトリ
- ✓ /workspace/src
- ✓ /workspace/app
- ✓ /workspace/scripts
- ✓ /workspace/data
- ✓ /workspace/outputs

### 重要ファイルの状態
- ✓ requirements.txt
- ✓ requirements_rag.txt
- ✗ docker-compose.yml (要作成)
- ✓ app/main_unified.py
- ✓ scripts/rag/index_documents.py

## 2. インストール済みパッケージ
- accelerate: 1.9.0
- fastapi: 0.116.1
- langchain: 0.1.7
- langchain-community: 0.0.20
- langchain-core: 0.1.23
- opentelemetry-instrumentation-fastapi: 0.57b0
- peft: 0.16.0
- qdrant-client: 1.7.3
- sentence-transformers: 2.3.1
- torch: 2.5.1
- torchaudio: 2.7.1+cu126
- torchelastic: 0.2.2
- torchvision: 0.20.1
- transformers: 4.54.1

## 3. ボリュームマウント設定
| ホスト | コンテナ | タイプ |
|--------|----------|--------|
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/374fa7fe65e6652c5e461e307cebb456b4baa615b2ad61b305b8f3941e5eb281 | /workspace/data | bind |
| /var/lib/docker/volumes/docker_ai_ft_rag_processed/_data | /workspace/outputs/rag_index | volume |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/1d55828d1e453f93dba56bdea6a7483a1d04a28eafb95eccaead4b24cb3521b9 | /workspace/logs | bind |
| ./examples | /workspace/examples | bind |
| /var/lib/docker/volumes/docker_ai_ft_rag_metadata/_data | /workspace/metadata | volume |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/a5b10350c884f4a91496df3ac448ef2ece4d69b43ba721eab553a1120f1c2994 | /workspace/src | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/13fd3f638b9277998776c3933c1213572cbd950773da027240c8f2b50f9c64a4 | /workspace/app | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/129999662200b16d8c15f41c6bf3b504287c00f5121c0d0f6e9336234fbbfb20 | /workspace/scripts | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/c8d15e26c06793c7ea06875a14e2d4c51049dedbd033279f3222ef947d74b758 | /root/.cache/huggingface | bind |
| ./qdrant_data | /workspace/qdrant_data | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/c9e3956c83279781fc6af3f39e29fcc4c442ebeefc2ce9c06aca90b8daa599d1 | /workspace/tests | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/10307fd3ed5a5f11a3b4757ee22bcb25728daf436871948f9886e46d2ed6cffe | /workspace/docs | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/e70b37e597beef9a1fb786e8091c8f3231ce9645ef09f7ece7f2e7178fee4c46 | /root/.wandb | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/39ea398362eddd7705fbe70372fb3c4303cc9257d4943c7931e5887915bd5f6a | /workspace/temp_uploads | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/15c02cdda4eb76b756a542e3bd2ee52d47053f8d7b9bcd8d1d3b18543b1e69ff | /workspace/outputs | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/61c22e57917cbbb8dca4731c80a6db94dfb58d4e385facd02c4b551cbe050af0 | /workspace/models | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/06a991fc22255c6b98609a5fff61cef0f2009f6d90ba36086412b42587f2d310 | /workspace/config | bind |
| /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/85cd271160e25e8e77005785ac2bede2db868b3b11b0b441e4434817199a21a3 | /workspace/notebooks | bind |

## 4. 環境構築手順

### 必要なファイルとディレクトリ

1. **プロジェクト構造の作成**
```bash
mkdir -p src app configs scripts data outputs docker
mkdir -p data/{raw,processed,uploaded,rag_documents}
mkdir -p outputs/rag_index
```

2. **必要なファイルの配置**
- `requirements.txt` - Python依存関係
- `docker/docker-compose.yml` - Docker構成
- `docker/Dockerfile` - コンテナイメージ定義
- `app/main_unified.py` - メインアプリケーション

3. **環境変数の設定**
```bash
# docker/.env ファイルを作成
HF_TOKEN=your_token_here
CUDA_VISIBLE_DEVICES=0,1
```

4. **コンテナの起動**
```bash
cd docker
docker-compose up -d
```

## 5. 動作確認

```bash
# コンテナにアクセス
docker exec -it ai-ft-container bash

# GPU確認
python -c "import torch; print(torch.cuda.is_available())"

# Webアプリケーション起動
python app/main_unified.py
```
