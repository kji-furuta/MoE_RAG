# 開発環境セットアップガイド

## 環境構築手順

### 1. WSL2環境の準備（Windows環境の場合）
```bash
# WSL2のインストール
wsl --install

# Ubuntu環境の設定
wsl -d Ubuntu

# 必要なパッケージのインストール
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3-pip git docker.io docker-compose
```

### 2. プロジェクトのクローン
```bash
cd /home/$USER
git clone [repository_url] AI_FT
cd AI_FT/AI_FT_3
```

### 3. Python環境の構築
```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
pip install -r requirements_rag.txt

# 開発ツールのインストール
pip install black flake8 isort pytest
```

### 4. Docker環境の構築
```bash
# Dockerビルド（完全構築）
./scripts/docker_build_rag.sh --no-cache

# Docker Composeで起動
cd docker
docker-compose up -d --build

# コンテナの確認
docker ps
docker exec -it ai-ft-container bash
```

### 5. 環境変数の設定
```bash
# .envファイルの作成
cat > .env << EOF
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# Dockerコンテナ内での確認
docker exec ai-ft-container env | grep -E "HF_TOKEN|WANDB_API_KEY"
```

### 6. GPU環境の確認
```bash
# ホストでのGPU確認
nvidia-smi

# コンテナ内でのGPU確認
docker exec ai-ft-container nvidia-smi

# PyTorchでのGPU確認
docker exec ai-ft-container python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 7. Webインターフェースの起動
```bash
# 方法1: スクリプトを使用
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# 方法2: 直接uvicornを起動（デバッグ用）
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload

# アクセス確認
curl http://localhost:8050/health
```

### 8. Qdrant（ベクトルDB）の確認
```bash
# Qdrantコンテナの状態確認
docker ps | grep qdrant

# 接続テスト
curl http://localhost:6333/collections
```

## トラブルシューティング

### ポート競合の解決
```bash
# 使用中のポートを確認
sudo netstat -tlnp | grep 8050

# プロセスの終了
sudo kill -9 [PID]
```

### メモリ不足の対処
```bash
# Dockerのメモリ制限を確認
docker stats

# スワップの追加（必要な場合）
sudo dd if=/dev/zero of=/swapfile bs=1G count=16
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### モジュールインポートエラー
```bash
# パスの確認
docker exec ai-ft-container python -c "import sys; print('\\n'.join(sys.path))"

# 手動でパスを追加
docker exec ai-ft-container python -c "import sys; sys.path.append('/workspace'); import app.main_unified"
```