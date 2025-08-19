# 推奨コマンド集

## 🚀 システム起動

### Docker環境の起動（推奨）
```bash
# 完全な環境構築（初回）
./scripts/docker_build_rag.sh --no-cache

# Docker Composeで起動
cd docker && docker-compose up -d

# Webインターフェース起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 直接起動（開発用）
```bash
# 統合サーバー起動
python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload

# MoE-RAG API起動
python app/moe_rag_api.py
```

## 📚 モデル訓練

### MoEモデル訓練
```bash
# Dockerコンテナ内で実行
docker exec ai-ft-container bash scripts/moe/train_moe.sh demo 1 2

# 直接実行
python scripts/moe/run_training.py \
  --model_name "cyberagent/calm3-22b-chat" \
  --num_epochs 3 \
  --batch_size 4
```

### LoRAファインチューニング
```bash
python scripts/test/simple_lora_tutorial.py
```

### 継続学習
```bash
bash scripts/run_continual_learning.sh
```

## 🔍 RAGシステム

### 文書インデックス作成
```bash
python scripts/rag/index_documents.py
```

### RAGクエリテスト
```bash
# API経由
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'

# ヘルスチェック
curl "http://localhost:8050/rag/health"
```

## 🧪 テスト実行

### 統合テスト
```bash
python scripts/test_integration.py
python scripts/test_docker_rag.py
python scripts/test_moe_rag_integration.py
```

### 単体テスト
```bash
# pytestで実行
pytest tests/

# 特定のテストファイル
pytest tests/test_moe_architecture.py -v
```

## 📊 監視・診断

### システム診断
```bash
python scripts/system_diagnosis.py
python scripts/system_status_report.py
```

### GPU監視
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### ディスク使用量管理
```bash
python scripts/disk_space_manager.py
```

## 🛠️ 開発ツール

### コードフォーマット
```bash
# Black実行
black src/ app/ --line-length 88

# isort実行
isort src/ app/ --profile black
```

### リント
```bash
flake8 src/ app/ --max-line-length 88
```

## 🐳 Docker操作

### コンテナ管理
```bash
# コンテナ一覧
docker ps -a

# ログ確認
docker logs ai-ft-container --tail 50 -f

# コンテナ内でシェル起動
docker exec -it ai-ft-container /bin/bash

# コンテナ再起動
docker-compose restart
```

### クリーンアップ
```bash
# 停止と削除
docker-compose down

# ボリューム含めて削除
docker-compose down -v

# 未使用イメージ削除
docker image prune -a
```

## 📦 モデル管理

### Ollamaモデル
```bash
# モデル一覧
ollama list

# モデルダウンロード
ollama pull llama3.2:3b

# モデル実行
ollama run llama3.2:3b
```

### 利用可能モデル確認
```bash
python scripts/check_available_models.py
```

## 🔧 環境設定

### 環境変数設定
```bash
cp .env.example .env
# .envファイルを編集
```

### パーミッション修正
```bash
bash scripts/setup_permissions.sh
```

## Linux システムコマンド

### ファイル操作
```bash
ls -la            # 詳細表示
find . -name "*.py"  # ファイル検索
grep -r "pattern" .  # パターン検索
```

### プロセス管理
```bash
ps aux | grep python
kill -9 <PID>
htop              # リソース監視
```

### Git操作
```bash
git status
git add .
git commit -m "feat: 機能追加"
git push origin main
```