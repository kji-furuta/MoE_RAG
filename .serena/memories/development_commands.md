# 開発コマンドリファレンス

## Docker環境管理

### 環境起動・停止
```bash
# 開発環境の起動（推奨）
./start_dev_env.sh

# 開発環境の停止
./stop_dev_env.sh

# Docker Compose操作
cd docker
docker-compose up -d --build  # 初回ビルド
docker-compose up -d           # 通常起動
docker-compose down            # 停止
docker-compose down -v         # ボリューム含む完全削除
```

### コンテナ操作
```bash
# コンテナ内にアクセス
docker exec -it ai-ft-container bash

# rootユーザーでアクセス
docker exec -u root ai-ft-container bash

# ログ確認
docker logs ai-ft-container --tail 50
docker logs -f ai-ft-container  # リアルタイム

# コンテナステータス確認
docker ps -a | grep -E "ai-ft|qdrant"
```

## Webサービス起動

```bash
# 統合Webインターフェース起動
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload

# 個別サービス起動（開発用）
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

## RAGシステム操作

```bash
# ドキュメントインデックス作成
python scripts/rag/index_documents.py

# RAGクエリテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'

# ヘルスチェック
curl "http://localhost:8050/rag/health"

# システム情報取得
curl "http://localhost:8050/rag/system-info"
```

## モデル学習

```bash
# LoRAファインチューニング
python scripts/test/simple_lora_tutorial.py

# 大規模モデル学習
python scripts/train_large_model.py
python scripts/train_calm3_22b.py

# 継続学習
python src/training/continual_learning_pipeline.py
```

## テスト実行

```bash
# 統合テスト
python scripts/test_integration.py
python scripts/test_docker_rag.py
python scripts/test_continual_learning_integration.py

# 設定テスト
python scripts/test_config_resolution.py
python scripts/test_model_path_resolution.py

# 機能テスト
python scripts/simple_feature_test.py
python scripts/test_specialized_features.py
```

## Ollama操作

```bash
# Ollamaサービス起動
ollama serve

# モデルダウンロード
ollama pull llama3.2:3b

# モデル一覧
ollama list

# API確認
curl http://localhost:11434/api/tags
```

## Git操作

```bash
# ステータス確認
git status

# 変更をステージング
git add .

# コミット
git commit -m "feat: 機能追加"

# プッシュ（originリモート）
git push origin main

# ブランチ操作
git checkout -b feature/new-feature
git checkout main
git merge feature/new-feature
```

## トラブルシューティング

```bash
# ポート使用状況確認
netstat -tlnp | grep 8050

# GPU状況確認
nvidia-smi

# メモリ確認
free -h

# ディスク容量確認
df -h

# プロセス確認
ps aux | grep python

# 権限修正（コンテナ内）
docker exec -u root ai-ft-container chmod -R 777 /workspace/logs
docker exec -u root ai-ft-container chown -R ai-user:ai-user /workspace
```