# 推奨コマンド集

## Docker環境管理
```bash
# 完全環境構築（推奨）
./scripts/docker_build_rag.sh --no-cache

# Docker Compose起動
cd docker && docker-compose up -d --build

# Webインターフェース起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# サーバー直接起動（デバッグ用）
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```

## テスト実行
```bash
# 統合テスト
python scripts/test/test_integration.py
python scripts/test/test_docker_rag.py

# 設定テスト
python scripts/test_config_resolution.py
python scripts/test_model_path_resolution.py

# pytest実行
pytest tests/ -v
```

## RAG操作
```bash
# ドキュメントインデックス
python scripts/rag/index_documents.py

# RAGクエリテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'

# ヘルスチェック
curl "http://localhost:8050/rag/health"
```

## モデル訓練
```bash
# LoRAファインチューニング
python scripts/test/simple_lora_tutorial.py

# 大規模モデル訓練
python scripts/train_large_model.py
python scripts/train_calm3_22b.py

# 継続学習
./scripts/run_continual_learning.sh
```

## 開発ツール
```bash
# コードフォーマット
black src/ --line-length 88

# インポート整理
isort src/ --profile black

# リント
flake8 src/

# GPU監視
nvidia-smi -l 1

# ログ監視
docker logs -f ai-ft-container --tail 50
```

## システム管理
```bash
# プロセス確認
netstat -tlnp | grep 8050

# メモリ確認
free -h

# ディスク容量確認
df -h

# Dockerクリーンアップ
docker system prune -a
```