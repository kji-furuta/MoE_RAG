# 開発コマンド集 (2025年9月18日更新)

## 環境構築・起動

### Docker環境
```bash
# 完全ビルド（推奨）
./scripts/docker_build_rag.sh --no-cache

# コンテナ起動
docker-compose up -d

# Webインターフェース起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# 直接サーバー起動（デバッグ用）
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```

### Ollama起動
```bash
# Ollamaサービス起動
ollama serve

# モデル取得
ollama pull llama3.2:3b
ollama pull deepseekrag:latest

# モデル確認
ollama list
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

## RAG操作

```bash
# 文書インデックス作成
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
# 大規模モデル訓練
python scripts/train_large_model.py
python scripts/train_calm3_22b.py

# LoRAファインチューニング
python scripts/test/simple_lora_tutorial.py

# 継続学習
python src/training/continual_learning_pipeline.py
```

## 開発ツール

### コード品質
```bash
# フォーマット
black app/ src/ scripts/
ruff check app/ src/

# リンティング
pylint app/main_unified.py
mypy app/main_unified.py
```

### Git操作
```bash
# ステータス確認
git status
git branch

# コミット（Co-authored付き）
git commit -m "feat: 機能追加

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# プッシュ
git push origin rag-development-20250901
```

### デバッグ
```bash
# ログ確認
docker logs -f ai-ft-container --tail 100

# コンテナ内でシェル
docker exec -it ai-ft-container bash

# GPU状態確認
nvidia-smi
watch -n 1 nvidia-smi

# メモリ確認
free -h
df -h
```

## トラブルシューティング

```bash
# ポート確認
netstat -tlnp | grep 8050
lsof -i :8050

# プロセス確認
ps aux | grep uvicorn
ps aux | grep python

# キャッシュクリア
rm -rf ~/.cache/huggingface/
docker exec ai-ft-container rm -rf /root/.cache/

# GPU リセット
sudo nvidia-smi --gpu-reset

# Docker クリーンアップ
docker system prune -a --volumes
```

## モデル変換

```bash
# LoRA → GGUF変換
python scripts/convert/apply_lora_to_gguf.py \
  --lora-path outputs/lora_20250918_135106 \
  --output outputs/model.gguf

# Ollama登録
ollama create mymodel -f outputs/model.gguf
```

## 監視・メトリクス

```bash
# システム情報取得
curl "http://localhost:8050/rag/system-info"

# メトリクス確認
curl "http://localhost:8050/metrics"

# タスク状態確認
curl "http://localhost:8050/api/continual/tasks"
```