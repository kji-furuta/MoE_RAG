# 統合コマンドリファレンス

## 目次
- [クイックスタート](#-クイックスタート)
- [Docker環境管理](#-docker環境管理)
- [Webサーバー管理](#-webサーバー管理)
- [テスト実行](#-テスト実行)
- [RAG操作](#-rag操作)
- [モデル訓練](#-モデル訓練)
- [開発ツール](#️-開発ツール)
- [システム管理](#-システム管理)
- [デバッグコマンド](#-デバッグコマンド)
- [トラブルシューティング](#-トラブルシューティング)
- [便利なエイリアス](#-便利なエイリアス)

## 🚀 クイックスタート
```bash
# 最小限の起動手順
cd docker && docker-compose up -d
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
# ブラウザで http://localhost:8050 にアクセス
```

## 🐳 Docker環境管理
```bash
# 完全環境構築（初回・大規模更新時）
./scripts/docker_build_rag.sh --no-cache

# 通常のビルド
./scripts/docker_build_rag.sh

# Docker Compose操作
cd docker
docker-compose up -d --build    # 起動
docker-compose down             # 停止
docker-compose logs -f          # ログ監視
docker-compose ps               # 状態確認

# コンテナ操作
docker exec -it ai-ft-container bash    # コンテナに入る
docker restart ai-ft-container          # 再起動
docker stats ai-ft-container            # リソース監視
```

## 🌐 Webサーバー管理
```bash
# 推奨: スクリプト経由で起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# デバッグ: uvicorn直接起動
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload

# ヘルスチェック
curl http://localhost:8050/health
curl http://localhost:8050/rag/health

# プロセス確認
netstat -tlnp | grep 8050
ps aux | grep uvicorn
```

## 🧪 テスト実行
```bash
# 基本テスト
pytest tests/ -v                                    # 全テスト
pytest tests/test_training.py -v                    # 特定テスト
pytest tests/ -k "test_lora" -v                    # キーワード指定

# 統合テスト
python scripts/test/test_integration.py             # API統合テスト
python scripts/test/test_docker_rag.py             # Docker RAGテスト
python scripts/test/test_memory_optimization.py     # メモリ最適化テスト

# モデルテスト
python scripts/test/test_japanese_simple.py         # 日本語モデル簡易テスト
python scripts/test/simple_lora_tutorial.py         # LoRAチュートリアル

# 設定テスト
python scripts/test_config_resolution.py           # 設定解決テスト
python scripts/test_model_path_resolution.py       # モデルパス解決テスト
```

## 📚 RAG操作
```bash
# ドキュメントインデックス作成
python scripts/rag/index_documents.py

# RAGクエリテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "設計速度80km/hの道路の最小曲線半径は？",
       "top_k": 5,
       "search_type": "hybrid"
     }'

# ドキュメントアップロード
curl -X POST "http://localhost:8050/rag/upload" \
     -F "file=@path/to/document.pdf"

# 検索履歴取得
curl "http://localhost:8050/rag/search-history?page=1&limit=10"

# ドキュメント一覧
curl "http://localhost:8050/rag/documents"
```

## 🤖 モデル訓練
```bash
# LoRAファインチューニング
python scripts/train_lora.py \
    --model_name "rinna/japanese-gpt2-small" \
    --dataset_path "data/train.jsonl" \
    --output_dir "outputs/lora_model"

# 大規模モデル訓練
python scripts/train_large_model.py            # 汎用大規模モデル
python scripts/train_calm3_22b.py              # CALM3 22B
./scripts/train_32b_model.sh                   # 32Bモデル用スクリプト

# 継続学習
./scripts/run_continual_learning.sh            # EWC継続学習
python scripts/continual_learning/run_tasks.py # タスクベース学習

# 訓練状況確認（API経由）
curl "http://localhost:8050/api/training-status/{task_id}"
```

## 🛠️ 開発ツール
```bash
# コードフォーマット
black src/ app/ --line-length 88        # Pythonコードフォーマット
isort src/ app/ --profile black         # インポート整理

# リント
flake8 src/ app/                        # コードチェック
flake8 --statistics                     # 統計情報付き

# 型チェック（オプション）
mypy src/ --ignore-missing-imports

# GPU監視
nvidia-smi -l 1                         # 1秒ごとに更新
gpustat -i 1                           # より見やすい表示（要インストール）

# メモリ監視
watch -n 1 free -h                      # システムメモリ
docker stats                            # Dockerコンテナのリソース

# ログ監視
docker logs -f ai-ft-container --tail 50    # 最新50行をフォロー
docker logs ai-ft-container 2>&1 | grep -i error    # エラーのみ
tail -f logs/training.log              # 特定ログファイル
```

## 🔧 システム管理
```bash
# プロセス管理
ps aux | grep python                    # Python プロセス確認
kill -9 $(ps aux | grep 'uvicorn' | awk '{print $2}')    # uvicorn強制終了

# ディスク管理
df -h                                   # ディスク使用状況
du -sh /workspace/*                     # ディレクトリサイズ
find /workspace -name "*.pyc" -delete   # キャッシュ削除

# Docker管理
docker system df                        # Docker使用状況
docker system prune -a                  # 不要なリソース削除
docker volume prune                     # 不要なボリューム削除

# ネットワーク確認
curl http://localhost:8050/docs        # FastAPI ドキュメント
curl http://localhost:6333              # Qdrant UI
```

## 🐞 デバッグコマンド
```bash
# Pythonインタラクティブ環境
docker exec -it ai-ft-container python

# インポートテスト
docker exec ai-ft-container python -c "import app.main_unified; print('OK')"

# 環境変数確認
docker exec ai-ft-container env | grep -E "HF_|WANDB_|CUDA_"

# パス確認
docker exec ai-ft-container python -c "import sys; print('\\n'.join(sys.path))"

# モジュール一覧
docker exec ai-ft-container pip list | grep -E "torch|transformers|peft"
```

## 🚨 トラブルシューティング
```bash
# メモリ不足時
echo 1 > /proc/sys/vm/drop_caches      # キャッシュクリア（要root）
docker restart ai-ft-container          # コンテナ再起動

# GPU メモリエラー時
nvidia-smi --gpu-reset                  # GPU リセット（要注意）
docker exec ai-ft-container python -c "import torch; torch.cuda.empty_cache()"

# ポート競合時
sudo lsof -i :8050                     # 使用プロセス確認
sudo fuser -k 8050/tcp                 # プロセス強制終了

# モジュールエラー時
docker exec ai-ft-container pip install -r requirements.txt --force-reinstall
```

## 📝 便利なエイリアス（.bashrcに追加）
```bash
alias dft='docker exec -it ai-ft-container'
alias dftb='docker exec -it ai-ft-container bash'
alias dftlog='docker logs -f ai-ft-container --tail 50'
alias dftstop='docker-compose -f docker/docker-compose.yml down'
alias dftstart='docker-compose -f docker/docker-compose.yml up -d'
```