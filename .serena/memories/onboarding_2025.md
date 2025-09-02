# MoE-RAG Project Onboarding Guide - 2025

## プロジェクト概要
**AI Fine-tuning Toolkit (AI_FT_7)** - 土木工学・道路設計特化型の日本語LLMファインチューニング＆RAGプラットフォーム

- **統合Webインターフェース**: ポート8050で動作する単一のFastAPIアプリケーション
- **主要機能**: ファインチューニング、RAG、継続学習、MoE訓練
- **GitHubリポジトリ**: https://github.com/kji-furuta/MoE_RAG.git (remote: moe_rag)

## ディレクトリ構造

```
/home/kjifu/MoE_RAG/
├── app/                    # Webアプリケーション
│   ├── main_unified.py     # メインFastAPIサーバー (port 8050)
│   ├── static/            # 静的ファイル
│   ├── routers/           # APIルーター
│   ├── continual_learning/ # 継続学習UI
│   └── monitoring/        # モニタリング機能
├── src/                   # コアシステム
│   ├── training/          # ファインチューニング
│   ├── rag/              # RAGシステム
│   ├── inference/        # 推論エンジン
│   ├── moe/              # MoE実装
│   └── moe_rag_integration/ # 統合レイヤー
├── docker/               # Docker設定
├── scripts/              # ユーティリティスクリプト
├── templates/            # HTMLテンプレート
├── configs/              # 設定ファイル
└── outputs/              # 出力モデル・データ
```

## 主要エントリーポイント

### メインサーバー
- **ファイル**: `app/main_unified.py`
- **起動コマンド**: `python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`
- **Docker起動**: `./scripts/docker_build_rag.sh --no-cache`

### Web UI ルート
- `/` - メインダッシュボード
- `/finetune` - ファインチューニング
- `/rag` - RAG検索
- `/continual` - 継続学習
- `/models` - モデル管理

## コア機能モジュール

### 1. ファインチューニングシステム (`src/training/`)
- `lora_finetuning.py` - LoRAファインチューニング
- `full_finetuning.py` - フルファインチューニング
- `dora/dora_implementation.py` - DoRA実装
- `ewc_utils.py` - EWC継続学習
- `multi_gpu_training.py` - マルチGPU対応
- `continual_learning_pipeline.py` - 継続学習パイプライン

### 2. RAGシステム (`src/rag/`)
- `core/query_engine.py` - クエリエンジン
- `indexing/vector_store.py` - Qdrantベクトルストア
- `retrieval/hybrid_search.py` - ハイブリッド検索
- `document_processing/` - 文書処理
- `specialized/` - 特化機能

### 3. 推論システム (`src/inference/`)
- `vllm_integration.py` - vLLM統合
- `awq_quantization.py` - AWQ量子化

### 4. MoEシステム (`src/moe/`)
- エキスパートモデル管理
- ルーティング機構
- 訓練・推論統合

## API構成

### ファインチューニングAPI
- `POST /api/train` - 訓練開始
- `GET /api/training-status/{task_id}` - ステータス確認
- `POST /api/generate` - テキスト生成
- `GET /api/models` - モデル一覧

### RAG API
- `POST /rag/query` - 文書検索
- `POST /rag/upload-document` - 文書アップロード
- `GET /rag/documents` - 文書一覧
- `POST /rag/stream-query` - ストリーミング検索

### 継続学習API
- `POST /api/continual/train` - タスク開始
- `GET /api/continual/tasks` - タスク一覧
- `GET /api/continual/task/{task_id}` - タスク状態

### MoE API
- `POST /api/moe/training/start` - MoE訓練開始
- `GET /api/moe/training/status/{task_id}` - 訓練状態
- `POST /api/moe/dataset/update` - データセット更新

## 重要な実装詳細

### メモリ最適化
- 32B/22Bモデル: 4ビット量子化
- 7B/8Bモデル: 8ビット量子化
- CPU offloading対応
- AWQ量子化で75%メモリ削減

### RAGベクトルストア
- Qdrant使用（UUID-based point IDs）
- ハイブリッド検索（ベクトル0.7 + キーワード0.3）
- multilingual-e5-large埋め込み
- チャンクサイズ512トークン（128オーバーラップ）

### 継続学習
- EWCベース（Fisher Information Matrix）
- デフォルトλ: 5000
- タスク履歴: `outputs/ewc_data/task_history.json`
- モデル保存: `outputs/continual_task_*`

### Ollama統合
- Llama 3.2 3Bモデル対応
- ポート11434で動作
- `ollama serve`で起動
- `ollama pull llama3.2:3b`でモデル取得

## クリティカルファイル

1. **メインサーバー**: `app/main_unified.py`
2. **RAGエンジン**: `src/rag/core/query_engine.py`
3. **ベクトルストア**: `src/rag/indexing/vector_store.py`
4. **LoRA訓練**: `src/training/lora_finetuning.py`
5. **継続学習**: `src/training/continual_learning_pipeline.py`
6. **MoE統合**: `src/moe_rag_integration/`

## 開発ワークフロー

1. **Docker環境使用推奨**
2. **テストスクリプト実行**: `scripts/test_*.py`
3. **ログ監視**: `docker logs -f ai-ft-container`
4. **GPU確認**: `nvidia-smi`
5. **モデルバックアップ**: `outputs/`ディレクトリ

## トラブルシューティング

### ポート8050が使用中
```bash
netstat -tlnp | grep 8050
```

### RAGエラー
```bash
python scripts/test_docker_rag.py
```

### GPU/メモリ不足
- バッチサイズ削減
- CPU offloading有効化
- 量子化使用

### Ollama接続エラー
```bash
curl http://localhost:11434/api/tags
ollama list
ollama pull llama3.2:3b
```

## 重要な注意事項

- **統一APIポート8050**を使用
- **Dockerコンテナ**での開発推奨
- **Ollama**は別ターミナルで起動維持
- **継続学習タスク**は`tasks_state.json`で管理
- **セキュリティ**: CORS制限、ファイル検証実装済み