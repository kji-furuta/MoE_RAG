# 技術スタック詳細 (2025年版)

## コアフレームワーク
- **FastAPI**: v0.104+ - Web API/アプリケーションサーバー
- **PyTorch**: v2.0+ - 深層学習フレームワーク
- **Transformers**: v4.36+ - HuggingFace LLMライブラリ
- **Pydantic**: v2.0+ - データバリデーション

## LLMモデル
### 対応モデル
- **Llama 3.2**: 3B/8B (Ollama経由)
- **CALM3**: 7B/22B (日本語特化)
- **Swallow**: 7B/13B/70B (日本語)
- **DeepSeek**: MoE対応
- **Qwen**: 多言語対応

### モデル管理
- **HuggingFace**: モデルハブ統合
- **Ollama**: ローカルモデル管理 (port 11434)
- **vLLM**: 高速推論エンジン
- **Model Registry**: `outputs/`ディレクトリ

## ファインチューニング技術
- **LoRA**: Parameter-Efficient Fine-Tuning
- **DoRA**: Weight-Decomposed LoRA
- **QLoRA**: 量子化LoRA
- **EWC**: Elastic Weight Consolidation
- **PEFT**: HuggingFace PEFT統合
- **DeepSpeed**: 分散訓練対応
- **Accelerate**: マルチGPU最適化

## RAG技術スタック
### ベクトルデータベース
- **Qdrant**: v1.7+ - メインベクトルストア
- **UUID管理**: ポイントID管理

### 埋め込みモデル
- **multilingual-e5-large**: 多言語対応
- **sentence-transformers**: 文埋め込み
- **OpenAI Embeddings**: オプション対応

### 文書処理
- **PyPDF2**: PDF解析
- **Tesseract OCR**: 画像文字認識
- **Tabula**: 表抽出
- **LangChain**: テキスト分割

### 検索技術
- **ハイブリッド検索**: ベクトル(0.7) + BM25(0.3)
- **Re-ranking**: Cross-encoder再ランキング
- **Semantic Search**: 意味検索

## 推論最適化
### 量子化
- **AWQ**: 4ビット量子化 (75%メモリ削減)
- **GPTQ**: GPU最適化量子化
- **BitsAndBytes**: 8/4ビット量子化
- **Dynamic Quantization**: 動的量子化

### 高速化
- **vLLM**: PagedAttention
- **Flash Attention 2**: 効率的アテンション
- **KV Cache**: キャッシュ最適化
- **Batch Processing**: バッチ推論

## インフラ・デプロイメント
### コンテナ化
- **Docker**: v24+
- **Docker Compose**: マルチサービス管理
- **NVIDIA Container Toolkit**: GPU対応

### GPU/計算
- **CUDA**: 11.8/12.1対応
- **cuDNN**: 8.9+
- **Multi-GPU**: DataParallel/DistributedDataParallel
- **Mixed Precision**: FP16/BF16対応

## Web技術
### フロントエンド
- **Bootstrap**: v5.3 - UIフレームワーク
- **JavaScript**: Vanilla JS + jQuery
- **Chart.js**: データビジュアライゼーション
- **SSE**: Server-Sent Events (ストリーミング)

### バックエンド
- **Uvicorn**: ASGIサーバー
- **WebSockets**: リアルタイム通信
- **Background Tasks**: 非同期タスク処理
- **CORS**: クロスオリジン対応

## データ管理
### データベース
- **SQLite**: メタデータ管理
- **JSON**: 設定・履歴管理
- **Parquet**: 大規模データセット

### ストレージ
- **Local Storage**: `outputs/`, `data/`
- **HuggingFace Hub**: モデル共有
- **S3互換**: オプション対応

## モニタリング・ログ
- **Python Logging**: 標準ログ
- **Prometheus**: メトリクス収集 (オプション)
- **WandB**: 実験追跡 (オプション)
- **TensorBoard**: 訓練可視化

## 開発ツール
- **pytest**: テストフレームワーク
- **Black/Ruff**: コードフォーマッタ
- **mypy**: 型チェック
- **pre-commit**: Gitフック

## セキュリティ
- **JWT**: 認証トークン (予定)
- **Rate Limiting**: レート制限
- **Input Validation**: Pydantic検証
- **CORS Policy**: オリジン制限

## 依存関係管理
- **pip**: requirements.txt
- **Poetry**: pyproject.toml (オプション)
- **Conda**: 環境管理 (オプション)

## バージョン要件
- **Python**: 3.10+
- **Node.js**: 18+ (開発用)
- **CUDA**: 11.8+ (GPU使用時)
- **Docker**: 24+
- **Ubuntu**: 20.04/22.04 (推奨)