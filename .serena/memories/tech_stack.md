# 技術スタック

## コア言語・フレームワーク
- **Python**: 3.8以上
- **FastAPI**: 0.109.0+ (Webフレームワーク)
- **Uvicorn**: ASGIサーバー

## 機械学習・AI
- **PyTorch**: 2.0.0+ (深層学習フレームワーク)
- **Transformers**: 4.30.0+ (Hugging Face)
- **PEFT**: 0.8.0+ (Parameter-Efficient Fine-Tuning)
- **Accelerate**: 0.20.0+ (分散学習)
- **DeepSpeed**: 0.12.0+ (大規模モデル訓練)
- **BitsAndBytes**: 0.41.0+ (量子化)

## RAG・ベクトルDB
- **Qdrant**: ベクトル検索エンジン
- **Sentence-Transformers**: 埋め込みモデル
- **LangChain**: RAGパイプライン構築

## 推論最適化
- **vLLM**: PagedAttention高速推論
- **AWQ**: 4ビット量子化
- **Ollama**: ローカルLLM実行環境
- **TensorRT**: NVIDIA GPU最適化

## データ処理
- **Pandas**: 1.3.0+ (データ分析)
- **NumPy**: 1.21.0+ (数値計算)
- **Datasets**: 2.12.0+ (データセット管理)
- **PyPDF2**: PDF処理
- **Tesseract OCR**: 光学文字認識

## 開発ツール
- **Black**: コードフォーマッター (line-length=88)
- **isort**: インポート整理
- **pytest**: テストフレームワーク
- **flake8**: リンター

## インフラ・運用
- **Docker**: コンテナ化
- **Docker Compose**: マルチコンテナ管理
- **NVIDIA Container Toolkit**: GPU対応
- **Redis**: キャッシュ (オプション)
- **PostgreSQL**: メタデータストレージ (オプション)

## 監視・ロギング
- **Tensorboard**: 訓練可視化
- **Weights & Biases**: 実験管理
- **Prometheus**: メトリクス収集 (オプション)
- **Grafana**: ダッシュボード (オプション)

## その他のライブラリ
- **PyYAML**: 設定ファイル管理
- **python-dotenv**: 環境変数管理
- **aiofiles**: 非同期ファイル操作
- **websockets**: WebSocket通信
- **psutil**: システムリソース監視
- **GPUtil**: GPU使用状況監視