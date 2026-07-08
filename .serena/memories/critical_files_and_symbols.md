# 重要ファイルとシンボルの参照

## コアシステムファイル

### メインアプリケーション
- `app/main_unified.py` - 統合FastAPIサーバー
  - クラス: `UnifiedServer`
  - 関数: `startup_event`, `shutdown_event`, `create_app`
  - 重要変数: `app`, `model_manager`, `rag_engine`

### モデル管理
- `app/memory_optimized_loader.py` - メモリ最適化モデルローダー
  - クラス: `MemoryOptimizedModelLoader`
  - メソッド: `load_model`, `quantize_model`, `offload_to_cpu`
  - 設定: `QUANTIZATION_CONFIG`, `OFFLOAD_CONFIG`

## ファインチューニングシステム

### LoRA訓練
- `src/training/lora_finetuning.py`
  - クラス: `LoRATrainer`
  - メソッド: `train`, `evaluate`, `save_adapter`
  - 設定: `LoraConfig`, `TrainingArguments`

### フル訓練
- `src/training/full_finetuning.py`
  - クラス: `FullTrainer`
  - メソッド: `train_model`, `validate`, `export_model`

### 継続学習
- `src/training/ewc_utils.py`
  - クラス: `EWCTrainer`
  - 関数: `compute_fisher_matrix`, `consolidate_weights`
  - 重要度: EWC継続学習の核心実装

### マルチGPU
- `src/training/multi_gpu_training.py`
  - 関数: `setup_distributed`, `train_distributed`
  - 設定: `WORLD_SIZE`, `RANK`, `LOCAL_RANK`

## RAGシステム

### コアエンジン
- `src/rag/core/query_engine.py`
  - クラス: `RAGQueryEngine`
  - メソッド: `query`, `retrieve`, `generate_response`
  - 設定: `RETRIEVAL_CONFIG`, `GENERATION_CONFIG`

### ベクトルストア
- `src/rag/indexing/vector_store.py`
  - クラス: `QdrantVectorStore`
  - メソッド: `add_documents`, `search`, `delete`
  - 接続: `QDRANT_HOST`, `QDRANT_PORT`

### ハイブリッド検索
- `src/rag/retrieval/hybrid_search.py`
  - クラス: `HybridSearcher`
  - メソッド: `vector_search`, `keyword_search`, `combine_results`
  - 重み: `VECTOR_WEIGHT=0.7`, `KEYWORD_WEIGHT=0.3`

### リランキング
- `src/rag/retrieval/reranker.py`
  - クラス: `MultiLayerReranker`
  - メソッド: `rerank`, `score_relevance`
  - モデル: `cross-encoder/ms-marco-MiniLM-L-12-v2`

## ドキュメント処理

### PDF処理
- `src/rag/document_processing/pdf_processor.py`
  - クラス: `PDFProcessor`
  - メソッド: `extract_text`, `extract_tables`, `extract_images`

### OCR処理
- `src/rag/document_processing/ocr_processor.py`
  - クラス: `OCRProcessor`
  - メソッド: `process_image`, `extract_text_from_scan`
  - エンジン: Tesseract OCR

### テーブル抽出
- `src/rag/document_processing/table_extractor.py`
  - クラス: `TableExtractor`
  - メソッド: `extract_tables`, `parse_table_structure`

## 特殊機能（土木設計）

### 設計基準チェック
- `src/rag/specialized/design_standard_checker.py`
  - クラス: `DesignStandardChecker`
  - メソッド: `check_compliance`, `validate_parameters`
  - 基準: 道路構造令、設計速度基準

### 数値処理
- `src/rag/specialized/numerical_processor.py`
  - クラス: `NumericalProcessor`
  - メソッド: `extract_numbers`, `validate_calculations`
  - 単位変換: `convert_units`, `normalize_values`

## 設定ファイル

### モデル設定
- `config/model_config.yaml` - モデル設定の定義
- `configs/training_config.yaml` - 訓練パラメータ
- `configs/rag_config.yaml` - RAGシステム設定

### 環境設定
- `.env` - 環境変数（HF_TOKEN, WANDB_API_KEY等）
- `docker/.env` - Docker環境変数
- `requirements.txt` - Pythonパッケージ依存関係

## テンプレートとUI

### HTMLテンプレート
- `templates/base.html` - ベーステンプレート
- `templates/index.html` - ダッシュボード
- `templates/finetune.html` - ファインチューニングUI
- `templates/rag.html` - RAG検索UI

### 静的ファイル
- `static/css/style.css` - カスタムスタイル
- `static/js/main.js` - メインJavaScript
- `static/js/training.js` - 訓練モニタリング
- `static/js/rag.js` - RAG機能のUI制御

## Docker関連

### Dockerファイル
- `docker/Dockerfile` - メインコンテナ定義
- `docker/docker-compose.yml` - サービス構成
- `docker/Dockerfile.rag` - RAG専用コンテナ

### 起動スクリプト
- `scripts/docker_build_rag.sh` - RAG環境構築
- `scripts/start_web_interface.sh` - Webインターフェース起動
- `scripts/start_unified_server.sh` - 統合サーバー起動

## テストファイル

### 統合テスト
- `scripts/test_integration.py` - システム統合テスト
- `scripts/test_docker_rag.py` - Docker環境テスト
- `scripts/test_config_resolution.py` - 設定解決テスト

### 機能テスト
- `scripts/simple_feature_test.py` - 基本機能テスト
- `scripts/test_specialized_features.py` - 特殊機能テスト
- `tests/test_rag_query.py` - RAGクエリテスト