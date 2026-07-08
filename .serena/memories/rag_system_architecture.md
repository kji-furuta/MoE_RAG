# RAGシステムアーキテクチャ詳細

## システム概要
MoE-RAG統合システムのRAG（Retrieval-Augmented Generation）コンポーネントは、土木工学・道路設計に特化した文書検索と知識拡張システムです。Qdrantベクトルデータベースとハイブリッド検索を組み合わせ、専門的な技術文書から高精度な情報検索を実現します。

## コアアーキテクチャ

### 1. ディレクトリ構造
```
src/rag/
├── core/                    # コア機能
│   ├── query_engine.py     # メインクエリエンジン
│   ├── citation_engine.py  # 引用管理
│   └── continual_model_manager.py  # 継続学習モデル管理
├── indexing/               # インデックス管理
│   ├── vector_store.py    # Qdrantベクトルストア
│   ├── embedding_model.py # 埋め込みモデル
│   └── metadata_manager.py # メタデータ管理
├── retrieval/              # 検索機能
│   ├── hybrid_search.py   # ハイブリッド検索
│   └── reranker.py       # リランキング
├── document_processing/    # 文書処理
│   ├── document_processor.py  # 文書処理メイン
│   ├── pdf_processor.py      # PDF処理
│   ├── ocr_processor.py      # OCR処理
│   ├── table_extractor.py    # テーブル抽出
│   └── chunking_strategy.py  # チャンキング戦略
├── specialized/           # 特殊機能
│   ├── numerical_processor.py    # 数値処理
│   ├── calculation_validator.py  # 計算検証
│   └── version_manager.py       # バージョン管理
├── config/               # 設定管理
│   ├── rag_config.yaml  # RAG設定
│   └── prompt_templates.yaml # プロンプトテンプレート
└── utils/               # ユーティリティ
```

### 2. 主要コンポーネント

#### RoadDesignQueryEngine (src/rag/core/query_engine.py)
- **役割**: RAGシステムの中核エンジン
- **主要メソッド**:
  - `initialize()`: システム初期化（3つのモード: lightweight, minimal, full）
  - `query()`: クエリ実行（MoE統合、Ollama連携、ハイブリッド検索）
  - `batch_query()`: バッチクエリ処理
- **統合機能**:
  - MoEシステムとの連携
  - Ollamaモデル（Llama 3.2 3B）統合
  - ハイブリッド検索（ベクトル+キーワード）

#### QdrantVectorStore (src/rag/indexing/vector_store.py)
- **役割**: ベクトルデータベース管理
- **特徴**:
  - UUID形式のpoint ID管理
  - multilingual-e5-large埋め込みモデル使用
  - メタデータベースのフィルタリング機能
  - doc_idフィールドによる文書特定

#### HybridSearchEngine (src/rag/retrieval/hybrid_search.py)
- **役割**: ハイブリッド検索実行
- **検索戦略**:
  - ベクトル類似度検索（重み: 0.7）
  - キーワードマッチング（重み: 0.3）
  - 技術用語抽出と専門用語対応
- **コンポーネント**:
  - TechnicalTermExtractor: 技術用語抽出
  - KeywordSearchEngine: BM25ベースのキーワード検索

#### RoadDesignDocumentProcessor (src/rag/document_processing/document_processor.py)
- **役割**: 文書処理とチャンキング
- **処理フロー**:
  1. PDF/テキスト文書の読み込み
  2. OCR処理（必要に応じて）
  3. テーブル/図表の抽出
  4. チャンキング（512トークン、128トークンオーバーラップ）
  5. メタデータ付与

### 3. MoE-RAG統合 (src/moe_rag_integration/unified_moe_rag_system.py)

#### UnifiedMoERAGSystem
- **役割**: MoEとRAGの統合管理
- **主要機能**:
  - エキスパート選択とRAGコンテキストの統合
  - エキスパート特化文書検索
  - 統合回答生成
- **処理フロー**:
  1. RAG検索実行 (`_execute_rag_search`)
  2. 文書からエキスパート選択 (`_select_experts_with_rag_context`)
  3. エキスパート別文書フィルタリング (`_filter_documents_by_experts`)
  4. 統合回答生成 (`_generate_unified_answer`)

### 4. API エンドポイント (app/main_unified.py)

#### 基本エンドポイント
- `GET /rag/health`: ヘルスチェック
- `GET /rag/system-info`: システム情報取得
- `POST /rag/update-settings`: 設定更新

#### クエリ関連
- `POST /rag/query`: 標準クエリ実行
- `POST /rag/batch-query`: バッチクエリ実行
- `POST /rag/stream-query`: ストリーミングクエリ
- `GET /rag/search`: 文書検索

#### 文書管理
- `POST /rag/upload-document`: 文書アップロード
- `GET /rag/documents`: 文書一覧取得
- `DELETE /rag/documents/{document_id}`: 文書削除
- `GET /rag/upload-status/{document_id}`: アップロード状態確認

#### 検索履歴管理
- `POST /rag/save-search`: 検索結果保存
- `GET /rag/search-history`: 履歴取得
- `DELETE /rag/search-history/{result_id}`: 履歴項目削除
- `DELETE /rag/search-history/clear`: 履歴クリア
- `GET /rag/export-searches`: 検索結果エクスポート

#### 統計・監視
- `GET /rag/statistics`: 統計情報取得

## データフロー

### 1. 文書インデックス作成フロー
```
文書アップロード
    ↓
DocumentProcessor（PDF/OCR処理）
    ↓
チャンキング（512トークン単位）
    ↓
EmbeddingModel（multilingual-e5-large）
    ↓
QdrantVectorStore（ベクトル保存）
    ↓
MetadataManager（メタデータ管理）
```

### 2. クエリ処理フロー
```
ユーザークエリ
    ↓
QueryEngine.query()
    ↓
[MoE統合モード]              [標準RAGモード]
    ↓                            ↓
UnifiedMoERAGSystem          HybridSearchEngine
    ↓                            ↓
エキスパート選択              ベクトル+キーワード検索
    ↓                            ↓
専門文書フィルタリング        Reranker（再ランキング）
    ↓                            ↓
統合回答生成                  LLMGenerator（回答生成）
    ↓                            ↓
CitationEngine（引用付与）
    ↓
最終回答
```

### 3. 文書フィルタリングメカニズム
- **問題**: 文書選択フィルタが機能しない問題を修正済み
- **解決策**: 
  - MetadataManagerのUUID（doc_id）をベクトルストアのメタデータに保存
  - 検索時にdoc_idフィールドでマッチング
  - 既存文書は再インデックスが必要

## 重要な設定パラメータ

### ベクトル検索設定
- **埋め込みモデル**: intfloat/multilingual-e5-large
- **ベクトル次元**: 1024
- **チャンクサイズ**: 512トークン
- **オーバーラップ**: 128トークン

### ハイブリッド検索重み
- **ベクトル類似度**: 0.7
- **キーワードマッチ**: 0.3

### MoE統合設定
- **エキスパート数**: 8（土木工学分野別）
- **選択閾値**: 設定可能
- **コンテキスト長**: 最大2048トークン

## パフォーマンス最適化

### メモリ最適化
- 動的量子化（モデルサイズに応じて4bit/8bit）
- CPU オフロード対応
- モデルキャッシング

### 検索最適化
- バッチ処理対応
- 非同期処理
- インデックスキャッシング

## トラブルシューティング

### よくある問題と対策
1. **文書フィルタリング不具合**
   - 原因: doc_idフィールドの不一致
   - 対策: 再インデックス実行

2. **メモリ不足**
   - 原因: 大規模モデル読み込み
   - 対策: 量子化有効化、CPU オフロード

3. **検索精度低下**
   - 原因: チャンクサイズ不適切
   - 対策: チャンクサイズ調整、オーバーラップ増加