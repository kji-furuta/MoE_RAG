# API エンドポイントリファレンス

## ファインチューニング API (ポート8050)

### 訓練関連
- `POST /api/train` - モデル訓練の開始
  - Parameters: model_name, training_type, dataset_path, epochs, batch_size, learning_rate
- `GET /api/training-status/{task_id}` - 訓練状態の確認
- `GET /api/training-logs/{task_id}` - 訓練ログの取得
- `DELETE /api/cancel-training/{task_id}` - 訓練のキャンセル

### 推論関連
- `POST /api/generate` - テキスト生成
  - Parameters: model_name, prompt, max_tokens, temperature, top_p
- `POST /api/chat` - チャット形式の対話
- `GET /api/models` - 利用可能なモデルリスト

### モデル管理
- `POST /api/models/load` - モデルのロード
- `POST /api/models/unload` - モデルのアンロード
- `GET /api/models/info/{model_name}` - モデル情報の取得
- `POST /api/models/quantize` - モデルの量子化

## RAG API (ポート8050)

### ヘルスチェック
- `GET /rag/health` - システムヘルスチェック
- `GET /rag/system-info` - システム情報

### ドキュメント管理
- `POST /rag/upload-document` - ドキュメントアップロード
  - Form data: file (multipart)
- `GET /rag/documents` - ドキュメント一覧
- `DELETE /rag/documents/{document_id}` - ドキュメント削除
- `POST /rag/index-documents` - ドキュメントのインデックス化

### 検索・クエリ
- `POST /rag/query` - 高度なドキュメント検索
  - Parameters: query, top_k, rerank, filter_criteria
- `POST /rag/stream-query` - ストリーミング検索
- `POST /rag/hybrid-search` - ハイブリッド検索（ベクトル＋キーワード）

### 特殊機能（土木設計）
- `POST /rag/design-check` - 設計基準チェック
  - Parameters: design_specs, standard_type
- `POST /rag/numerical-validation` - 数値検証
  - Parameters: values, validation_rules
- `POST /rag/extract-tables` - 表データ抽出
  - Parameters: document_id, page_number

## Web UI ルート (ポート8050)

### メインページ
- `/` - ダッシュボード
- `/finetune` - ファインチューニングインターフェース
- `/rag` - RAGインターフェース
- `/models` - モデル管理画面

### ドキュメント
- `/manual` - ユーザーマニュアル
- `/system-overview` - システムドキュメント
- `/api-docs` - Swagger UI (FastAPI自動生成)

### 静的ファイル
- `/static/` - CSS, JavaScript, 画像ファイル
- `/templates/` - HTMLテンプレート

## WebSocket エンドポイント

### リアルタイム通信
- `WS /ws/training/{task_id}` - 訓練進捗のリアルタイム更新
- `WS /ws/logs/{task_id}` - ログのストリーミング

## 認証ヘッダー

### APIキー認証
```
Authorization: Bearer {HF_TOKEN}
X-API-Key: {WANDB_API_KEY}
```

### CORS設定
- 許可オリジン: http://localhost:8050, http://127.0.0.1:8050
- 許可メソッド: GET, POST, PUT, DELETE, OPTIONS
- 許可ヘッダー: Content-Type, Authorization, X-API-Key