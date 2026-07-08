# MoE Dataset Update Feature - 実装記録

## 実装日時
2024-08-22

## 実装内容

### 1. フロントエンド変更
**ファイル**: `app/static/moe_training.html`

#### 追加機能：
- データセット更新セクション（初期状態では非表示）
- JSONLファイルアップロード機能
- 現在のデータセットダウンロード機能
- データセット統計情報表示
  - サンプル数
  - エキスパート分布
  - 最終更新日時
  - ファイルサイズ

#### JavaScript関数：
- `handleDatasetChange()` - データセット選択時の表示制御
- `loadDatasetStats()` - 統計情報の取得
- `updateDataset()` - データセット更新処理
- `downloadCurrentDataset()` - データセットダウンロード
- `formatFileSize()` - ファイルサイズフォーマット

### 2. バックエンドAPI追加
**ファイル**: `app/main_unified.py`

#### 新規エンドポイント：
```python
# データセット管理
GET  /api/moe/dataset/stats/{dataset_name}    # 統計情報取得
POST /api/moe/dataset/update                  # データセット更新
GET  /api/moe/dataset/download/{dataset_name} # ダウンロード

# MoEトレーニング
POST /api/moe/training/start                  # トレーニング開始
GET  /api/moe/training/status/{task_id}       # ステータス確認
POST /api/moe/training/stop/{task_id}         # トレーニング停止
GET  /api/moe/training/logs/{task_id}         # ログ取得
GET  /api/moe/training/gpu-status             # GPU状態
GET  /api/moe/training/history                # 履歴取得
POST /api/moe/training/deploy/{task_id}       # モデルデプロイ
```

### 3. UI アクセス改善
**ファイル**: `templates/index.html`

#### MoE-RAGセクションのボタン追加：
```html
<button onclick="window.open('/static/moe_training.html', '_blank')" 
        class="btn btn-primary" 
        style="background: #27ae60; border-color: #229954;">
    <i class="fas fa-external-link-alt"></i> 
    MoEトレーニング管理画面を新規タブで開く
</button>
```

## 使用方法

### データセット更新手順：
1. http://localhost:8050/ にアクセス
2. 「MoE-RAG」タブをクリック
3. 「MoEトレーニング管理画面を新規タブで開く」ボタンをクリック
4. データセットセクションで以下のいずれかを選択：
   - 土木工学データセット（civil_engineering）
   - 道路設計データセット（road_design）
5. 更新セクションが表示される
6. JSONLファイルを選択して「データセットを更新」をクリック

### データ形式：
```jsonl
{"expert_domain": "road_design", "question": "質問", "answer": "回答", "keywords": ["キーワード"]}
```

## 機能詳細

### 自動バックアップ
- 更新前に `data/backups/` にタイムスタンプ付きでバックアップ
- 形式: `{dataset_name}_{YYYYmmdd_HHMMSS}.jsonl`

### データ検証
- 必須フィールド: question, answer
- 無効な行は自動スキップ
- 検証結果をレスポンスで返却

### エラーハンドリング
- JSONデコードエラーの検出
- 必須フィールド欠落の検出
- 詳細なエラーメッセージの提供

## 依存関係
- python-multipart（Formデータ処理用）
- インストール済み: `pip install --break-system-packages python-multipart`

## テストスクリプト
`scripts/test_moe_dataset_update.py` - 全機能の動作テスト

## 対応データセット
- `data/moe_training_corpus.jsonl` - 土木工学データセット
- `data/moe_training_sample.jsonl` - 道路設計データセット

## 注意事項
- データセット更新機能は初期状態では非表示
- civil_engineeringまたはroad_designを選択時のみ表示
- サーバー起動コマンド: `docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`