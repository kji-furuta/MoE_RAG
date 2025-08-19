# タスク完了時のチェックリスト

## コード変更後の確認事項

### 1. コードフォーマット
```bash
# Blackでフォーマット
black src/ app/ --line-length 88

# isortでインポート整理
isort src/ app/ --profile black
```

### 2. リント実行
```bash
# flake8でコード品質チェック
flake8 src/ app/ --max-line-length 88 --ignore E203,W503
```

### 3. テスト実行
```bash
# 変更に関連するテストを実行
pytest tests/ -v

# 統合テストの実行（重要な変更の場合）
python scripts/test_integration.py
```

### 4. 型チェック（オプション）
```bash
# mypyが設定されている場合
mypy src/ --ignore-missing-imports
```

## モデル訓練完了後

### 1. モデル検証
- 出力ディレクトリの確認: `ls -la outputs/`
- モデルファイルの整合性確認
- training_info.jsonの確認

### 2. 推論テスト
```bash
# 簡単な推論テスト実行
python scripts/test_model_inference.py --model_path outputs/latest_model
```

### 3. メトリクス確認
- 学習ログの確認
- Tensorboardでの可視化（利用可能な場合）
- 損失値の収束確認

## RAG更新後

### 1. インデックス確認
```bash
# RAGヘルスチェック
curl "http://localhost:8050/rag/health"
```

### 2. 検索テスト
```bash
# サンプルクエリでテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "テストクエリ", "top_k": 3}'
```

## API変更後

### 1. エンドポイントテスト
- 新規/変更されたエンドポイントの動作確認
- エラーハンドリングの確認
- レスポンス形式の検証

### 2. ドキュメント更新
- APIドキュメントの更新（必要に応じて）
- README.mdの更新（重要な変更の場合）

## Docker環境変更後

### 1. ビルド確認
```bash
# イメージの再ビルド
docker-compose build --no-cache

# コンテナの起動確認
docker-compose up -d
docker logs ai-ft-container --tail 50
```

### 2. 依存関係確認
```bash
# requirements.txtの更新確認
docker exec ai-ft-container pip list
```

## デプロイ前の最終確認

### 1. セキュリティチェック
- 環境変数にシークレットが含まれていないか
- デバッグモードが無効になっているか
- 不要なログ出力が削除されているか

### 2. パフォーマンス確認
- メモリ使用量の確認
- GPU使用率の確認
- レスポンス時間の測定

### 3. ドキュメント
- CLAUDE.mdの更新（必要に応じて）
- 変更履歴の記録

## トラブル時の対処

### エラー発生時
1. エラーログの確認
2. 関連するテストの再実行
3. 前のコミットとの差分確認

### パフォーマンス問題
1. プロファイリングツールの使用
2. メモリリークの確認
3. バッチサイズやパラメータの調整

## 重要な注意事項

⚠️ **本番環境へのデプロイ前に必ず実施**:
- 全テストスイートの実行
- コードレビュー（可能な場合）
- バックアップの作成
- ロールバック手順の確認