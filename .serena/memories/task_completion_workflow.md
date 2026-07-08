# タスク完了時のワークフロー

## 必須チェック項目

### 1. コード品質チェック

```bash
# Blackでフォーマット
black src/ app/ scripts/

# isortでインポート整理  
isort src/ app/ scripts/

# flake8でリント
flake8 src/ app/ scripts/ --max-line-length=88
```

### 2. テスト実行

```bash
# ユニットテスト
pytest tests/

# 統合テスト（Docker環境で）
docker exec ai-ft-container python scripts/test_integration.py

# RAGシステムテスト
docker exec ai-ft-container python scripts/test_docker_rag.py
```

### 3. 型チェック

```bash
# mypyで型チェック（オプショナル）
mypy src/ --ignore-missing-imports
```

### 4. ドキュメント更新

- [ ] docstringの更新確認
- [ ] README.mdの更新（必要な場合）
- [ ] CLAUDE.mdの更新（大きな変更の場合）
- [ ] API仕様の更新（エンドポイント追加時）

### 5. 動作確認

```bash
# Webインターフェース確認
curl http://localhost:8050/rag/health

# RAGクエリ動作確認
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "テストクエリ", "top_k": 3}'

# ログ確認
docker logs ai-ft-container --tail 20
```

### 6. Git操作

```bash
# 変更内容確認
git status
git diff

# ステージング
git add -A

# コミット（セマンティックコミット）
git commit -m "feat: 新機能追加" # 新機能
git commit -m "fix: バグ修正"     # バグ修正
git commit -m "docs: ドキュメント更新" # ドキュメント
git commit -m "refactor: リファクタリング" # リファクタリング
git commit -m "test: テスト追加/修正" # テスト
git commit -m "chore: その他の変更" # その他
```

### 7. 環境クリーンアップ

```bash
# 不要なログファイル削除
docker exec ai-ft-container rm -rf /workspace/logs/*.log

# キャッシュクリア（必要時）
docker exec ai-ft-container python -c "import torch; torch.cuda.empty_cache()"

# 一時ファイル削除
docker exec ai-ft-container rm -rf /workspace/temp_uploads/*
```

## 重要なチェックポイント

### メモリ/パフォーマンス
- [ ] GPU メモリリークがないか確認 (`nvidia-smi`)
- [ ] CPU/メモリ使用率が正常範囲内か (`htop`)
- [ ] レスポンスタイムが許容範囲内か

### セキュリティ
- [ ] APIキーがコードに直書きされていないか
- [ ] SQLインジェクション対策されているか
- [ ] ファイルアップロードの検証があるか
- [ ] エラーメッセージに機密情報が含まれていないか

### Docker環境
- [ ] Dockerfileの変更が必要か
- [ ] docker-compose.ymlの更新が必要か
- [ ] 新しい環境変数の追加が必要か

## デプロイ前の最終確認

```bash
# 全テストスイート実行
docker exec ai-ft-container pytest tests/ -v

# ビルド確認
cd docker && docker-compose build

# サービス再起動
docker-compose down && docker-compose up -d

# ヘルスチェック
sleep 10
curl http://localhost:8050/rag/health
```

## トラブルシューティング時の確認

1. **ログ確認**
   ```bash
   docker logs ai-ft-container --tail 100
   ```

2. **権限確認**
   ```bash
   docker exec ai-ft-container ls -la /workspace/
   ```

3. **依存関係確認**
   ```bash
   docker exec ai-ft-container pip list | grep -E "torch|transformers|qdrant"
   ```

4. **ポート確認**
   ```bash
   netstat -tlnp | grep -E "8050|6333|11434"
   ```