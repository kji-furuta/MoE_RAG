# タスク完了時チェックリスト

## コード変更後の確認事項

### 1. コード品質チェック
```bash
# フォーマット適用
black src/ --line-length 88
black app/ --line-length 88

# インポート整理
isort src/ --profile black
isort app/ --profile black

# リントチェック
flake8 src/ app/
```

### 2. テスト実行
```bash
# ユニットテスト
pytest tests/ -v

# 統合テスト
python scripts/test/test_integration.py

# 特定機能のテスト
python scripts/test/test_docker_rag.py  # RAG変更時
python scripts/test/test_memory_optimization.py  # メモリ最適化変更時
```

### 3. Docker環境での動作確認
```bash
# コンテナ内でのモジュール確認
docker exec ai-ft-container python -c "import sys; print(sys.path)"

# Webインターフェース起動確認
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# APIエンドポイント確認
curl http://localhost:8050/health
curl http://localhost:8050/rag/health
```

### 4. GPU/メモリ状況確認
```bash
# GPU使用状況
nvidia-smi

# メモリ使用状況
free -h

# Dockerリソース確認
docker stats ai-ft-container
```

### 5. ログ確認
```bash
# アプリケーションログ
docker logs ai-ft-container --tail 100

# エラーログのみ
docker logs ai-ft-container 2>&1 | grep -i error
```

### 6. ドキュメント更新（必要時）
- README.md の更新（新機能追加時）
- CLAUDE.md の更新（開発ワークフロー変更時）
- API仕様の更新（エンドポイント変更時）

### 7. Git操作（ユーザー要求時のみ）
```bash
# 変更確認
git status
git diff

# コミット（ユーザー明示的要求時のみ）
git add .
git commit -m "feat: 機能説明"
```

## 注意事項
- **GPU大規模モデル訓練前**: nvidia-smiでVRAM確認必須
- **RAG変更時**: Qdrant接続確認必須
- **依存関係追加時**: requirements.txt更新必須
- **エラー発生時**: ログ確認→モジュールパス確認→Docker再起動