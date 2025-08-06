# タスク完了チェックリスト（詳細版）

## 🔍 コード変更前の確認

### 1. 影響範囲の分析
```bash
# 変更対象の依存関係を確認
serena:find_referencing_symbols "変更する関数/クラス名" "."

# 関連テストの特定
grep -r "変更する関数名" tests/
```

### 2. 現状の記録
```bash
# 変更前の動作確認
python scripts/test/test_integration.py
curl http://localhost:8050/health

# メトリクスの記録
nvidia-smi > before_change_gpu.log
docker stats --no-stream > before_change_docker.log
```

## ✅ コード実装チェック

### 1. コード品質
- [ ] 型ヒントを追加したか
- [ ] 適切なエラーハンドリングを実装したか
- [ ] ログ出力を適切なレベルで追加したか
- [ ] docstringを記載したか
- [ ] 定数を設定ファイルに外出ししたか

### 2. パフォーマンス
- [ ] 不要なループを避けているか
- [ ] メモリリークの可能性はないか
- [ ] GPU/CPUリソースを効率的に使用しているか
- [ ] キャッシュを適切に活用しているか

## 🧹 コード整形

### 1. 自動フォーマット
```bash
# Python コードのフォーマット
black src/ app/ scripts/ --line-length 88

# インポートの整理
isort src/ app/ scripts/ --profile black

# 未使用インポートの削除
autoflake --in-place --remove-all-unused-imports -r src/ app/
```

### 2. リンティング
```bash
# flake8 チェック
flake8 src/ app/ --max-line-length 88 --ignore E203,W503

# 複雑度チェック
flake8 src/ app/ --max-complexity 10

# 型チェック（オプション）
mypy src/ --ignore-missing-imports
```

## 🧪 テスト実行

### 1. ユニットテスト
```bash
# 変更に関連するテストを実行
pytest tests/test_specific_module.py -v

# カバレッジ確認
pytest tests/ --cov=src --cov-report=html
```

### 2. 統合テスト
```bash
# Docker環境での統合テスト
docker exec ai-ft-container pytest tests/ -v

# APIエンドポイントテスト
python scripts/test/test_integration.py

# RAG機能テスト（RAG変更時）
python scripts/test/test_docker_rag.py
```

### 3. 負荷テスト（必要時）
```bash
# 同時リクエストテスト
ab -n 100 -c 10 http://localhost:8050/health

# メモリ使用量の監視
python scripts/test/test_memory_optimization.py
```

## 🐳 Docker環境確認

### 1. コンテナ状態
```bash
# コンテナの再ビルド（大きな変更時）
cd docker && docker-compose up -d --build

# ログ確認
docker logs ai-ft-container --tail 100 | grep -i error

# コンテナ内での動作確認
docker exec ai-ft-container python -c "import app.main_unified; print('OK')"
```

### 2. リソース確認
```bash
# Docker統計
docker stats --no-stream

# ディスク使用量
docker system df

# 不要なリソースのクリーンアップ
docker system prune -f
```

## 🔧 環境別確認

### 1. 開発環境
```bash
# ローカルサーバー起動確認
python -m uvicorn app.main_unified:app --reload

# ホットリロード動作確認
# コード変更 → 自動リロード確認
```

### 2. GPU環境（該当時）
```bash
# GPU使用状況
nvidia-smi

# CUDA availability確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# メモリクリア（必要時）
python -c "import torch; torch.cuda.empty_cache()"
```

## 📚 ドキュメント更新

### 1. コードドキュメント
- [ ] 新規関数/クラスのdocstring記載
- [ ] 変更した関数の引数説明更新
- [ ] 使用例の追加/更新

### 2. プロジェクトドキュメント
- [ ] README.md（新機能追加時）
- [ ] CHANGELOG.md（重要な変更時）
- [ ] API仕様書（エンドポイント変更時）

### 3. 設定ファイル
- [ ] requirements.txt（新規ライブラリ追加時）
- [ ] docker-compose.yml（サービス変更時）
- [ ] 環境変数の文書化（.env.example）

## 🚀 デプロイ前確認

### 1. 最終動作確認
```bash
# 全体テストスイート実行
pytest tests/ -v

# エンドツーエンドテスト
curl -X POST http://localhost:8050/api/train -H "Content-Type: application/json" -d '{...}'

# ヘルスチェック
curl http://localhost:8050/health
curl http://localhost:8050/rag/health
```

### 2. パフォーマンス確認
```bash
# レスポンスタイム測定
time curl http://localhost:8050/health

# メモリ使用量確認
free -h
docker stats --no-stream
```

## 📝 最終チェックリスト

- [ ] コードレビューを受けたか
- [ ] テストが全て成功したか
- [ ] ログにエラーがないか
- [ ] メモリリークがないか
- [ ] ドキュメントを更新したか
- [ ] 環境変数の変更を文書化したか
- [ ] ロールバック手順を確認したか

## 🔄 問題発生時の対処

### 1. ロールバック準備
```bash
# 現在の状態を記録
git stash
docker commit ai-ft-container ai-ft-container:backup

# 問題発生時
git stash pop
docker run -d --name ai-ft-container-rollback ai-ft-container:backup
```

### 2. デバッグ情報収集
```bash
# 詳細ログ取得
docker logs ai-ft-container > debug.log 2>&1

# システム情報
uname -a > system_info.log
pip freeze > pip_packages.log
nvidia-smi -q > gpu_info.log
```