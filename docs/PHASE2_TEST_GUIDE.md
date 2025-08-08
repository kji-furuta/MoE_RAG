# Phase 2 テスト実行手順

## 修正内容

1. **ServiceScope → ServiceScopeEnum に名前変更**
   - クラス名の重複を解消
   - container.py と services.py で修正済み

2. **Docker環境でのテスト実行スクリプト作成**
   - `run_phase2_tests_docker.sh` を作成

## テスト実行方法

### 1. スクリプトに実行権限を付与

```bash
chmod +x scripts/check_docker_dependencies.sh
chmod +x scripts/run_phase2_tests_docker.sh
```

### 2. Docker環境の依存関係を再確認

```bash
bash scripts/check_docker_dependencies.sh
```

### 3. Phase 2テストをDocker環境で実行

```bash
bash scripts/run_phase2_tests_docker.sh
```

### 4. ホスト環境でテストする場合（必要な依存関係をインストール後）

```bash
# psutilをインストール
pip3 install psutil

# Phase 2統合テスト
python3 scripts/test_phase2_integration.py

# システム最適化テスト
python3 scripts/optimize_rag_system.py
```

## エラーが発生した場合

### "ModuleNotFoundError: No module named 'psutil'"
```bash
# Docker環境の場合
docker exec ai-ft-container pip install psutil

# ホスト環境の場合
pip3 install psutil
```

### その他の依存関係エラー
```bash
# Docker環境で必要なパッケージをインストール
docker exec ai-ft-container pip install loguru pydantic
```

## 期待される結果

### check_docker_dependencies.sh の実行結果
- ✅ Docker container is running
- ✅ System Can Run: True (Docker内)
- 📦 Installed Packages のリスト表示

### run_phase2_tests_docker.sh の実行結果
- ✅ DI Container テスト成功
- ✅ Health Check System テスト成功
- ✅ Metrics Collection テスト成功
- ✅ System Optimization 分析完了

## トラブルシューティング

1. **Dockerコンテナが起動していない場合**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **依存関係のインポートエラー**
   - Docker環境内で実行することを推奨
   - ホスト環境の場合は必要なパッケージをインストール

3. **ファイルが見つからないエラー**
   - プロジェクトルートから実行していることを確認
   ```bash
   cd ~/AI_FT/AI_FT_3
   ```
