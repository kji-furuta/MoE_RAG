# 開発環境セットアップ完了

## セットアップ内容

### 1. リポジトリ状態確認
- ✅ Git リポジトリ構造の確認完了
- ✅ 不要なZone.Identifierファイルを削除

### 2. Docker環境
- ✅ docker-compose.yml の構成確認
  - ai-ftコンテナ（メイン）
  - Qdrant（ベクトルDB）
  - TensorBoard（オプション）
  - Jupyter Lab（オプション）

### 3. 依存関係
- ✅ requirements.txt 確認（Fine-tuning用）
- ✅ requirements_rag.txt 確認（RAG用）

### 4. 設定ファイル
- ✅ .env ファイル確認（APIキー設定済み）
- ✅ docker/.env ファイル確認

### 5. スクリプト権限設定
- ✅ 全てのシェルスクリプトに実行権限付与
- ✅ scriptsディレクトリ内のPythonスクリプトに実行権限付与

### 6. 開発ツール作成
- ✅ `start_dev_env.sh` - 開発環境起動スクリプト
- ✅ `stop_dev_env.sh` - 開発環境停止スクリプト

## 使用方法

### 開発環境の起動
```bash
./start_dev_env.sh
```

### 開発環境の停止
```bash
./stop_dev_env.sh
```

### Docker環境の手動操作
```bash
# コンテナ起動
cd docker
docker-compose up -d

# Webインターフェース起動
docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload

# コンテナ停止
docker-compose down
```

## アクセスポイント

- **統合Webインターフェース**: http://localhost:8050
- **RAG API**: http://localhost:8050/rag
- **Fine-tuning API**: http://localhost:8050/api
- **Qdrant UI**: http://localhost:6333/dashboard
- **Jupyter Lab**: http://localhost:8888
- **TensorBoard**: http://localhost:6006

## テストコマンド

### RAGシステムテスト
```bash
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度80km/hの道路の最小曲線半径は？", "top_k": 5}'
```

### ヘルスチェック
```bash
curl "http://localhost:8050/rag/health"
```

## トラブルシューティング

### ポート競合の場合
```bash
# 使用中のポート確認
netstat -tlnp | grep 8050
```

### ログ確認
```bash
# コンテナログ
docker logs ai-ft-container --tail 50

# リアルタイムログ
docker logs -f ai-ft-container
```

### コンテナ内部へのアクセス
```bash
docker exec -it ai-ft-container bash
```

## 次のステップ

1. `./start_dev_env.sh` を実行して開発環境を起動
2. http://localhost:8050 にアクセスして動作確認
3. 必要に応じて追加の設定やカスタマイズを実施

## 注意事項

- WSL2環境での動作を前提としています
- GPU利用にはnvidia-docker2が必要です
- 初回起動時はDockerイメージのビルドに時間がかかります