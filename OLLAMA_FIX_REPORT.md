# Ollama起動設定修正レポート

## 問題の概要
システム起動時にOllamaサービスが稼働するように設定されていたが、ホストからアクセスできない状態だった。

## 原因
Ollamaがデフォルトでlocalhost（127.0.0.1）のみでリッスンしており、Dockerコンテナ外（ホスト）からアクセスできなかった。

## 修正内容

### 1. **Dockerfile** (`/home/kjifu/MoE_RAG/docker/Dockerfile`)
```dockerfile
# Ollamaをホストからアクセス可能にする設定
ENV OLLAMA_HOST=0.0.0.0
```

### 2. **docker-compose.yml** (`/home/kjifu/MoE_RAG/docker/docker-compose.yml`)
```yaml
environment:
  - OLLAMA_HOST=0.0.0.0  # 追加
```

### 3. **entrypoint.sh** (`/home/kjifu/MoE_RAG/docker/entrypoint.sh`)
```bash
# ホストからアクセス可能にするため、0.0.0.0でリッスン
export OLLAMA_HOST=0.0.0.0
nohup ollama serve > /var/log/ollama.log 2>&1 &
```

### 4. **start_web_interface.sh** (`/home/kjifu/MoE_RAG/scripts/start_web_interface.sh`)
```bash
# ホストからアクセス可能にするため、0.0.0.0でリッスン
export OLLAMA_HOST=0.0.0.0
nohup ollama serve > /dev/null 2>&1 &
```

## 検証結果

### システム再起動後の状態
```bash
# 環境変数確認
$ docker exec ai-ft-container env | grep OLLAMA
OLLAMA_HOST=0.0.0.0  ✅

# プロセス確認
$ docker exec ai-ft-container ps aux | grep ollama
ollama serve  ✅ (稼働中)

# API確認（ホストから）
$ curl http://localhost:11434/api/tags
{"models":[...]}  ✅ (3モデル検出)
```

### メトリクス測定結果
```json
{
  "ollama": {
    "status": "online",
    "models_count": 3
  }
}
```

### 利用可能なモデル
1. **deepseek-32b-japanese:latest** (18 GB)
2. **llama3.2:1b** (1.3 GB)
3. **llama3.2:3b** (2.0 GB)

## 結論
修正により、Ollamaサービスが起動時に正しく設定され、ホストからアクセス可能になった。

### 修正前
- Ollamaはlocalhost（127.0.0.1）のみでリッスン
- コンテナ内からはアクセス可能
- ホストからはアクセス不可 ❌

### 修正後
- Ollamaは0.0.0.0でリッスン
- コンテナ内からアクセス可能 ✅
- ホストからもアクセス可能 ✅
- メトリクス測定で正常検出 ✅

## 今後の推奨事項
1. **自動起動の確実性**: systemdやsupervisorを使用した、より堅牢なプロセス管理
2. **ヘルスチェック**: docker-compose.ymlにヘルスチェック定義を追加
3. **ログ管理**: Ollamaログのローテーション設定

---
*修正完了日時: 2025-09-08 20:52*
*検証者: Claude Code*