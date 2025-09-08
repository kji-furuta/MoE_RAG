# スクリプト完全復元 - 2025-09-08

## 復元された重要スクリプト一覧

### 1. start_web_interface.sh (オリジナル版)
- 整理前の完全版に復元済み
- すべての初期設定機能を含む
- setup_permissions.shを参照

### 2. setup_permissions.sh (完全復元)
- 全ディレクトリの権限管理
- チェックモード対応 (--check)
- カラー出力でステータス表示
- ai-userへの所有者変更機能

### 3. init_ollama_models.sh (新規作成)
- Ollamaサービス管理
- モデル自動ダウンロード
- カスタムModelfile登録

### 4. manage_services.sh (復元)
- Docker-composeサービス管理
- アプリケーション・監視サービスの独立管理
- ステータス確認機能
- ログ表示機能

## Dockerfile修正内容
```dockerfile
# 変更前（削除されたスクリプトを参照）
COPY scripts/setup/fix_permissions.sh /workspace/scripts/setup/

# 変更後（復元したスクリプトを参照）
COPY scripts/setup_permissions.sh /workspace/scripts/
RUN chmod +x /workspace/scripts/setup_permissions.sh && \
    /workspace/scripts/setup_permissions.sh && \
```

## 使用方法

### Webインターフェース起動（全機能）
```bash
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### サービス管理
```bash
./scripts/manage_services.sh status      # ステータス確認
./scripts/manage_services.sh start-all   # 全サービス起動
./scripts/manage_services.sh stop-all    # 全サービス停止
./scripts/manage_services.sh logs-app    # アプリログ表示
```

### 権限管理
```bash
docker exec ai-ft-container bash /workspace/scripts/setup_permissions.sh --check  # チェックのみ
docker exec ai-ft-container bash /workspace/scripts/setup_permissions.sh          # 権限修正
```

### Ollamaモデル管理
```bash
docker exec ai-ft-container bash /workspace/scripts/init_ollama_models.sh
```

## 復元されたスクリプトの機能

### start_web_interface.sh
- Ollamaサービスの自動起動
- llama3.2:3bモデルの自動ダウンロード
- 権限チェックと自動修正
- 継続学習設定の初期化
- Webサーバー起動（ポート8050）

### manage_services.sh
- Docker-composeによるサービス管理
- アプリケーション（ポート8050）の管理
- Grafana/Prometheus監視の管理
- Redis、Qdrant、Ollamaのステータス確認

これですべての主要な連動スクリプトが復元され、start_web_interface.shの全機能が利用可能になりました。