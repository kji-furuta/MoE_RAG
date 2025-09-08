# 全関連スクリプト復元完了 - 2025-09-08

## 復元されたスクリプト一覧

### 初期化・起動系
1. **start_web_interface.sh** - メイン起動スクリプト（オリジナル版）
2. **start_web_interface_production.sh** - 本番環境用起動スクリプト
3. **setup_permissions.sh** - 権限管理スクリプト
4. **setup_environment.sh** - 環境セットアップスクリプト

### Docker関連
5. **docker_build_rag.sh** - Docker環境構築スクリプト
6. **manage_services.sh** - サービス管理ツール

### Ollama/モデル管理
7. **init_ollama_models.sh** - Ollamaモデル初期化
8. **ensure_ollama_models.sh** - Ollamaモデル確認・ダウンロード
9. **init_deepseek_model.sh** - DeepSeekモデル初期化

### 変換・処理系
10. **apply_lora_to_gguf_improved.py** - LoRA→GGUF変換スクリプト
11. **setup_llama_cpp_standalone.sh** - llama.cpp独立セットアップ
12. **check_quantization_status.sh** - 量子化状態チェック

### ユーティリティ
13. **clean_logs.sh** - ログクリーンアップ

## 実行権限
すべての.shスクリプトに実行権限を設定済み

## 主要スクリプトの用途

### システム起動
```bash
# 開発環境（自動リロード有効）
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# 本番環境（自動リロード無効）
docker exec ai-ft-container bash /workspace/scripts/start_web_interface_production.sh
```

### Docker環境構築
```bash
./scripts/docker_build_rag.sh --no-cache
```

### サービス管理
```bash
./scripts/manage_services.sh status     # ステータス確認
./scripts/manage_services.sh start-all  # 全サービス起動
./scripts/manage_services.sh stop-all   # 全サービス停止
```

### モデル管理
```bash
# Ollamaモデル初期化
docker exec ai-ft-container bash /workspace/scripts/init_ollama_models.sh

# DeepSeekモデル初期化
docker exec ai-ft-container bash /workspace/scripts/init_deepseek_model.sh
```

## 復元状態
- すべての主要スクリプトが復元済み
- 実行権限設定済み
- システムは正常稼働中