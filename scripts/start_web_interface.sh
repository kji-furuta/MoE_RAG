#!/bin/bash

# AI_FT_3 統合Webインターフェース起動スクリプト
# 継続学習管理システムを含む全機能を起動

echo "🚀 AI_FT_3 統合Webインターフェースを起動中..."

# Ollamaサービスを起動（存在する場合）
if command -v ollama &> /dev/null; then
    echo "🤖 Ollamaサービスを起動中..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollamaサービスを起動しました (port 11434)"
    
    # Ollamaモデルの確認と自動ダウンロード
    echo "📦 Ollamaモデルを確認中..."
    if ! ollama list | grep -q "llama3.2:3b"; then
        echo "📥 llama3.2:3bモデルをダウンロード中..."
        ollama pull llama3.2:3b
        echo "✅ モデルのダウンロードが完了しました"
    else
        echo "✅ llama3.2:3bモデルが利用可能です"
    fi
fi

# 作業ディレクトリを設定
cd /workspace

# パーミッションをチェック・修正
echo "🔐 パーミッションをチェック中..."
if [ -f /workspace/scripts/setup_permissions.sh ]; then
    /workspace/scripts/setup_permissions.sh --check
    if [ $? -ne 0 ]; then
        echo "🔧 パーミッションを修正中..."
        if [ "$EUID" -eq 0 ]; then
            /workspace/scripts/setup_permissions.sh
        else
            sudo /workspace/scripts/setup_permissions.sh 2>/dev/null || /workspace/scripts/setup_permissions.sh
        fi
    fi
else
    echo "⚠️ パーミッション設定スクリプトが見つかりません"
    # フォールバック: 基本的なディレクトリ作成
    mkdir -p /workspace/outputs
    mkdir -p /workspace/data/continual_learning
    mkdir -p /workspace/logs
    mkdir -p /workspace/temp_uploads
fi

# 継続学習管理システムの初期化
echo "📋 継続学習管理システムを初期化中..."

# 継続学習用の設定ファイルを作成
cat > /workspace/config/continual_learning_config.yaml << EOF
# 継続学習管理システム設定
continual_learning:
  enabled: true
  base_models:
    - cyberagent/calm3-22b-chat
    - cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
    - Qwen/Qwen2.5-14B-Instruct
    - Qwen/Qwen2.5-32B-Instruct
  
  ewc_settings:
    default_lambda: 5000
    fisher_computation_batches: 100
    use_efficient_storage: true
  
  training_settings:
    default_epochs: 3
    default_learning_rate: 2e-5
    default_batch_size: 4
    max_sequence_length: 512

# モデル管理設定
model_management:
  auto_detect_finetuned: true
  include_base_models: true
  cache_enabled: true
EOF

echo "✅ 継続学習管理システムの設定を完了"

# 実行モードを確認（デフォルト: development）
MODE="${1:-development}"

# Webサーバーを起動
echo "🌐 統合Webサーバーを起動中 (モード: $MODE)..."
echo "📊 利用可能な機能:"
echo "  - メインダッシュボード: http://localhost:8050/"
echo "  - ファインチューニング: http://localhost:8050/finetune"
echo "  - 継続学習管理: http://localhost:8050/continual"
echo "  - RAGシステム: http://localhost:8050/rag"
echo "  - モデル管理: http://localhost:8050/models"

# モードに応じてuvicornオプションを設定
if [ "$MODE" = "production" ]; then
    echo ""
    echo "⚠️  本番モード: 自動リロード無効"
    echo "📌 llama.cppビルドや量子化処理中の再起動を防ぎます"
    # 本番モード: リロード無効、llama.cppディレクトリを除外
    exec python3 -m uvicorn app.main_unified:app \
        --host 0.0.0.0 \
        --port 8050 \
        --workers 1 \
        --log-level info
else
    echo ""
    echo "🔄 開発モード: 自動リロード有効"
    echo "⚠️  量子化処理を実行する場合は production モードを推奨"
    echo "   使用方法: ./start_web_interface.sh production"
    # 開発モード: リロード有効だがllama.cppを除外
    exec python3 -m uvicorn app.main_unified:app \
        --host 0.0.0.0 \
        --port 8050 \
        --reload \
        --reload-exclude ".*llama\.cpp.*" \
        --reload-exclude ".*outputs.*" \
        --reload-exclude ".*\.gguf$" \
        --reload-exclude ".*\.safetensors$" \
        --reload-exclude ".*\.bin$" \
        --reload-exclude ".*\.pt$" \
        --reload-exclude ".*\.pth$"
fi
