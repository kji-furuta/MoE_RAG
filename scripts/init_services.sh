#!/bin/bash

# サービス初期化スクリプト - 適切な順序でサービスを起動

echo "🚀 Initializing services..."

# ログディレクトリの作成
mkdir -p /workspace/logs

# 1. Ollamaサービスを起動
if command -v ollama &> /dev/null; then
    echo "🤖 Starting Ollama service..."

    # 既存のOllamaプロセスを確認
    if pgrep -x ollama > /dev/null 2>&1; then
        echo "✅ Ollama is already running"
    else
        nohup ollama serve > /workspace/logs/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo "Started Ollama with PID: $OLLAMA_PID"

        # Ollamaが完全に起動するまで待機
        echo "⏳ Waiting for Ollama to be ready..."
        MAX_WAIT=30
        WAIT_COUNT=0

        while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
            if ollama list > /dev/null 2>&1; then
                echo "✅ Ollama is ready"
                break
            fi
            sleep 1
            WAIT_COUNT=$((WAIT_COUNT + 1))
            if [ $((WAIT_COUNT % 5)) -eq 0 ]; then
                echo "Still waiting... ($WAIT_COUNT/$MAX_WAIT)"
            fi
        done

        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "⚠️ Ollama did not start properly within $MAX_WAIT seconds"
        fi
    fi

    # デフォルトモデルの確認とダウンロード
    echo "📦 Checking Ollama models..."
    if ! ollama list | grep -q "llama3.2:3b"; then
        echo "📥 Downloading llama3.2:3b model..."
        ollama pull llama3.2:3b
        echo "✅ Model download complete"
    fi
fi

# 2. GGUFモデルレジストリの初期化
if [ -f /workspace/scripts/init_gguf_models.py ]; then
    echo "📚 Initializing GGUF model registry..."
    python3 /workspace/scripts/init_gguf_models.py
    echo "✅ GGUF registry initialized"
fi

# 3. Ollamaモデル設定の更新
if [ -f /workspace/scripts/update_ollama_models_config.py ]; then
    echo "🔄 Updating Ollama model configuration..."
    python3 /workspace/scripts/update_ollama_models_config.py

    if [ $? -eq 0 ]; then
        echo "✅ Ollama model configuration updated"
    else
        echo "⚠️ Failed to update Ollama model configuration"
    fi
fi

# 4. パーミッションの確認
if [ -f /workspace/scripts/setup_permissions.sh ]; then
    echo "🔐 Checking permissions..."
    /workspace/scripts/setup_permissions.sh --check
    if [ $? -ne 0 ]; then
        echo "🔧 Fixing permissions..."
        /workspace/scripts/setup_permissions.sh
    fi
fi

echo "✅ Service initialization complete"
echo ""
echo "📊 Ready to start web interface:"
echo "   Run: python3 -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload"