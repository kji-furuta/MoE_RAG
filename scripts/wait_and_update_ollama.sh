#!/bin/bash

# Ollamaが完全に起動するまで待機してからモデルリストを更新するスクリプト

echo "⏳ Waiting for Ollama service to fully initialize..."

# Ollamaサービスが起動するまで待機（最大30秒）
MAX_WAIT=30
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if pgrep -x ollama > /dev/null 2>&1; then
        # サービスが起動してもすぐには使えないので、実際に応答するか確認
        if ollama list > /dev/null 2>&1; then
            echo "✅ Ollama service is ready"
            break
        fi
    fi
    echo "Waiting for Ollama... ($WAIT_COUNT/$MAX_WAIT)"
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "⚠️ Timeout waiting for Ollama service"
    exit 1
fi

# さらに2秒待機してOllamaが完全に初期化されるのを確実にする
sleep 2

# Ollamaモデルリストを更新
echo "🔄 Updating Ollama models configuration..."
python3 /workspace/scripts/update_ollama_models_config.py

if [ $? -eq 0 ]; then
    echo "✅ Successfully updated Ollama models configuration"
else
    echo "❌ Failed to update Ollama models configuration"
    exit 1
fi