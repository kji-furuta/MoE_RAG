#!/bin/bash

# Docker Entrypoint Script
# Ollamaモデルの初期化とその他のスタートアップタスクを実行

echo "========================================="
echo "Starting AI-FT Container..."
echo "========================================="

# Ollamaサービスの起動
if command -v ollama &> /dev/null; then
    echo "Starting Ollama service..."
    nohup ollama serve > /var/log/ollama.log 2>&1 &
    
    # Ollamaが起動するまで待機
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            echo "✅ Ollama service is ready!"
            break
        fi
        echo "Waiting for Ollama... (attempt $i/30)"
        sleep 2
    done
    
    # Ollamaモデルの初期化
    if [ -f /workspace/scripts/init_ollama_models.sh ]; then
        echo "Initializing Ollama models..."
        /workspace/scripts/init_ollama_models.sh
    fi
fi

# パーミッションの設定（必要に応じて）
if [ -f /workspace/scripts/setup_permissions.sh ]; then
    /workspace/scripts/setup_permissions.sh --check
    if [ $? -ne 0 ]; then
        echo "Setting up permissions..."
        /workspace/scripts/setup_permissions.sh
    fi
fi

# 継続学習用ディレクトリの作成
mkdir -p /workspace/data/continual_learning
mkdir -p /workspace/outputs/ewc_data
mkdir -p /workspace/outputs/continual_task

echo "========================================="
echo "Container initialization complete!"
echo "========================================="

# コマンドが指定されていない場合はbashを起動
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi