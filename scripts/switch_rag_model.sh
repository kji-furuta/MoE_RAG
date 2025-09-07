#!/bin/bash

# RAGシステムのモデルを切り替えるスクリプト
# Usage: ./switch_rag_model.sh [model_name]

MODEL_NAME=${1:-"deepseek-32b-finetuned:latest"}
CONFIG_FILE="/workspace/src/rag/config/rag_config.yaml"

echo "=========================================="
echo "RAG Model Switcher"
echo "=========================================="
echo "Switching to model: $MODEL_NAME"

# Dockerコンテナ内で設定を更新
docker exec ai-ft-container bash -c "
    # YAMLファイルを更新
    sed -i \"s|model: .*|model: $MODEL_NAME|g\" $CONFIG_FILE
    sed -i \"s|ollama_model: .*|ollama_model: $MODEL_NAME|g\" $CONFIG_FILE
    
    echo 'Configuration updated successfully.'
    echo 'Current settings:'
    grep -E '(model:|ollama_model:)' $CONFIG_FILE | head -5
"

echo ""
echo "Model switched to: $MODEL_NAME"
echo "The change will take effect immediately for new requests."
echo ""
echo "Available models:"
echo "  - deepseek-32b-finetuned:latest"
echo "  - llama3.2:3b"
echo "  - gpt-neox-20b-finetuned:latest"
echo ""
echo "Usage: $0 [model_name]"