#!/bin/bash

# Ollamaモデルの初期化スクリプト
# システム起動時に必要なOllamaモデルを自動的に登録

echo "========================================="
echo "Initializing Ollama models..."
echo "========================================="

# Ollamaサービスが起動するまで待機
echo "Waiting for Ollama service to be ready..."
for i in {1..30}; do
    if ollama list &>/dev/null; then
        echo "Ollama service is ready!"
        break
    fi
    echo "Waiting for Ollama... (attempt $i/30)"
    sleep 2
done

# DeepSeek-32B ファインチューニング済みモデルの登録
DEEPSEEK_MODEL="/workspace/models/deepseek-32b-finetuned.gguf"
DEEPSEEK_MODELFILE="/workspace/models/Modelfile_finetuned"

if [ -f "$DEEPSEEK_MODEL" ] && [ -f "$DEEPSEEK_MODELFILE" ]; then
    echo "Checking DeepSeek-32B finetuned model..."
    
    # モデルが既に登録されているか確認
    if ! ollama list | grep -q "deepseek-32b-finetuned"; then
        echo "Registering DeepSeek-32B finetuned model..."
        cd /workspace/models
        ollama create deepseek-32b-finetuned -f Modelfile_finetuned
        
        if [ $? -eq 0 ]; then
            echo "✅ DeepSeek-32B finetuned model registered successfully!"
        else
            echo "❌ Failed to register DeepSeek-32B finetuned model"
        fi
    else
        echo "✅ DeepSeek-32B finetuned model already registered"
    fi
else
    echo "⚠️  DeepSeek-32B model files not found. Skipping..."
fi

# GPT-NeoX-20Bモデルの確認と取得
echo "Checking GPT-NeoX-20B model..."
# Dockerコンテナ内で実行されているか確認
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    # Dockerコンテナ内の場合、直接ollamaコマンドを使用
    if ollama list 2>/dev/null | grep -q "gpt-neox:20b"; then
        echo "✅ GPT-NeoX-20B model already available in Ollama"
    else
        echo "ℹ️  GPT-NeoX-20B not found in Ollama. Checking if GGUF file exists..."
        # GGUFファイルが存在する場合は登録を試みる
        if [ -f "/workspace/models/gpt-neox-20b.Q4_K_M.gguf" ]; then
            echo "Found GGUF file. Registering in Ollama..."
            # Modelfileを作成
            cat > /workspace/models/Modelfile_gpt_neox_20b << EOF
FROM /workspace/models/gpt-neox-20b.Q4_K_M.gguf

# GPT-NeoX-20B Base Model (TensorBlock GGUF)
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are GPT-NeoX, a 20B parameter language model trained by EleutherAI. 
You are designed to be helpful, harmless, and honest in your responses."""
EOF
            # Ollamaに登録
            cd /workspace/models
            ollama create gpt-neox:20b -f Modelfile_gpt_neox_20b
            if [ $? -eq 0 ]; then
                echo "✅ GPT-NeoX-20B registered in Ollama successfully!"
            else
                echo "⚠️  Failed to register GPT-NeoX-20B in Ollama"
            fi
        else
            echo "ℹ️  GPT-NeoX-20B GGUF file not found. TensorBlock version can be downloaded."
            echo "    URL: https://huggingface.co/tensorblock/gpt-neox-20b-GGUF/"
        fi
    fi
else
    # ホストマシンから実行されている場合
    if docker exec ai-ft-container ollama list 2>/dev/null | grep -q "gpt-neox:20b"; then
        echo "✅ GPT-NeoX-20B model already available in Ollama (Docker container)"
    else
        echo "ℹ️  GPT-NeoX-20B not found in Docker container's Ollama"
        echo "    Run inside container to register: docker exec ai-ft-container bash scripts/init_ollama_models.sh"
    fi
fi

# Llama 3.2 3Bモデルの確認と取得
echo "Checking Llama 3.2 3B model..."
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "Pulling Llama 3.2 3B model..."
    ollama pull llama3.2:3b
    
    if [ $? -eq 0 ]; then
        echo "✅ Llama 3.2 3B model pulled successfully!"
    else
        echo "❌ Failed to pull Llama 3.2 3B model"
    fi
else
    echo "✅ Llama 3.2 3B model already available"
fi

# その他のカスタムモデルの登録（必要に応じて追加）
# 例: LoRAアダプター統合済みモデルなど
CUSTOM_MODELS_DIR="/workspace/outputs/ollama_models"
if [ -d "$CUSTOM_MODELS_DIR" ]; then
    echo "Checking for custom models in $CUSTOM_MODELS_DIR..."
    for modelfile in "$CUSTOM_MODELS_DIR"/Modelfile_*; do
        if [ -f "$modelfile" ]; then
            model_name=$(basename "$modelfile" | sed 's/Modelfile_//')
            if ! ollama list | grep -q "$model_name"; then
                echo "Registering custom model: $model_name"
                cd "$CUSTOM_MODELS_DIR"
                ollama create "$model_name" -f "$(basename "$modelfile")"
            fi
        fi
    done
fi

echo "========================================="
echo "Ollama model initialization complete!"
echo "Available models:"
ollama list
echo "========================================="