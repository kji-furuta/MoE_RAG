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