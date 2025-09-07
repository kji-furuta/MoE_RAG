#!/bin/bash

# Ollamaモデルの永続化と確認スクリプト
# systemdサービスやDockerコンテナ起動時に実行

echo "========================================="
echo "Ollamaモデル確認・復元スクリプト"
echo "========================================="

# Ollamaサービスが起動するまで待機
echo "Ollamaサービスの起動を待機中..."
for i in {1..30}; do
    if ollama list &>/dev/null; then
        echo "✅ Ollamaサービスが起動しました"
        break
    fi
    echo "待機中... ($i/30)"
    sleep 2
done

# 現在のモデル一覧を表示
echo ""
echo "現在登録されているモデル:"
ollama list

# GPT-NeoX-20Bの確認と登録
echo ""
echo "GPT-NeoX-20Bモデルを確認中..."
if ! ollama list | grep -q "gpt-neox:20b"; then
    echo "GPT-NeoX-20Bが見つかりません。登録を試みます..."
    
    # GGUFファイルの存在確認
    if [ -f "/workspace/models/gpt-neox-20b.Q4_K_M.gguf" ]; then
        # Modelfileが存在しない場合は作成
        if [ ! -f "/workspace/models/Modelfile_gpt_neox_20b" ]; then
            cat > /workspace/models/Modelfile_gpt_neox_20b << 'EOF'
FROM /workspace/models/gpt-neox-20b.Q4_K_M.gguf

# GPT-NeoX-20B Base Model (TensorBlock GGUF)
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are GPT-NeoX, a 20B parameter language model trained by EleutherAI. 
You are designed to be helpful, harmless, and honest in your responses."""
EOF
        fi
        
        # Ollamaに登録
        cd /workspace/models
        ollama create gpt-neox:20b -f Modelfile_gpt_neox_20b
        
        if [ $? -eq 0 ]; then
            echo "✅ GPT-NeoX-20Bを正常に登録しました"
        else
            echo "❌ GPT-NeoX-20Bの登録に失敗しました"
        fi
    else
        echo "⚠️ GPT-NeoX-20B GGUFファイルが見つかりません"
    fi
else
    echo "✅ GPT-NeoX-20Bは既に登録されています"
fi

# 最新のLoRAモデルを検索して登録
echo ""
echo "LoRAモデルを確認中..."
LATEST_LORA=$(ls -dt /workspace/outputs/lora_* 2>/dev/null | head -1)

if [ -n "$LATEST_LORA" ] && [ -d "$LATEST_LORA" ]; then
    if [ -f "$LATEST_LORA/training_info.json" ]; then
        TIMESTAMP=$(basename "$LATEST_LORA" | sed 's/lora_//')
        MODEL_NAME="gpt-neox-20b-lora-$TIMESTAMP"
        
        if ! ollama list | grep -q "$MODEL_NAME"; then
            echo "最新のLoRAモデルを登録中: $MODEL_NAME"
            
            # register_lora_to_ollama.shが存在する場合は使用
            if [ -f "/workspace/scripts/register_lora_to_ollama.sh" ]; then
                echo "y" | /workspace/scripts/register_lora_to_ollama.sh "$LATEST_LORA"
            else
                echo "⚠️ LoRA登録スクリプトが見つかりません"
            fi
        else
            echo "✅ 最新のLoRAモデルは既に登録されています: $MODEL_NAME"
        fi
    fi
fi

# DeepSeek-32Bファインチューニングモデルの確認
echo ""
echo "DeepSeek-32Bファインチューニングモデルを確認中..."
if ! ollama list | grep -q "deepseek-32b-finetuned"; then
    if [ -f "/workspace/models/deepseek-32b-finetuned.gguf" ] && [ -f "/workspace/models/Modelfile_finetuned" ]; then
        echo "DeepSeek-32Bファインチューニングモデルを登録中..."
        cd /workspace/models
        ollama create deepseek-32b-finetuned -f Modelfile_finetuned
    fi
else
    echo "✅ DeepSeek-32Bファインチューニングモデルは既に登録されています"
fi

# 最終的なモデル一覧を表示
echo ""
echo "========================================="
echo "最終的な登録モデル一覧:"
ollama list
echo "========================================="

# RAG設定の確認
echo ""
echo "RAG設定の現在のOllamaモデル:"
grep "ollama_model:" /workspace/src/rag/config/rag_config.yaml | head -1

echo ""
echo "✅ Ollamaモデル確認・復元完了"