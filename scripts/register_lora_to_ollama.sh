#!/bin/bash

# LoRAアダプターをOllamaに登録するスクリプト
# 使用方法: ./register_lora_to_ollama.sh <lora_output_dir>

LORA_DIR="${1:-}"
if [ -z "$LORA_DIR" ]; then
    echo "使用方法: $0 <lora_output_directory>"
    echo "例: $0 outputs/lora_20250906_104924"
    exit 1
fi

# 絶対パスに変換
if [[ "$LORA_DIR" != /* ]]; then
    LORA_DIR="/workspace/$LORA_DIR"
fi

if [ ! -d "$LORA_DIR" ]; then
    echo "エラー: ディレクトリが見つかりません: $LORA_DIR"
    exit 1
fi

# training_info.jsonからベースモデル情報を取得
if [ ! -f "$LORA_DIR/training_info.json" ]; then
    echo "エラー: training_info.jsonが見つかりません"
    exit 1
fi

# JSONからベースモデルを抽出
BASE_MODEL=$(python3 -c "import json; print(json.load(open('$LORA_DIR/training_info.json'))['base_model'])" 2>/dev/null)
TIMESTAMP=$(python3 -c "import json; print(json.load(open('$LORA_DIR/training_info.json'))['timestamp'])" 2>/dev/null)

if [ -z "$BASE_MODEL" ]; then
    echo "エラー: ベースモデル情報を取得できません"
    exit 1
fi

echo "========================================="
echo "LoRAアダプターをOllamaに登録"
echo "========================================="
echo "LoRAディレクトリ: $LORA_DIR"
echo "ベースモデル: $BASE_MODEL"
echo "タイムスタンプ: $TIMESTAMP"

# モデル名を生成（ベースモデル名から生成）
MODEL_NAME=$(echo "$BASE_MODEL" | sed 's/.*\///g' | tr '[:upper:]' '[:lower:]')-lora-$TIMESTAMP

# ベースモデルに応じたGGUFファイルを特定
case "$BASE_MODEL" in
    "EleutherAI/gpt-neox-20b")
        GGUF_FILE="/workspace/models/gpt-neox-20b.Q4_K_M.gguf"
        if [ ! -f "$GGUF_FILE" ]; then
            echo "エラー: GPT-NeoX-20B GGUFファイルが見つかりません"
            echo "ファイルパス: $GGUF_FILE"
            exit 1
        fi
        ;;
    "cyberagent/calm3-22b-chat")
        GGUF_FILE="/workspace/models/calm3-22b-chat.Q4_K_M.gguf"
        ;;
    "Qwen/Qwen2.5-32B-Instruct")
        GGUF_FILE="/workspace/models/qwen2.5-32b-instruct.Q4_K_M.gguf"
        ;;
    *)
        echo "警告: ベースモデル '$BASE_MODEL' のGGUFファイルが不明です"
        echo "手動でGGUFファイルを指定してください"
        exit 1
        ;;
esac

# Modelfileを作成
MODELFILE="/tmp/Modelfile_${MODEL_NAME}"
cat > "$MODELFILE" << EOF
FROM $GGUF_FILE

# LoRA Adapter Applied Model
# Base: $BASE_MODEL
# LoRA: $LORA_DIR
# Created: $(date)

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are a helpful AI assistant fine-tuned with LoRA adapters.
Base model: $BASE_MODEL
Training timestamp: $TIMESTAMP"""
EOF

echo ""
echo "Modelfileを作成しました: $MODELFILE"

# Ollamaに登録
echo "Ollamaにモデルを登録中..."
if [ -f /.dockerenv ]; then
    # Dockerコンテナ内で実行
    ollama create "$MODEL_NAME" -f "$MODELFILE"
else
    # ホストから実行
    docker exec ai-ft-container ollama create "$MODEL_NAME" -f "$MODELFILE"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ モデルが正常に登録されました: $MODEL_NAME"
    echo ""
    echo "使用方法:"
    echo "  ollama run $MODEL_NAME"
    echo ""
    echo "RAG設定ファイルに追加:"
    echo "  ollama_model: $MODEL_NAME:latest"
    
    # RAG設定ファイルを自動更新するオプション
    echo ""
    read -p "RAG設定ファイルを自動更新しますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RAG_CONFIG="/workspace/src/rag/config/rag_config.yaml"
        if [ -f "$RAG_CONFIG" ]; then
            # バックアップを作成
            cp "$RAG_CONFIG" "${RAG_CONFIG}.bak"
            
            # ollama_modelを更新
            sed -i "s/ollama_model:.*/ollama_model: $MODEL_NAME:latest/" "$RAG_CONFIG"
            sed -i "s/model: .*/model: $MODEL_NAME:latest/" "$RAG_CONFIG"
            
            echo "✅ RAG設定ファイルを更新しました"
            echo "   バックアップ: ${RAG_CONFIG}.bak"
        fi
    fi
else
    echo "❌ モデルの登録に失敗しました"
    exit 1
fi

# 登録済みモデル一覧を表示
echo ""
echo "現在登録されているモデル:"
if [ -f /.dockerenv ]; then
    ollama list
else
    docker exec ai-ft-container ollama list
fi