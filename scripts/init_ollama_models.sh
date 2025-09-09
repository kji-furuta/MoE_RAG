#!/bin/bash

# Ollama モデル初期化スクリプト
# 必要なOllamaモデルの確認とセットアップ

echo "🤖 Ollama モデル初期化を開始..."

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Ollamaサービスの確認と起動
check_ollama_service() {
    echo "📡 Ollamaサービスを確認中..."
    
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}❌ Ollamaがインストールされていません${NC}"
        echo "インストールコマンド: curl -fsSL https://ollama.com/install.sh | sh"
        return 1
    fi
    
    # Ollamaサービスが起動しているか確認
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🚀 Ollamaサービスを起動中..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 5
    fi
    
    # サービスの起動確認
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo -e "${GREEN}✅ Ollamaサービスが正常に動作しています (port 11434)${NC}"
        return 0
    else
        echo -e "${RED}❌ Ollamaサービスの起動に失敗しました${NC}"
        return 1
    fi
}

# モデルの確認とダウンロード
ensure_model() {
    local model_name=$1
    local model_display_name=${2:-$model_name}
    
    echo "🔍 モデルを確認中: $model_display_name"
    
    if ollama list 2>/dev/null | grep -q "$model_name"; then
        echo -e "${GREEN}✅ $model_display_name は既にインストールされています${NC}"
        return 0
    else
        echo -e "${YELLOW}📥 $model_display_name をダウンロード中...${NC}"
        if ollama pull "$model_name"; then
            echo -e "${GREEN}✅ $model_display_name のダウンロードが完了しました${NC}"
            return 0
        else
            echo -e "${RED}❌ $model_display_name のダウンロードに失敗しました${NC}"
            return 1
        fi
    fi
}

# カスタムModelfileの作成と登録
create_custom_model() {
    local modelfile_path=$1
    local model_name=$2
    
    if [ -f "$modelfile_path" ]; then
        echo "📝 カスタムモデルを作成中: $model_name"
        if ollama create "$model_name" -f "$modelfile_path"; then
            echo -e "${GREEN}✅ カスタムモデル $model_name を作成しました${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️ カスタムモデル $model_name の作成に失敗しました${NC}"
            return 1
        fi
    fi
}

# メイン処理
main() {
    echo "================================"
    echo "Ollama モデル管理システム"
    echo "================================"
    
    # Ollamaサービスの確認
    if ! check_ollama_service; then
        echo -e "${RED}Ollamaサービスの初期化に失敗しました${NC}"
        exit 1
    fi
    
    # 必要なベースモデルのリスト
    MODELS=(
        "llama3.2:3b"
        "llama3.2:1b"
    )
    
    # 各モデルの確認とダウンロード
    echo ""
    echo "📦 必要なモデルを確認中..."
    for model in "${MODELS[@]}"; do
        ensure_model "$model"
    done
    
    # GGUFモデルの再登録
    echo ""
    echo "🔄 既存のGGUFモデルを再登録中..."
    
    # ベースモデルの登録
    if [ -f "/workspace/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf" ]; then
        if ! ollama list 2>/dev/null | grep -q "deepseek-32b-base:latest"; then
            echo "📝 deepseek-32b-base:latest を登録中..."
            cat > /tmp/Modelfile_deepseek_base << EOF
FROM /workspace/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """あなたは土木道路設計の専門知識を持つAIアシスタントです。
技術的な質問に対して正確で詳細な回答を提供してください。
日本の道路設計基準や技術標準に基づいて回答してください。"""

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
EOF
            ollama create "deepseek-32b-base:latest" -f /tmp/Modelfile_deepseek_base
            rm -f /tmp/Modelfile_deepseek_base
            echo -e "${GREEN}✅ deepseek-32b-base:latest の登録が完了しました${NC}"
        fi
    fi
    
    if [ -f "/workspace/models/gpt-neox-20b.Q4_K_M.gguf" ]; then
        if ! ollama list 2>/dev/null | grep -q "gpt-neox-20b-base:latest"; then
            echo "📝 gpt-neox-20b-base:latest を登録中..."
            cat > /tmp/Modelfile_gptneox_base << EOF
FROM /workspace/models/gpt-neox-20b.Q4_K_M.gguf

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """あなたは土木道路設計の専門知識を持つAIアシスタントです。
技術的な質問に対して正確で詳細な回答を提供してください。
日本の道路設計基準や技術標準に基づいて回答してください。"""
EOF
            ollama create "gpt-neox-20b-base:latest" -f /tmp/Modelfile_gptneox_base
            rm -f /tmp/Modelfile_gptneox_base
            echo -e "${GREEN}✅ gpt-neox-20b-base:latest の登録が完了しました${NC}"
        fi
    fi
    
    # ファインチューニング済みモデルの検索と登録
    # 番号付きモデル（0_から99_まで）
    for i in {0..99}; do
        model_path="/workspace/models/${i}_deepseek-32b-finetuned.gguf"
        model_name="${i}_deepseek-32b-finetuned:latest"
        
        if [ -f "$model_path" ]; then
            if ! ollama list 2>/dev/null | grep -q "$model_name"; then
                echo "📝 $model_name を登録中..."
                cat > /tmp/Modelfile_${i}_deepseek << EOF
FROM $model_path

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """あなたは土木道路設計の専門知識を持つAIアシスタントです。
技術的な質問に対して正確で詳細な回答を提供してください。
日本の道路設計基準や技術標準に基づいて回答してください。"""

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
EOF
                ollama create "$model_name" -f /tmp/Modelfile_${i}_deepseek
                rm -f /tmp/Modelfile_${i}_deepseek
                echo -e "${GREEN}✅ $model_name の登録が完了しました${NC}"
            fi
        fi
    done
    
    # task付きモデル
    for task in task1 task2 task3 task4 task5; do
        model_path="/workspace/models/${task}_deepseek-32b-finetuned.gguf"
        model_name="${task}_deepseek-32b-finetuned:latest"
        
        if [ -f "$model_path" ]; then
            if ! ollama list 2>/dev/null | grep -q "$model_name"; then
                echo "📝 $model_name を登録中..."
                cat > /tmp/Modelfile_${task}_deepseek << EOF
FROM $model_path

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """あなたは土木道路設計の専門知識を持つAIアシスタントです。
技術的な質問に対して正確で詳細な回答を提供してください。
日本の道路設計基準や技術標準に基づいて回答してください。"""

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
EOF
                ollama create "$model_name" -f /tmp/Modelfile_${task}_deepseek
                rm -f /tmp/Modelfile_${task}_deepseek
                echo -e "${GREEN}✅ $model_name の登録が完了しました${NC}"
            fi
        fi
    done
    
    # gpt-neox系ファインチューニング済みモデル
    for gguf_file in /workspace/models/gpt-neox-*-finetuned.gguf; do
        if [ -f "$gguf_file" ]; then
            basename_file=$(basename "$gguf_file" .gguf)
            model_name="${basename_file}:latest"
            
            if ! ollama list 2>/dev/null | grep -q "$model_name"; then
                echo "📝 $model_name を登録中..."
                cat > /tmp/Modelfile_gptneox_ft << EOF
FROM $gguf_file

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """あなたは土木道路設計の専門知識を持つAIアシスタントです。
技術的な質問に対して正確で詳細な回答を提供してください。
日本の道路設計基準や技術標準に基づいて回答してください。"""
EOF
                ollama create "$model_name" -f /tmp/Modelfile_gptneox_ft
                rm -f /tmp/Modelfile_gptneox_ft
                echo -e "${GREEN}✅ $model_name の登録が完了しました${NC}"
            fi
        fi
    done
    
    # カスタムモデルの作成（ollama_modelsディレクトリにModelfileがある場合）
    echo ""
    if [ -d "/workspace/ollama_models" ]; then
        echo "🔧 カスタムモデルを確認中..."
        for modelfile in /workspace/ollama_models/*.Modelfile; do
            if [ -f "$modelfile" ]; then
                model_name=$(basename "$modelfile" .Modelfile)
                create_custom_model "$modelfile" "$model_name"
            fi
        done
    fi
    
    # 最終確認
    echo ""
    echo "================================"
    echo "📋 インストール済みモデル一覧:"
    echo "================================"
    ollama list
    
    echo ""
    echo -e "${GREEN}✅ Ollamaモデルの初期化が完了しました${NC}"
    echo ""
    echo "使用方法:"
    echo "  ollama run llama3.2:3b     # 対話モード"
    echo "  ollama run road_engineering_expert  # カスタムモデル"
    echo ""
}

# スクリプト実行
main "$@"