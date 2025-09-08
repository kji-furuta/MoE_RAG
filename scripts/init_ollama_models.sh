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