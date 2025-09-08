#!/bin/bash

# DeepSeek-R1-Distill-Qwen-32B GGUF モデル初期化スクリプト
# 高性能な日本語対応モデルのダウンロードと設定

echo "🚀 DeepSeek-R1-Distill-Qwen-32B モデル初期化を開始..."

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# モデル情報
MODEL_URL="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
MODEL_DIR="/workspace/models/gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# ディレクトリ作成
mkdir -p "${MODEL_DIR}"

# モデルのダウンロード状況確認
check_model() {
    if [ -f "${MODEL_PATH}" ]; then
        # ファイルサイズ確認（約20GB）
        local size=$(stat -c%s "${MODEL_PATH}" 2>/dev/null)
        if [ "$size" -gt 10000000000 ]; then  # 10GB以上
            echo -e "${GREEN}✅ DeepSeekモデルは既にダウンロード済みです${NC}"
            echo "   場所: ${MODEL_PATH}"
            echo "   サイズ: $(du -h ${MODEL_PATH} | cut -f1)"
            return 0
        else
            echo -e "${YELLOW}⚠️ モデルファイルが不完全です。再ダウンロードします...${NC}"
            rm -f "${MODEL_PATH}"
        fi
    fi
    return 1
}

# モデルのダウンロード
download_model() {
    echo -e "${BLUE}📥 DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf をダウンロード中...${NC}"
    echo "   サイズ: 約20GB"
    echo "   これには時間がかかる場合があります..."
    
    # wgetまたはcurlでダウンロード（進捗表示付き）
    if command -v wget &> /dev/null; then
        wget --show-progress -O "${MODEL_PATH}" "${MODEL_URL}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${MODEL_PATH}" "${MODEL_URL}"
    else
        echo -e "${RED}❌ wgetまたはcurlが必要です${NC}"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ モデルのダウンロードが完了しました${NC}"
        return 0
    else
        echo -e "${RED}❌ モデルのダウンロードに失敗しました${NC}"
        return 1
    fi
}

# Ollamaカスタムモデルファイルの作成
create_ollama_modelfile() {
    local modelfile_path="/workspace/ollama_models/deepseek-32b-japanese.Modelfile"
    
    echo -e "${BLUE}📝 Ollama用Modelfileを作成中...${NC}"
    
    mkdir -p /workspace/ollama_models
    
    cat > "${modelfile_path}" << 'EOF'
FROM /workspace/models/gguf/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """
あなたは高度な日本語理解能力を持つAIアシスタントです。
正確で分かりやすい回答を提供し、必要に応じて詳細な説明や例を含めます。
専門的な内容でも、ユーザーのレベルに合わせて適切に説明します。
"""

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""
EOF
    
    echo -e "${GREEN}✅ Modelfileを作成しました: ${modelfile_path}${NC}"
    
    # Ollamaにモデルを登録
    if command -v ollama &> /dev/null; then
        echo -e "${BLUE}🤖 Ollamaにモデルを登録中...${NC}"
        ollama create deepseek-32b-japanese -f "${modelfile_path}"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Ollamaモデル 'deepseek-32b-japanese' を登録しました${NC}"
        fi
    fi
}

# メイン処理
main() {
    echo "================================"
    echo "DeepSeek-32B モデル管理"
    echo "================================"
    
    # モデルの確認
    if ! check_model; then
        # ダウンロード
        if ! download_model; then
            echo -e "${RED}初期化に失敗しました${NC}"
            exit 1
        fi
    fi
    
    # Ollama Modelfileの作成
    create_ollama_modelfile
    
    echo ""
    echo "================================"
    echo -e "${GREEN}✅ DeepSeekモデルの初期化が完了しました${NC}"
    echo ""
    echo "使用方法:"
    echo "  1. RAGシステムで使用: モデル選択で 'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf' を選択"
    echo "  2. Ollamaで使用: ollama run deepseek-32b-japanese"
    echo "  3. LoRA適用: apply_lora_to_gguf_improved.py で --base-model-name に指定"
    echo ""
    echo "モデルパス: ${MODEL_PATH}"
    echo "================================"
}

# スクリプト実行
main "$@"