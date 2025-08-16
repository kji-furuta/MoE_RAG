#!/bin/bash

# AI_FT_7 パーミッション設定スクリプト
# このスクリプトは全ての必要なディレクトリの権限を設定します

echo "🔐 パーミッション設定を開始..."

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ユーザー情報
USER_NAME="ai-user"
USER_GROUP="ai-user"

# 必要なディレクトリとその権限設定
declare -A DIRECTORIES=(
    # 書き込み可能ディレクトリ (777)
    ["/workspace/outputs"]="777"
    ["/workspace/temp_uploads"]="777"
    ["/workspace/data/uploaded"]="777"
    ["/workspace/logs"]="777"
    ["/workspace/metadata"]="777"
    ["/workspace/outputs/rag_index"]="777"
    ["/workspace/outputs/rag_index/processed_documents"]="777"
    ["/workspace/qdrant_data"]="777"
    ["/workspace/data/continual_learning"]="777"
    ["/workspace/ollama_modelfiles"]="777"
    
    # 読み取り専用ディレクトリ (755)
    ["/workspace/src"]="755"
    ["/workspace/config"]="755"
    ["/workspace/scripts"]="755"
    ["/workspace/app"]="755"
    ["/workspace/templates"]="755"
    
    # データディレクトリ (775)
    ["/workspace/data"]="775"
    ["/workspace/data/raw"]="775"
    ["/workspace/data/processed"]="775"
    ["/workspace/data/rag_documents"]="775"
    ["/workspace/models"]="775"
    ["/workspace/models/checkpoints"]="775"
)

# ディレクトリ作成と権限設定関数
setup_directory() {
    local dir=$1
    local perm=$2
    
    # ディレクトリが存在しない場合は作成
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✅ Created:${NC} $dir"
    fi
    
    # 所有者を設定（rootユーザーの場合のみ）
    if [ "$EUID" -eq 0 ]; then
        chown -R ${USER_NAME}:${USER_GROUP} "$dir" 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Could not change owner:${NC} $dir"
        }
    fi
    
    # 権限を設定
    chmod -R "$perm" "$dir" 2>/dev/null || {
        echo -e "${RED}❌ Failed to set permissions:${NC} $dir"
        return 1
    }
    
    echo -e "${GREEN}✅ Set permissions ($perm):${NC} $dir"
    return 0
}

# 権限チェック関数
check_permissions() {
    local dir=$1
    local expected_perm=$2
    
    if [ ! -d "$dir" ]; then
        echo -e "${RED}❌ Directory not found:${NC} $dir"
        return 1
    fi
    
    # 現在の権限を取得
    current_perm=$(stat -c "%a" "$dir")
    
    # 所有者を取得
    current_owner=$(stat -c "%U:%G" "$dir")
    
    # 権限チェック
    if [ "$current_perm" != "$expected_perm" ]; then
        echo -e "${YELLOW}⚠️  Permission mismatch:${NC} $dir (current: $current_perm, expected: $expected_perm)"
        return 1
    fi
    
    # 所有者チェック（rootユーザーの場合のみ）
    if [ "$EUID" -eq 0 ] && [ "$current_owner" != "${USER_NAME}:${USER_GROUP}" ]; then
        echo -e "${YELLOW}⚠️  Owner mismatch:${NC} $dir (current: $current_owner, expected: ${USER_NAME}:${USER_GROUP})"
        return 1
    fi
    
    echo -e "${GREEN}✓${NC} $dir [${current_perm}] [${current_owner}]"
    return 0
}

# メイン処理
main() {
    local setup_mode=${1:-"setup"}  # setup または check
    local errors=0
    
    if [ "$setup_mode" == "check" ]; then
        echo "📋 権限チェックモード"
        echo "================================"
        for dir in "${!DIRECTORIES[@]}"; do
            check_permissions "$dir" "${DIRECTORIES[$dir]}" || ((errors++))
        done
        
        echo "================================"
        if [ $errors -eq 0 ]; then
            echo -e "${GREEN}✅ 全てのディレクトリの権限が正しく設定されています${NC}"
        else
            echo -e "${YELLOW}⚠️  $errors 個のディレクトリで権限の問題があります${NC}"
        fi
        
    else
        echo "🔧 権限設定モード"
        echo "================================"
        
        # 全ディレクトリの権限を設定
        for dir in "${!DIRECTORIES[@]}"; do
            setup_directory "$dir" "${DIRECTORIES[$dir]}" || ((errors++))
        done
        
        # Hugging Faceキャッシュディレクトリ
        if [ -d "/home/${USER_NAME}/.cache" ]; then
            chown -R ${USER_NAME}:${USER_GROUP} "/home/${USER_NAME}/.cache" 2>/dev/null
            chmod -R 755 "/home/${USER_NAME}/.cache" 2>/dev/null
            echo -e "${GREEN}✅ Set HF cache permissions${NC}"
        fi
        
        # Jupyter設定ディレクトリ
        if [ -d "/home/${USER_NAME}/.jupyter" ]; then
            chown -R ${USER_NAME}:${USER_GROUP} "/home/${USER_NAME}/.jupyter" 2>/dev/null
            chmod -R 755 "/home/${USER_NAME}/.jupyter" 2>/dev/null
            echo -e "${GREEN}✅ Set Jupyter permissions${NC}"
        fi
        
        echo "================================"
        if [ $errors -eq 0 ]; then
            echo -e "${GREEN}✅ 全ての権限設定が完了しました${NC}"
        else
            echo -e "${YELLOW}⚠️  $errors 個のエラーが発生しました${NC}"
        fi
    fi
    
    return $errors
}

# スクリプト実行
if [ "$1" == "--check" ] || [ "$1" == "-c" ]; then
    main "check"
else
    main "setup"
fi

exit $?