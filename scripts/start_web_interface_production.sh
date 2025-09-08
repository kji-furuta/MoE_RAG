#!/bin/bash

# 本番環境用Webインターフェース起動スクリプト
# 自動リロードを無効化し、llama.cppディレクトリを除外

echo "================================"
echo "AI Fine-tuning Toolkit - Production Mode"
echo "================================"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 環境チェック
check_environment() {
    echo -e "${YELLOW}環境チェック中...${NC}"
    
    # Pythonチェック
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3が見つかりません${NC}"
        exit 1
    fi
    
    # 必要なディレクトリチェック
    if [ ! -d "/workspace/app" ]; then
        echo -e "${RED}❌ /workspace/appディレクトリが見つかりません${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 環境チェック完了${NC}"
}

# ポートチェック
check_port() {
    if lsof -Pi :8050 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}⚠️  ポート8050は既に使用中です${NC}"
        echo "既存のプロセスを停止しますか？ (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            echo "既存プロセスを停止中..."
            lsof -ti:8050 | xargs kill -9 2>/dev/null
            sleep 2
        else
            echo "終了します"
            exit 1
        fi
    fi
}

# Ollamaサービスチェック
check_ollama() {
    echo -e "${YELLOW}Ollamaサービスをチェック中...${NC}"
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Ollamaサービスが起動していません${NC}"
        echo "Ollamaを起動しますか？ (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            echo "Ollamaを起動中..."
            nohup ollama serve > /dev/null 2>&1 &
            sleep 3
        fi
    else
        echo -e "${GREEN}✅ Ollamaサービス稼働中${NC}"
    fi
}

# メイン処理
main() {
    echo ""
    echo "Production Mode - 自動リロード無効"
    echo "================================"
    
    check_environment
    check_port
    check_ollama
    
    echo ""
    echo -e "${GREEN}Webインターフェースを起動中...${NC}"
    echo "URL: http://localhost:8050"
    echo ""
    echo "停止するには Ctrl+C を押してください"
    echo ""
    
    # 作業ディレクトリに移動
    cd /workspace || exit 1
    
    # PYTHONPATHを設定
    export PYTHONPATH=/workspace:$PYTHONPATH
    
    # 本番モードで起動（リロード無効、ワーカー数指定）
    exec python3 -m uvicorn app.main_unified:app \
        --host 0.0.0.0 \
        --port 8050 \
        --workers 1 \
        --log-level info \
        --no-access-log
}

# シグナルハンドラー
trap 'echo -e "\n${YELLOW}シャットダウン中...${NC}"; exit 0' INT TERM

# メイン実行
main