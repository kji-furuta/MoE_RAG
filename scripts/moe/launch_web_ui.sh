#!/bin/bash

# MoE Web UI Launcher
# MoE WebUIとAPIの起動スクリプト

set -e

echo "==========================================="
echo "🏗️ MoE 土木・建設AI システム"
echo "Web UI & API Server Launcher"
echo "==========================================="

# プロジェクトルート
PROJECT_ROOT="/home/kjifu/AI_FT_7"
cd $PROJECT_ROOT

# 仮想環境のアクティベート
echo "仮想環境をアクティベート中..."
source venv/bin/activate

# 必要なパッケージの確認とインストール
echo "必要なパッケージを確認中..."

# Streamlitのインストール確認
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlitをインストール中..."
    pip install streamlit --quiet
fi

# Plotlyのインストール確認
if ! python -c "import plotly" 2>/dev/null; then
    echo "Plotlyをインストール中..."
    pip install plotly --quiet
fi

# FastAPIのインストール確認
if ! python -c "import fastapi" 2>/dev/null; then
    echo "FastAPIをインストール中..."
    pip install fastapi uvicorn --quiet
fi

# 起動モードの選択
MODE=${1:-"ui"}  # ui, api, or both

case $MODE in
    "ui")
        echo ""
        echo "📊 Streamlit Web UIを起動中..."
        echo "ブラウザで http://localhost:8501 にアクセスしてください"
        echo ""
        streamlit run app/moe_web_ui.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --theme.primaryColor "#764ba2" \
            --theme.backgroundColor "#FFFFFF" \
            --theme.secondaryBackgroundColor "#F0F2F6" \
            --theme.textColor "#262730"
        ;;
    
    "api")
        echo ""
        echo "🚀 FastAPI サーバーを起動中..."
        echo "API Docs: http://localhost:8000/docs"
        echo ""
        python -m uvicorn app.moe_api:app \
            --host 0.0.0.0 \
            --port 8000 \
            --reload
        ;;
    
    "both")
        echo ""
        echo "📊 Web UIとAPIを同時起動中..."
        echo ""
        
        # APIをバックグラウンドで起動
        echo "APIサーバーをバックグラウンドで起動..."
        python -m uvicorn app.moe_api:app \
            --host 0.0.0.0 \
            --port 8000 \
            --reload &
        API_PID=$!
        echo "API PID: $API_PID"
        
        # 少し待機
        sleep 3
        
        # UIを起動
        echo "Web UIを起動..."
        echo ""
        echo "================================"
        echo "📊 Web UI: http://localhost:8501"
        echo "🚀 API Docs: http://localhost:8000/docs"
        echo "================================"
        echo ""
        
        # トラップ設定（終了時にAPIも停止）
        trap "echo 'Stopping API server...'; kill $API_PID 2>/dev/null" EXIT
        
        streamlit run app/moe_web_ui.py \
            --server.port 8501 \
            --server.address 0.0.0.0
        ;;
    
    *)
        echo "使用方法: $0 [ui|api|both]"
        echo "  ui   - Streamlit Web UIのみ起動"
        echo "  api  - FastAPI サーバーのみ起動"
        echo "  both - 両方を起動（デフォルト: ui）"
        exit 1
        ;;
esac
