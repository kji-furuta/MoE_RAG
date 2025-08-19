#!/bin/bash

# MoE Simple Web UI Launcher
# シンプルなWebUI起動スクリプト

echo "==========================================="
echo "🏗️ MoE 土木・建設AI システム"
echo "Simple Web UI Launcher"
echo "==========================================="

cd /home/kjifu/AI_FT_7

# 仮想環境のアクティベート
source venv/bin/activate

# Flaskの確認とインストール
if ! python -c "import flask" 2>/dev/null; then
    echo "Flaskをインストール中..."
    pip install flask --quiet
fi

echo ""
echo "シンプルWebUIを起動中..."
echo "ブラウザで以下のURLにアクセスしてください:"
echo ""
echo "  🌐 http://localhost:5000"
echo ""
echo "終了するには Ctrl+C を押してください"
echo ""

# Flask アプリの起動
python app/moe_simple_ui.py
