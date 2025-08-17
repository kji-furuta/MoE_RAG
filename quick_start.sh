#!/bin/bash

# MoE WebUI Quick Start Script
# Docker不要の簡単起動スクリプト

set -e

echo "==========================================="
echo "🏗️ MoE 土木・建設AI システム"
echo "Quick Start (No Docker Required)"
echo "==========================================="

# プロジェクトディレクトリ
PROJECT_ROOT="/home/kjifu/AI_FT_7"
cd $PROJECT_ROOT

# Pythonの確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3が見つかりません"
    exit 1
fi

echo "✅ Python3が見つかりました: $(python3 --version)"

# 仮想環境の作成または活性化
if [ ! -d "venv" ]; then
    echo "📦 仮想環境を作成中..."
    python3 -m venv venv
fi

echo "🔄 仮想環境を活性化中..."
source venv/bin/activate

# 必要最小限のパッケージをインストール
echo "📥 必要なパッケージをインストール中..."

# Flaskのインストール（最も軽量）
pip install flask --quiet 2>/dev/null || {
    echo "Flask インストール中..."
    pip install flask
}

echo ""
echo "✅ セットアップ完了！"
echo ""
echo "==========================================="
echo "🌐 WebUIを起動します"
echo "==========================================="
echo ""
echo "ブラウザで以下のURLにアクセスしてください:"
echo ""
echo "  📍 http://localhost:5000"
echo ""
echo "終了する場合は Ctrl+C を押してください"
echo "==========================================="
echo ""

# WebUIの起動
python app/moe_simple_ui.py
