#!/bin/bash

# Docker Clean and Rebuild Script
# Dockerの問題を解決するスクリプト

echo "==========================================="
echo "Docker環境のクリーンアップと再構築"
echo "==========================================="

# Dockerのクリーンアップ
echo "1. Dockerキャッシュをクリア中..."
docker system prune -f

echo "2. ビルドキャッシュをクリア中..."
docker builder prune -f

echo "3. 未使用のイメージを削除中..."
docker image prune -a -f

echo "4. 既存のコンテナを停止..."
docker-compose down 2>/dev/null || true

echo ""
echo "==========================================="
echo "代替方法: Pythonで直接起動"
echo "==========================================="

cd /home/kjifu/AI_FT_7

# Python環境の確認
if command -v python3 &> /dev/null; then
    echo "✅ Python3が利用可能です"
    
    # 仮想環境の確認
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ 仮想環境を活性化しました"
    else
        echo "📦 仮想環境を作成中..."
        python3 -m venv venv
        source venv/bin/activate
    fi
    
    # Flaskのインストール
    echo "📥 Flaskをインストール中..."
    pip install flask --quiet
    
    echo ""
    echo "✅ 準備完了！"
    echo ""
    echo "以下のコマンドでWebUIを起動できます:"
    echo ""
    echo "  bash quick_start.sh"
    echo ""
    echo "または:"
    echo ""
    echo "  python app/moe_simple_ui.py"
    echo ""
else
    echo "❌ Python3が見つかりません"
    echo "以下のコマンドでインストールしてください:"
    echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv"
fi
