#!/bin/bash

# Ollama インストールとセットアップスクリプト

echo "================================"
echo "Ollama Setup Script"
echo "================================"

# Ollamaがインストールされているか確認
if ! command -v ollama &> /dev/null; then
    echo "📥 Ollamaをインストール中..."
    
    # インストール方法を選択
    if [ -f /.dockerenv ]; then
        # Docker環境の場合
        echo "🐳 Docker環境でインストール中..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        # 通常のLinux環境
        echo "🐧 Linux環境でインストール中..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollamaのインストールが完了しました"
    else
        echo "❌ Ollamaのインストールに失敗しました"
        echo "手動でインストールしてください："
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
else
    echo "✅ Ollamaは既にインストールされています"
    ollama --version
fi

# Ollamaサービスを起動
echo ""
echo "🚀 Ollamaサービスを起動中..."

# 既存のプロセスを確認
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollamaサービスは既に起動しています"
else
    # バックグラウンドで起動
    export OLLAMA_HOST=0.0.0.0:11434
    nohup ollama serve > /tmp/ollama_setup.log 2>&1 &
    OLLAMA_PID=$!
    
    echo "⏳ サービスの起動を待機中..."
    sleep 5
    
    # 起動確認
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "✅ Ollamaサービスが起動しました (PID: $OLLAMA_PID)"
    else
        echo "⚠️ Ollamaサービスの起動に問題があります"
        echo "ログを確認してください: cat /tmp/ollama_setup.log"
    fi
fi

# 利用可能なモデルを確認
echo ""
echo "📋 現在インストールされているモデル:"
ollama list 2>/dev/null || echo "モデルがありません"

# モデルをダウンロード
echo ""
echo "📥 推奨モデルをダウンロード..."

# ダウンロードするモデルのリスト（小さいものから）
MODELS=(
    "tinyllama:latest:1.1b"
    "phi:latest:2.7b"
    "llama2:7b:7b"
    "mistral:7b:7b"
)

echo "以下のモデルから選択してダウンロードします："
echo "1) tinyllama:latest (1.1B, 最軽量)"
echo "2) phi:latest (2.7B, 軽量)"
echo "3) llama2:7b (7B, 標準)"
echo "4) mistral:7b (7B, 高性能)"
echo "5) すべてスキップ"

read -p "選択してください (1-5): " choice

case $choice in
    1)
        echo "📥 tinyllama:latest をダウンロード中..."
        ollama pull tinyllama:latest
        ;;
    2)
        echo "📥 phi:latest をダウンロード中..."
        ollama pull phi:latest
        ;;
    3)
        echo "📥 llama2:7b をダウンロード中..."
        ollama pull llama2:7b
        ;;
    4)
        echo "📥 mistral:7b をダウンロード中..."
        ollama pull mistral:7b
        ;;
    5)
        echo "スキップしました"
        ;;
    *)
        echo "📥 デフォルトで tinyllama:latest をダウンロード中..."
        ollama pull tinyllama:latest
        ;;
esac

# 最終確認
echo ""
echo "================================"
echo "セットアップ完了"
echo "================================"
echo ""
echo "📋 インストール済みモデル:"
ollama list 2>/dev/null

echo ""
echo "🔧 テストコマンド:"
echo "  ollama run tinyllama:latest"
echo ""
echo "📌 APIエンドポイント:"
echo "  http://localhost:11434"
echo ""
