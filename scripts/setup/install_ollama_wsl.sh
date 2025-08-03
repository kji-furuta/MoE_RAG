#!/bin/bash
# WSL環境用Ollamaインストールスクリプト

echo "WSL環境でのOllamaインストールを開始します..."

# システムの更新
echo "システムパッケージを更新中..."
sudo apt update && sudo apt upgrade -y

# 必要な依存関係をインストール
echo "必要な依存関係をインストール中..."
sudo apt install -y curl wget git build-essential

# Ollamaのインストール
echo "Ollamaをインストール中..."
curl -fsSL https://ollama.ai/install.sh | sh

# 環境変数の設定
echo "環境変数を設定中..."
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Ollamaサービスの起動
echo "Ollamaサービスを起動中..."
ollama serve &

# 起動確認
sleep 5
if ollama --version; then
    echo "✅ Ollamaのインストールが完了しました"
    echo "Ollamaバージョン: $(ollama --version)"
else
    echo "❌ Ollamaのインストールに失敗しました"
    exit 1
fi

# 基本的なモデルのダウンロード（オプション）
echo "基本的なモデルをダウンロードしますか？ (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "基本的なモデルをダウンロード中..."
    ollama pull llama2:7b
fi

echo "🎉 WSL環境でのOllamaセットアップが完了しました！"
echo "使用方法: ollama run llama2:7b 'Hello, world!'" 