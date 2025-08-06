#!/bin/bash
# OllamaでRAG用のモデルをセットアップするスクリプト

echo "OllamaでRAG用のモデルをセットアップします..."

# Ollamaサービスが起動しているか確認
if ! docker exec ai-ft-container pgrep ollama > /dev/null; then
    echo "Ollamaサービスを起動しています..."
    docker exec ai-ft-container bash -c "ollama serve &" 
    sleep 5
fi

# 軽量で高性能なモデルをpull
echo "Llama 3.1 8Bモデルをダウンロード中..."
docker exec ai-ft-container ollama pull llama3.1:8b

# モデルがダウンロードされたか確認
echo "利用可能なモデル一覧:"
docker exec ai-ft-container ollama list

echo "✅ Ollamaモデルのセットアップが完了しました"
echo "RAGシステムはOllamaフォールバックでこのモデルを使用できます"