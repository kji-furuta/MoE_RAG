#!/bin/bash

echo "================================"
echo "ベースモデル設定ファイルの準備"
echo "================================"

MODEL_DIR="/workspace/models/deepseek-base"
mkdir -p $MODEL_DIR

echo "元のモデルの設定ファイルをダウンロード中..."

# cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japaneseの設定ファイルをダウンロード
# （LoRAアダプターのベースモデル）
wget -P $MODEL_DIR https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese/raw/main/config.json
wget -P $MODEL_DIR https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese/raw/main/tokenizer_config.json
wget -P $MODEL_DIR https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese/raw/main/tokenizer.json

echo ""
echo "ダウンロードしたファイル:"
ls -la $MODEL_DIR/

echo ""
echo "✅ 設定ファイルの準備完了"
echo "このディレクトリをベースモデルとして使用できます: $MODEL_DIR"