#!/bin/bash

echo "================================"
echo "不足パッケージのインストール"
echo "================================"

echo "protobufをインストール中..."
pip install --no-cache-dir protobuf

if [ $? -eq 0 ]; then
    echo "✅ protobufインストール成功"
else
    echo "❌ protobufインストール失敗"
    exit 1
fi

echo ""
echo "インストール済みパッケージの確認:"
pip list | grep -E "protobuf|gguf|sentencepiece"

echo ""
echo "✅ 完了"