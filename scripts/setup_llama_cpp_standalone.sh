#!/bin/bash

# llama.cppを独立してセットアップ
# Webサーバーに影響を与えない

echo "================================"
echo "llama.cpp スタンドアロンセットアップ"
echo "================================"

# 既存のllama.cppをチェック
if [ -f "/workspace/llama.cpp/build/bin/llama-quantize" ]; then
    echo "✅ llama.cppは既にセットアップ済みです"
    exit 0
fi

# 既存のディレクトリを削除
if [ -d "/workspace/llama.cpp" ]; then
    echo "既存のllama.cppディレクトリを削除中..."
    rm -rf /workspace/llama.cpp
fi

echo "llama.cppをクローン中..."
cd /workspace
git clone --depth 1 https://github.com/ggerganov/llama.cpp

if [ $? -ne 0 ]; then
    echo "❌ クローンに失敗しました"
    exit 1
fi

cd llama.cpp

echo "ビルド準備中..."
# CPUのみのビルド（CUDAを使わない）
cmake -B build \
    -DLLAMA_CURL=OFF \
    -DGGML_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake設定に失敗しました"
    exit 1
fi

echo "ビルド中（数分かかります）..."
cmake --build build --config Release -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ ビルドに失敗しました"
    exit 1
fi

# ビルド成功を確認
if [ -f "build/bin/llama-quantize" ]; then
    echo "✅ llama.cppセットアップ完了"
    echo "   llama-quantize: /workspace/llama.cpp/build/bin/llama-quantize"
    
    # 必要なPythonパッケージもインストール
    echo "依存関係をインストール中..."
    pip install --no-cache-dir gguf sentencepiece protobuf torch transformers
    
    echo "✅ すべてのセットアップが完了しました"
else
    echo "❌ llama-quantizeバイナリが見つかりません"
    exit 1
fi