#!/bin/bash

# AI Fine-tuning Toolkit Web Interface 起動スクリプト

echo "🚀 AI Fine-tuning Toolkit Web Interface を起動中..."

# 必要なディレクトリを作成
mkdir -p data/uploaded
mkdir -p outputs
mkdir -p app/static
mkdir -p logs

# 依存関係の確認
echo "📦 依存関係を確認中..."
python -c "
import sys
try:
    import fastapi, uvicorn, torch
    from fastapi import File, UploadFile
    import psutil
    print('✅ 基本依存関係OK')
except ImportError as e:
    print(f'❌ 依存関係エラー: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  依存関係が不足しています。インストール中..."
    pip install fastapi uvicorn python-multipart psutil
    echo "✅ 依存関係インストール完了"
fi

# GPU確認
echo "🔍 GPU状況を確認中..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  GPU が検出されませんでした')
"

# 既存のプロセス停止
echo "🔄 既存のWebサーバープロセスを停止中..."
pkill -f uvicorn 2>/dev/null || true
sleep 2

# Webサーバー起動
echo "🌐 Webサーバーを起動中..."
echo "   アクセス URL: http://localhost:8050"
echo "   停止するには Ctrl+C を押してください"
echo ""

# 統合版を起動
echo "🔧 統合版Webインターフェースを起動中..."
python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload