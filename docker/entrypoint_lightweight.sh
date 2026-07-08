#!/bin/bash

# Docker用エントリーポイントスクリプト（軽量版）
# メモリ問題を回避する最小限の初期化

echo "🐳 Docker Container Starting (Lightweight Mode)..."
echo "📦 Minimal Environment Setup..."

# Docker環境フラグ
export DOCKER_CONTAINER=true
export MEMORY_MANAGER_INITIALIZED=1

# RAGシステムの自動初期化を無効化
export RAG_DISABLE_AUTO_INIT=true
export RAG_LAZY_LOAD=true
export RAG_USE_CPU=true

# 軽量な埋め込みモデルを使用
export RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export RAG_EMBEDDING_DIMENSION=384

# メモリ設定（最小限）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

# Pythonパス
export PYTHONPATH=/workspace:$PYTHONPATH

cd /workspace

# 必要最小限のディレクトリ作成
mkdir -p outputs data logs temp_uploads config 2>/dev/null

echo "✅ Minimal environment ready"

# コマンド引数処理
if [ $# -gt 0 ]; then
    echo "🚀 Executing: $@"
    exec "$@"
else
    echo "🌐 Starting Lightweight Web Interface..."
    
    # 既存プロセスの確認と終了
    if lsof -i:8050 > /dev/null 2>&1; then
        echo "⚠️ Killing existing process on port 8050..."
        kill $(lsof -t -i:8050) 2>/dev/null || true
        sleep 2
    fi
    
    # 軽量版を起動
    exec python3 -m uvicorn app.main_unified_lightweight:app \
        --host 0.0.0.0 \
        --port 8050 \
        --workers 1 \
        --loop asyncio \
        --timeout-keep-alive 30 \
        --limit-max-requests 1000 \
        --log-level warning
fi
