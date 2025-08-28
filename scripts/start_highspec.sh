#!/bin/bash

# 高スペック環境用起動スクリプト
# タイムアウトを増やし、フルモデルを使用

echo "🚀 Starting AI_FT_3 (High-Spec Mode)..."
echo "💪 System Resources:"
echo "  - GPU: $(nvidia-smi --query-gpu=count --format=csv,noheader) devices"
echo "  - RAM: $(free -h | grep "^Mem:" | awk '{print $2}')"
echo "  - CPU: $(nproc) cores"

cd /workspace

# 高スペック環境用の設定
export PYTHONPATH=/workspace:$PYTHONPATH

# メモリ最適化（大容量向け）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# タイムアウト設定を大幅に増加
export TRANSFORMERS_TIMEOUT=600  # 10分
export HF_HUB_DOWNLOAD_TIMEOUT=600
export UVICORN_TIMEOUT_KEEP_ALIVE=300  # 5分
export REQUEST_TIMEOUT=300

# バッファとログ設定
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true  # 高スペックなら並列化OK

# RAGシステムはフル機能を使用
unset RAG_DISABLE_AUTO_INIT
unset RAG_USE_CPU
export RAG_DEVICE=cuda
export RAG_BATCH_SIZE=32

# 必要なディレクトリ
mkdir -p outputs data logs temp_uploads config

# Ollamaチェック
if command -v ollama &> /dev/null; then
    echo "🤖 Starting Ollama service..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi

echo ""
echo "🌐 Starting Full-Feature Web Server..."
echo "📊 Available endpoints:"
echo "  - Dashboard: http://localhost:8050/"
echo "  - API Docs: http://localhost:8050/docs"
echo "  - Fine-tuning: http://localhost:8050/static/moe_training.html"
echo "  - RAG System: http://localhost:8050/static/moe_rag_ui.html"
echo ""

# 既存プロセスをクリア
if lsof -i:8050 > /dev/null 2>&1; then
    echo "⚠️ Clearing port 8050..."
    kill $(lsof -t -i:8050) 2>/dev/null || true
    sleep 2
fi

# uvicornを高スペック設定で起動
exec python3 -m uvicorn app.main_unified:app \
    --host 0.0.0.0 \
    --port 8050 \
    --workers 1 \
    --loop uvloop \
    --timeout-keep-alive 300 \
    --timeout-notify 300 \
    --limit-max-requests 0 \
    --limit-concurrency 100 \
    --backlog 2048 \
    --log-level info
