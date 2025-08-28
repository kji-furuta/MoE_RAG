#!/bin/bash

# Docker用エントリーポイントスクリプト
# メモリ管理の初期化問題を回避

echo "🐳 Docker Container Starting..."
echo "📦 Environment Setup..."

# Docker環境フラグを設定
export DOCKER_CONTAINER=true
export MEMORY_MANAGER_INITIALIZED=1

# メモリ最適化設定（Docker用の控えめな設定）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# CUDA_LAUNCH_BLOCKINGは設定しない（Docker内で問題を起こす可能性）

# Pythonパスの設定
export PYTHONPATH=/workspace:$PYTHONPATH

# 作業ディレクトリ
cd /workspace

# 必要なディレクトリを作成
mkdir -p outputs data logs temp_uploads config
mkdir -p data/continual_learning data/uploaded

# パーミッションの設定
chmod -R 755 /workspace/scripts 2>/dev/null || true
chmod -R 777 /workspace/outputs 2>/dev/null || true
chmod -R 777 /workspace/data 2>/dev/null || true
chmod -R 777 /workspace/logs 2>/dev/null || true

echo "✅ Environment ready"

# コマンド引数がある場合は実行
if [ $# -gt 0 ]; then
    echo "🚀 Executing: $@"
    exec "$@"
else
    echo "🌐 Starting Web Interface..."
    # Docker用の起動スクリプトを使用
    if [ -f /workspace/scripts/start_web_interface_docker.sh ]; then
        exec bash /workspace/scripts/start_web_interface_docker.sh
    else
        # フォールバック：直接起動
        exec python3 -m uvicorn app.main_unified_docker:app \
            --host 0.0.0.0 \
            --port 8050 \
            --workers 1 \
            --loop asyncio \
            --log-level info
    fi
fi
