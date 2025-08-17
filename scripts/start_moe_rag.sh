#!/bin/bash

echo "=========================================="
echo "🚀 MoE-RAG統合システム起動"
echo "=========================================="

# カラー定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Pythonパスの設定
export PYTHONPATH=/workspace:$PYTHONPATH

# プロセスの確認と停止
echo -e "${BLUE}既存のMoE-RAGプロセスを確認中...${NC}"
if pgrep -f "moe_rag_api" > /dev/null; then
    echo "既存のMoE-RAGプロセスを停止中..."
    pkill -f "moe_rag_api"
    sleep 2
fi

# ログディレクトリの作成
mkdir -p /workspace/logs

# MoE-RAG APIの起動
echo -e "${BLUE}MoE-RAG APIを起動中...${NC}"
nohup python /workspace/app/moe_rag_api.py > /workspace/logs/moe_rag_api.log 2>&1 &

# 起動確認
sleep 3
if pgrep -f "moe_rag_api" > /dev/null; then
    echo -e "${GREEN}✅ MoE-RAG APIが正常に起動しました${NC}"
    echo ""
    echo "=========================================="
    echo "📌 アクセス情報"
    echo "=========================================="
    echo "Web UI: http://localhost:8050/static/moe_rag_ui.html"
    echo "API Docs: http://localhost:8050/docs"
    echo "Health Check: http://localhost:8050/api/moe-rag/health"
    echo ""
    echo "ログ確認: tail -f /workspace/logs/moe_rag_api.log"
    echo "=========================================="
else
    echo "❌ MoE-RAG APIの起動に失敗しました"
    echo "ログを確認してください: /workspace/logs/moe_rag_api.log"
    tail -n 20 /workspace/logs/moe_rag_api.log
    exit 1
fi