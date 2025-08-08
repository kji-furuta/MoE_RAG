#!/bin/bash
# Docker環境内でPhase 2テストを実行するスクリプト

echo "======================================================================"
echo "🐳 Running Phase 2 Tests in Docker Container"
echo "======================================================================"
echo ""

# Dockerコンテナが実行中か確認
if ! docker ps | grep -q ai-ft-container; then
    echo "❌ Docker container 'ai-ft-container' is not running"
    echo ""
    echo "Starting container..."
    cd docker
    docker-compose up -d
    sleep 5
fi

echo "📦 Installing psutil in container (if needed)..."
docker exec ai-ft-container pip install psutil 2>/dev/null

# キャッシュディレクトリの作成（書き込み可能な場所）
echo "🔧 Setting up cache directories..."
docker exec ai-ft-container mkdir -p /tmp/ai_ft_cache
docker exec ai-ft-container chmod 777 /tmp/ai_ft_cache
docker exec ai-ft-container mkdir -p /workspace/.cache
docker exec ai-ft-container chmod 777 /workspace/.cache

echo ""
echo "======================================================================"
echo "1. Testing Phase 2 Integration in Container"
echo "======================================================================"
docker exec ai-ft-container python3 /workspace/scripts/test_phase2_integration.py

echo ""
echo "======================================================================"
echo "2. Testing System Optimization in Container"
echo "======================================================================"
docker exec ai-ft-container python3 /workspace/scripts/optimize_rag_system.py

echo ""
echo "======================================================================"
echo "✅ Tests completed in Docker container"
echo "======================================================================"
