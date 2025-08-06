#!/bin/bash
# check_container_status.sh
# コンテナの状態を確認するシンプルなスクリプト

echo "=== Docker コンテナ状態確認 ==="
echo ""

echo "1. 実行中のすべてのコンテナ:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "2. ai-ft-container の状態:"
if docker ps -q -f name=ai-ft-container | grep -q .; then
    echo "✓ ai-ft-container は実行中です"
    docker ps -f name=ai-ft-container --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "✗ ai-ft-container は実行されていません"
    echo ""
    echo "停止中のコンテナを確認:"
    docker ps -a -f name=ai-ft-container --format "table {{.Names}}\t{{.Status}}"
fi

echo ""
echo "3. Docker Compose サービスの状態:"
cd docker 2>/dev/null && docker-compose ps || echo "docker-compose.yml が見つかりません"
