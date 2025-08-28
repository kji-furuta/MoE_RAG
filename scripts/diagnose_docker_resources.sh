#!/bin/bash

# Dockerリソース診断スクリプト

echo "======================================"
echo "Docker Resource Diagnostics"
echo "======================================"

# 1. Dockerのメモリ制限を確認
echo -e "\n1. Docker Memory Limits:"
docker stats --no-stream ai-ft-container 2>/dev/null || echo "Container not running"

# 2. コンテナ内のメモリ状況
echo -e "\n2. Container Memory Info:"
docker exec ai-ft-container cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable" 2>/dev/null || echo "Cannot check"

# 3. GPU情報
echo -e "\n3. GPU Status in Container:"
docker exec ai-ft-container nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# 4. Dockerデーモンの設定
echo -e "\n4. Docker Daemon Config:"
if [ -f /etc/docker/daemon.json ]; then
    cat /etc/docker/daemon.json
else
    echo "No daemon.json found"
fi

# 5. WSL2のメモリ設定（Windowsの場合）
echo -e "\n5. WSL2 Memory Config:"
if [ -f /mnt/c/Users/*/wslconfig ]; then
    cat /mnt/c/Users/*/wslconfig 2>/dev/null || echo "No .wslconfig found"
else
    echo "Not WSL2 or no config"
fi

# 6. Docker Composeのリソース制限確認
echo -e "\n6. Docker Compose Resource Limits:"
cd /home/kjifu/MoE_RAG/docker
docker-compose config | grep -A10 "deploy:" 2>/dev/null || echo "No resource limits set"

# 7. 現在のシステムリソース
echo -e "\n7. Host System Resources:"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep "^Mem:" | awk '{print $2}') total"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")"

echo -e "\n======================================"
