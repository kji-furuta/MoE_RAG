#!/bin/bash

# Docker Desktop リソース設定確認スクリプト

echo "======================================"
echo "Docker Desktop Resource Check"
echo "======================================"

# 1. Docker情報
echo -e "\n1. Docker Version & Info:"
docker version --format '{{.Server.Version}}'
docker info --format '{{json .}}' | python3 -m json.tool | grep -E '"MemTotal"|"NCPU"|"Driver"' || docker info | grep -E "Total Memory|CPUs|Storage"

# 2. 現在のコンテナリソース使用状況
echo -e "\n2. Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# 3. Docker Desktopの設定値を取得（Windows）
echo -e "\n3. Docker Desktop Settings (if available):"
if [ -f ~/.docker/config.json ]; then
    cat ~/.docker/config.json | python3 -m json.tool 2>/dev/null | head -20
fi

# 4. WSL2メモリ情報
echo -e "\n4. WSL2 Memory Information:"
if grep -q microsoft /proc/version; then
    echo "Running in WSL2"
    echo "Total Memory: $(free -h | grep "^Mem:" | awk '{print $2}')"
    echo "Available Memory: $(free -h | grep "^Mem:" | awk '{print $7}')"
    echo ""
    echo "WSL2 Config Location: C:\\Users\\[YourUsername]\\.wslconfig"
    echo ""
    # WSL2の制限を確認
    cat /proc/meminfo | grep -E "MemTotal|MemAvailable"
else
    echo "Not running in WSL2"
fi

# 5. Docker実行時の制限確認
echo -e "\n5. Runtime Constraints:"
docker run --rm alpine sh -c "
echo 'Container Limits:'
echo '  CPU Quota:' \$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 'unlimited')
echo '  CPU Period:' \$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 'N/A')
echo '  Memory Limit:' \$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | numfmt --to=iec || echo 'unlimited')
echo '  Memory Usage:' \$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null | numfmt --to=iec || echo 'N/A')
"

# 6. GPU設定確認
echo -e "\n6. GPU Configuration:"
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>/dev/null || echo "GPU not accessible in Docker"

# 7. 推奨設定
echo -e "\n======================================"
echo "Recommended Settings for Your System:"
echo "======================================"
echo ""
echo "1. Docker Desktop Settings (GUI):"
echo "   - CPUs: 16"
echo "   - Memory: 80 GB"
echo "   - Swap: 8-16 GB"
echo "   - Disk image size: 200+ GB"
echo ""
echo "2. Create/Edit: C:\\Users\\[YourUsername]\\.wslconfig"
echo "   Copy content from: $(pwd)/docker/wslconfig.example"
echo ""
echo "3. After changing settings:"
echo "   a. Close Docker Desktop"
echo "   b. Run in PowerShell: wsl --shutdown"
echo "   c. Restart Docker Desktop"
echo ""
echo "======================================"
