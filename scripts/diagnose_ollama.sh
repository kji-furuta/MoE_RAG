#!/bin/bash

# Ollama問題診断スクリプト

echo "================================"
echo "Ollama Diagnostics"
echo "================================"

# 1. Ollamaのインストール状況
echo -e "\n1. Ollama Installation:"
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    echo "   Version: $(ollama --version 2>&1)"
    echo "   Path: $(which ollama)"
else
    echo "❌ Ollama is NOT installed"
fi

# 2. Ollamaプロセスの確認
echo -e "\n2. Ollama Process:"
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service is running"
    ps aux | grep "[o]llama serve"
else
    echo "❌ Ollama service is NOT running"
fi

# 3. ポートの確認
echo -e "\n3. Port Status (11434):"
if lsof -i:11434 > /dev/null 2>&1; then
    echo "✅ Port 11434 is in use"
    lsof -i:11434
else
    echo "⚠️ Port 11434 is not in use"
fi

# 4. API接続テスト
echo -e "\n4. API Connection Test:"
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "✅ API is accessible"
    echo "   Version: $(curl -s http://localhost:11434/api/version | python3 -m json.tool 2>/dev/null || echo 'N/A')"
else
    echo "❌ API is NOT accessible"
fi

# 5. インストール済みモデル
echo -e "\n5. Installed Models:"
if command -v ollama &> /dev/null; then
    ollama list 2>/dev/null || echo "⚠️ Cannot list models (service may not be running)"
else
    echo "⚠️ Ollama not installed"
fi

# 6. ネットワーク接続確認
echo -e "\n6. Network Connectivity:"
if ping -c 1 ollama.com > /dev/null 2>&1; then
    echo "✅ Can reach ollama.com"
else
    echo "❌ Cannot reach ollama.com (network issue?)"
fi

# 7. ディスク容量
echo -e "\n7. Disk Space:"
df -h /var/lib/ollama 2>/dev/null || df -h / | grep -E "Filesystem|/"

# 8. Docker環境確認
echo -e "\n8. Environment:"
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
else
    echo "🖥️ Running on host system"
fi

# 9. 権限確認
echo -e "\n9. Permissions:"
if [ -d /usr/share/ollama ]; then
    ls -ld /usr/share/ollama
fi
if [ -d ~/.ollama ]; then
    ls -ld ~/.ollama
fi

# 10. 推奨アクション
echo -e "\n================================"
echo "Recommended Actions:"
echo "================================"

if ! command -v ollama &> /dev/null; then
    echo "1. Install Ollama:"
    echo "   bash /workspace/scripts/setup_ollama.sh"
elif ! pgrep -x "ollama" > /dev/null; then
    echo "1. Start Ollama service:"
    echo "   ollama serve &"
elif ! ollama list 2>/dev/null | grep -q "tinyllama"; then
    echo "1. Download a model:"
    echo "   ollama pull tinyllama:latest"
else
    echo "✅ Ollama appears to be working correctly"
fi

echo ""
