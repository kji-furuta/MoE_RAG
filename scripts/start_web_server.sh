#!/bin/bash
# Webサーバーを起動するスクリプト（バックグラウンド実行用）

echo "🚀 Webサーバーを起動中..."

# 既存のプロセスを確認
if lsof -i:8050 > /dev/null 2>&1; then
    echo "⚠️ ポート8050は既に使用されています"
    echo "既存のプロセスを終了しますか？ (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        kill $(lsof -t -i:8050) 2>/dev/null || true
        sleep 2
    else
        echo "起動をキャンセルしました"
        exit 1
    fi
fi

# Webサーバーをバックグラウンドで起動
cd /workspace
nohup python3 -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload > /workspace/logs/web_server.log 2>&1 &

echo "✅ Webサーバーを起動しました"
echo "📊 ログ: /workspace/logs/web_server.log"
echo "🌐 URL: http://localhost:8050/"
echo ""
echo "プロセスを確認: ps aux | grep uvicorn"
echo "ログを確認: tail -f /workspace/logs/web_server.log"
echo "停止: kill \$(lsof -t -i:8050)"
