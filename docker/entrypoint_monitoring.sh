#!/bin/bash

# Dockerコンテナのエントリーポイント（監視統合版）
# main_unified.pyを起動してGrafanaで監視可能にする

echo "🐳 AI-FT コンテナを起動します（監視統合版）..."

# 環境変数の確認
echo "📋 環境設定:"
echo "  - REDIS_HOST: ${REDIS_HOST:-redis}"
echo "  - REDIS_PORT: ${REDIS_PORT:-6379}"
echo "  - CACHE_DEFAULT_TTL: ${CACHE_DEFAULT_TTL:-3600}"
echo "  - 監視モード: Prometheus/Grafana統合"

# Redisへの接続をテスト（最大30秒待機）
echo "🔍 Redisへの接続を確認中..."
for i in {1..30}; do
    if python3 -c "
import redis
import sys
try:
    r = redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379})
    r.ping()
    print('✅ Redis接続成功!')
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "⚠️ Redis接続タイムアウト - キャッシュなしで起動します"
    else
        echo "  待機中... ($i/30)"
        sleep 1
    fi
done

# 必要なパッケージの確認とインストール
echo "📦 依存関係の確認..."
pip install --no-cache-dir pydantic==2.5.0 pydantic-settings==2.1.0 redis hiredis prometheus-client 2>/dev/null

# アプリケーションの起動
echo "🚀 メインアプリケーション（統合版）を起動します..."
echo "📊 Prometheusメトリクス有効: http://localhost:8050/metrics"

# main_unified.pyを起動（フル機能版）
exec python3 -m uvicorn app.main_unified:app \
    --host 0.0.0.0 \
    --port 8050 \
    --reload \
    --reload-dir /workspace/app \
    --log-level info