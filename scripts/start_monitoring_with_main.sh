#!/bin/bash

# 監視システムとメインアプリケーション統合起動スクリプト

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  AI-FT 監視システム統合起動スクリプト${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd /home/kjifuruta/AI_FT/AI_FT_3

# 既存のコンテナを停止
echo -e "${YELLOW}🛑 既存のサービスを停止します...${NC}"
docker-compose -f docker/docker-compose.yml down 2>/dev/null
docker-compose -f docker/docker-compose-monitoring.yml down 2>/dev/null

echo ""
echo -e "${BLUE}🚀 監視システム統合環境を起動します...${NC}"
echo "  - メインアプリケーション (main_unified.py)"
echo "  - Prometheus"
echo "  - Grafana"
echo "  - Redis"
echo "  - Qdrant"
echo ""

# 監視統合環境を起動（main_unified.pyが起動される）
docker-compose -f docker/docker-compose-monitoring.yml up -d

echo -e "${YELLOW}⏳ サービス起動待機中...${NC}"
sleep 10

# サービスの状態確認
echo ""
echo -e "${BLUE}📊 サービス状態確認${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# メインアプリケーション確認
if curl -s http://localhost:8050/ > /dev/null 2>&1; then
    echo -e "  メインアプリケーション: ${GREEN}✅ 稼働中${NC}"
    echo "    → http://localhost:8050"
    
    # メトリクスエンドポイント確認
    if curl -s http://localhost:8050/metrics | grep -q "ai_ft_"; then
        echo -e "  メトリクスエンドポイント: ${GREEN}✅ 正常${NC}"
        echo "    → ai_ft_* メトリクスが出力されています"
    else
        echo -e "  メトリクスエンドポイント: ${YELLOW}⚠️ 確認中${NC}"
    fi
else
    echo -e "  メインアプリケーション: ${RED}❌ 起動失敗${NC}"
fi

# Prometheus確認
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "  Prometheus: ${GREEN}✅ 稼働中${NC}"
    echo "    → http://localhost:9090"
else
    echo -e "  Prometheus: ${RED}❌ 起動失敗${NC}"
fi

# Grafana確認
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "  Grafana: ${GREEN}✅ 稼働中${NC}"
    echo "    → http://localhost:3000 (admin/admin)"
else
    echo -e "  Grafana: ${YELLOW}⚠️ 起動中...${NC}"
fi

# Redis確認
if docker exec ai-ft-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "  Redis: ${GREEN}✅ 稼働中${NC}"
else
    echo -e "  Redis: ${RED}❌ 起動失敗${NC}"
fi

# Qdrant確認
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "  Qdrant: ${GREEN}✅ 稼働中${NC}"
else
    echo -e "  Qdrant: ${RED}❌ 起動失敗${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ 監視統合環境の起動が完了しました${NC}"
echo ""
echo "📊 アクセスURL:"
echo "  - メインアプリケーション: http://localhost:8050"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "💡 ヒント:"
echo "  - Grafanaダッシュボード設定: ./scripts/setup_grafana_dashboard.sh"
echo "  - ログ確認: docker logs -f ai-ft-container"
echo "  - 停止: docker-compose -f docker/docker-compose-monitoring.yml down"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"