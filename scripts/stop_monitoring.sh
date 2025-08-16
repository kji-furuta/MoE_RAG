#!/bin/bash

# 監視システムを停止するスクリプト

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  監視システム停止スクリプト${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# プロジェクトディレクトリに移動
cd /home/kjifu/AI_FT_7

echo -e "${YELLOW}🛑 監視システムを停止中...${NC}"

# 個別にコンテナを停止（削除もサポート）
echo "  - Grafana を停止..."
docker stop ai-ft-grafana 2>/dev/null && echo -e "    ${GREEN}✓ Grafana 停止完了${NC}" || echo -e "    ${YELLOW}! Grafana は起動していません${NC}"

echo "  - Prometheus を停止..."
docker stop ai-ft-prometheus 2>/dev/null && echo -e "    ${GREEN}✓ Prometheus 停止完了${NC}" || echo -e "    ${YELLOW}! Prometheus は起動していません${NC}"

echo "  - Redis Exporter を停止..."
docker stop ai-ft-redis-exporter 2>/dev/null && echo -e "    ${GREEN}✓ Redis Exporter 停止完了${NC}" || echo -e "    ${YELLOW}! Redis Exporter は起動していません${NC}"

echo "  - Node Exporter を停止..."
docker stop ai-ft-node-exporter 2>/dev/null && echo -e "    ${GREEN}✓ Node Exporter 停止完了${NC}" || echo -e "    ${YELLOW}! Node Exporter は起動していません${NC}"

echo ""
echo -e "${GREEN}✅ 監視システムの停止が完了しました${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "💡 再起動する場合:"
echo "  ./scripts/start_monitoring_with_main.sh"
echo "  または"
echo "  docker-compose -f docker/docker-compose-monitoring.yml up -d"
echo ""