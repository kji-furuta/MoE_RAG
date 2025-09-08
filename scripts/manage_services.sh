#!/bin/bash

# サービス管理スクリプト
# メインアプリケーションとGrafana監視を独立して管理

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo "📊 AI-FT サービス管理ツール"
    echo "================================"
    echo ""
    echo "使用方法: $0 [コマンド]"
    echo ""
    echo "コマンド:"
    echo "  start-app       - メインアプリケーションのみ起動"
    echo "  start-monitor   - Grafana/Prometheus監視のみ起動"
    echo "  start-all       - すべてのサービスを起動"
    echo "  stop-app        - メインアプリケーションを停止"
    echo "  stop-monitor    - 監視サービスを停止"
    echo "  stop-all        - すべてのサービスを停止"
    echo "  status          - すべてのサービスの状態確認"
    echo "  restart-app     - アプリケーションを再起動"
    echo "  logs-app        - アプリケーションのログ表示"
    echo "  logs-monitor    - 監視サービスのログ表示"
    echo ""
    echo "例:"
    echo "  $0 start-app    # アプリケーションのみ起動"
    echo "  $0 status       # 状態確認"
}

start_app() {
    echo -e "${BLUE}🚀 メインアプリケーションを起動します...${NC}"
    cd /workspace 2>/dev/null || cd /home/kjifu/MoE_RAG
    
    # 通常のdocker-compose.ymlを使用（main_unified.pyが起動）
    docker-compose -f docker/docker-compose.yml up -d ai-ft redis qdrant
    
    echo "⏳ 起動待機中..."
    sleep 5
    
    # アプリケーションの正常起動を確認
    if curl -s http://localhost:8050/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ アプリケーションが正常に起動しました${NC}"
        echo "   URL: http://localhost:8050"
    else
        echo -e "${YELLOW}⚠️  アプリケーションの起動確認中...${NC}"
        echo "   数秒後に http://localhost:8050 にアクセスしてください"
    fi
}

start_monitor() {
    echo -e "${BLUE}📊 監視サービスを起動します...${NC}"
    cd /workspace 2>/dev/null || cd /home/kjifu/MoE_RAG
    
    # 監視サービスのみ起動（アプリケーションは起動しない）
    if [ -f docker/docker-compose-monitoring.yml ]; then
        docker-compose -f docker/docker-compose-monitoring.yml up -d prometheus grafana redis-exporter node-exporter
    else
        echo -e "${YELLOW}⚠️ 監視設定ファイルが見つかりません${NC}"
        return 1
    fi
    
    echo "⏳ 起動待機中..."
    sleep 5
    
    echo -e "${GREEN}✅ 監視サービスが起動しました${NC}"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
    echo "   Prometheus: http://localhost:9090"
}

start_all() {
    echo -e "${BLUE}🔧 すべてのサービスを起動します...${NC}"
    
    # まずアプリケーションを起動
    start_app
    
    echo ""
    
    # 次に監視サービスを起動
    start_monitor
}

stop_app() {
    echo -e "${BLUE}🛑 メインアプリケーションを停止します...${NC}"
    cd /workspace 2>/dev/null || cd /home/kjifu/MoE_RAG
    docker-compose -f docker/docker-compose.yml down
    echo -e "${GREEN}✅ アプリケーションを停止しました${NC}"
}

stop_monitor() {
    echo -e "${BLUE}🛑 監視サービスを停止します...${NC}"
    cd /workspace 2>/dev/null || cd /home/kjifu/MoE_RAG
    if [ -f docker/docker-compose-monitoring.yml ]; then
        docker-compose -f docker/docker-compose-monitoring.yml stop prometheus grafana redis-exporter node-exporter
        docker-compose -f docker/docker-compose-monitoring.yml rm -f prometheus grafana redis-exporter node-exporter
    fi
    echo -e "${GREEN}✅ 監視サービスを停止しました${NC}"
}

stop_all() {
    echo -e "${BLUE}🛑 すべてのサービスを停止します...${NC}"
    stop_monitor
    stop_app
}

show_status() {
    echo -e "${BLUE}📊 サービス状態${NC}"
    echo "================================"
    
    echo ""
    echo "🐳 Docker コンテナ:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAME|ai-ft|grafana|prometheus|exporter|redis|qdrant)"
    
    echo ""
    echo "🌐 サービスチェック:"
    
    # アプリケーション
    if curl -s http://localhost:8050/ > /dev/null 2>&1; then
        echo -e "  アプリケーション: ${GREEN}✅ 稼働中${NC} (http://localhost:8050)"
    else
        echo -e "  アプリケーション: ${RED}❌ 停止${NC}"
    fi
    
    # Grafana
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "  Grafana: ${GREEN}✅ 稼働中${NC} (http://localhost:3000)"
    else
        echo -e "  Grafana: ${YELLOW}⚠️ 停止または未起動${NC}"
    fi
    
    # Prometheus
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e "  Prometheus: ${GREEN}✅ 稼働中${NC} (http://localhost:9090)"
    else
        echo -e "  Prometheus: ${YELLOW}⚠️ 停止または未起動${NC}"
    fi
    
    # Redis
    if docker exec ai-ft-redis redis-cli ping > /dev/null 2>&1; then
        echo -e "  Redis: ${GREEN}✅ 稼働中${NC}"
    else
        echo -e "  Redis: ${YELLOW}⚠️ 停止または未起動${NC}"
    fi
    
    # Qdrant
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo -e "  Qdrant: ${GREEN}✅ 稼働中${NC}"
    else
        echo -e "  Qdrant: ${YELLOW}⚠️ 停止または未起動${NC}"
    fi
    
    # Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "  Ollama: ${GREEN}✅ 稼働中${NC} (http://localhost:11434)"
    else
        echo -e "  Ollama: ${YELLOW}⚠️ 停止または未起動${NC}"
    fi
}

restart_app() {
    echo -e "${BLUE}🔄 アプリケーションを再起動します...${NC}"
    stop_app
    sleep 2
    start_app
}

logs_app() {
    echo -e "${BLUE}📝 アプリケーションログ（最新50行）${NC}"
    docker logs ai-ft-container --tail 50
}

logs_monitor() {
    echo -e "${BLUE}📝 監視サービスログ${NC}"
    echo "Grafana:"
    docker logs ai-ft-grafana --tail 20 2>/dev/null || echo "Grafanaコンテナが見つかりません"
    echo ""
    echo "Prometheus:"
    docker logs ai-ft-prometheus --tail 20 2>/dev/null || echo "Prometheusコンテナが見つかりません"
}

# メイン処理
case "$1" in
    start-app)
        start_app
        ;;
    start-monitor)
        start_monitor
        ;;
    start-all)
        start_all
        ;;
    stop-app)
        stop_app
        ;;
    stop-monitor)
        stop_monitor
        ;;
    stop-all)
        stop_all
        ;;
    status)
        show_status
        ;;
    restart-app)
        restart_app
        ;;
    logs-app)
        logs_app
        ;;
    logs-monitor)
        logs_monitor
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ 不明なコマンド: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac