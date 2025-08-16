#!/bin/bash

# Web UIから監視システムを制御するためのスクリプト
# このスクリプトはホスト側で実行される

ACTION=$1
PROJECT_DIR="/home/kjifu/AI_FT_7"
DOCKER_DIR="$PROJECT_DIR/docker"

cd $PROJECT_DIR

case $ACTION in
    "start")
        echo "🚀 監視システムを起動中..."
        
        # Prometheus と Grafana のみ起動（メインコンテナは既に起動済み）
        docker-compose -f $DOCKER_DIR/docker-compose-monitoring.yml up -d prometheus grafana redis 2>&1
        
        # 起動確認
        sleep 3
        
        # サービス状態確認
        if docker ps | grep -q "ai-ft-prometheus"; then
            echo "✅ Prometheus: 起動成功"
        else
            echo "❌ Prometheus: 起動失敗"
            exit 1
        fi
        
        if docker ps | grep -q "ai-ft-grafana"; then
            echo "✅ Grafana: 起動成功"
        else
            echo "❌ Grafana: 起動失敗"
            exit 1
        fi
        
        echo "✅ 監視システムの起動が完了しました"
        echo "Grafana: http://localhost:3000 (admin/admin)"
        echo "Prometheus: http://localhost:9090"
        ;;
        
    "stop")
        echo "🛑 監視システムを停止中..."
        docker stop ai-ft-prometheus ai-ft-grafana ai-ft-redis 2>/dev/null
        echo "✅ 監視システムを停止しました"
        ;;
        
    "status")
        echo "📊 監視システムの状態確認"
        
        # Prometheus
        if docker ps | grep -q "ai-ft-prometheus"; then
            echo "Prometheus: Running"
            if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
                echo "Prometheus: Healthy"
            else
                echo "Prometheus: Unhealthy"
            fi
        else
            echo "Prometheus: Stopped"
        fi
        
        # Grafana
        if docker ps | grep -q "ai-ft-grafana"; then
            echo "Grafana: Running"
            if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
                echo "Grafana: Healthy"
            else
                echo "Grafana: Starting"
            fi
        else
            echo "Grafana: Stopped"
        fi
        
        # Redis
        if docker ps | grep -q "ai-ft-redis"; then
            echo "Redis: Running"
        else
            echo "Redis: Stopped"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac