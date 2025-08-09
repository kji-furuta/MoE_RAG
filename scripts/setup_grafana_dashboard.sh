#!/bin/bash

# Grafanaダッシュボード設定スクリプト

echo "📊 Grafanaダッシュボードを設定します..."

# ダッシュボードJSON作成
cat > /tmp/ai-ft-dashboard.json << 'EOF'
{
  "dashboard": {
    "annotations": {
      "list": []
    },
    "editable": true,
    "gnetId": null,
    "graphTooltip": 0,
    "hideControls": false,
    "id": null,
    "links": [],
    "panels": [
      {
        "datasource": "Prometheus",
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "mappings": [],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 80}
              ]
            },
            "unit": "percent"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
        "id": 1,
        "options": {
          "orientation": "auto",
          "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": false
          },
          "showThresholdLabels": false,
          "showThresholdMarkers": true
        },
        "targets": [
          {
            "expr": "ai_ft_cpu_usage_percent",
            "refId": "A"
          }
        ],
        "title": "CPU使用率",
        "type": "gauge"
      },
      {
        "datasource": "Prometheus",
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "mappings": [],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 60},
                {"color": "red", "value": 80}
              ]
            },
            "unit": "percent"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
        "id": 2,
        "options": {
          "orientation": "auto",
          "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": false
          },
          "showThresholdLabels": false,
          "showThresholdMarkers": true
        },
        "targets": [
          {
            "expr": "ai_ft_memory_usage_percent",
            "refId": "A"
          }
        ],
        "title": "メモリ使用率",
        "type": "gauge"
      },
      {
        "datasource": "Prometheus",
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "palette-classic"
            },
            "custom": {
              "axisLabel": "",
              "axisPlacement": "auto",
              "barAlignment": 0,
              "drawStyle": "line",
              "fillOpacity": 10,
              "gradientMode": "none",
              "hideFrom": {
                "legend": false,
                "tooltip": false,
                "viz": false
              },
              "lineInterpolation": "linear",
              "lineWidth": 1,
              "pointSize": 5,
              "scaleDistribution": {
                "type": "linear"
              },
              "showPoints": "never",
              "spanNulls": true,
              "stacking": {
                "group": "A",
                "mode": "none"
              },
              "thresholdsStyle": {
                "mode": "off"
              }
            },
            "mappings": [],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null}
              ]
            },
            "unit": "decmbytes"
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "id": 3,
        "options": {
          "legend": {
            "calcs": [],
            "displayMode": "list",
            "placement": "bottom"
          },
          "tooltip": {
            "mode": "single"
          }
        },
        "targets": [
          {
            "expr": "ai_ft_gpu_memory_used_mb",
            "legendFormat": "GPU {{gpu_id}}",
            "refId": "A"
          }
        ],
        "title": "GPU メモリ使用量",
        "type": "timeseries"
      },
      {
        "datasource": "Prometheus",
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "mappings": [
              {
                "options": {
                  "0": {"color": "red", "index": 0, "text": "利用不可"},
                  "1": {"color": "green", "index": 1, "text": "利用可能"}
                },
                "type": "value"
              }
            ],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": null},
                {"color": "green", "value": 1}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        "id": 4,
        "options": {
          "colorMode": "background",
          "graphMode": "none",
          "justifyMode": "center",
          "orientation": "auto",
          "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": false
          },
          "textMode": "value_and_name"
        },
        "targets": [
          {
            "expr": "ai_ft_gpu_available",
            "refId": "A"
          }
        ],
        "title": "GPU状態",
        "type": "stat"
      },
      {
        "datasource": "Prometheus",
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "mappings": [],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "blue", "value": null}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
        "id": 5,
        "options": {
          "colorMode": "background",
          "graphMode": "none",
          "justifyMode": "center",
          "orientation": "auto",
          "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": false
          },
          "textMode": "value_and_name"
        },
        "targets": [
          {
            "expr": "ai_ft_gpu_count",
            "refId": "A"
          }
        ],
        "title": "GPU数",
        "type": "stat"
      },
      {
        "datasource": "Prometheus",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
        "id": 6,
        "options": {
          "showHeader": true
        },
        "targets": [
          {
            "expr": "ai_ft_system_info",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ],
        "title": "システム情報",
        "type": "table"
      }
    ],
    "refresh": "5s",
    "schemaVersion": 27,
    "style": "dark",
    "tags": ["ai-ft", "monitoring"],
    "templating": {
      "list": []
    },
    "time": {
      "from": "now-30m",
      "to": "now"
    },
    "timepicker": {},
    "timezone": "",
    "title": "AI Fine-tuning Toolkit Dashboard",
    "uid": "ai-ft-main",
    "version": 1
  },
  "overwrite": true
}
EOF

echo "📌 GrafanaにAPIでダッシュボードを登録中..."

# GrafanaのAPIを使用してダッシュボードを登録
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @/tmp/ai-ft-dashboard.json \
  2>/dev/null | python3 -m json.tool

echo ""
echo "✅ ダッシュボード設定完了"
echo ""
echo "📊 アクセス方法:"
echo "  1. Grafana: http://localhost:3000"
echo "  2. ログイン: admin/admin"
echo "  3. Dashboards → Browse → AI Fine-tuning Toolkit Dashboard"
echo ""
echo "📝 利用可能なメトリクス:"
echo "  - ai_ft_cpu_usage_percent: CPU使用率"
echo "  - ai_ft_memory_usage_percent: メモリ使用率"
echo "  - ai_ft_gpu_memory_used_mb: GPU メモリ使用量"
echo "  - ai_ft_gpu_available: GPU利用可能状態"
echo "  - ai_ft_gpu_count: GPU数"
echo "  - ai_ft_system_info: システム情報"
echo ""

# メトリクスが取得できているか確認
echo "🔍 メトリクス確認:"
if curl -s http://localhost:8050/metrics | grep -q "ai_ft_"; then
    echo "  ✅ アプリケーションメトリクスが正常に出力されています"
else
    echo "  ⚠️  アプリケーションメトリクスが見つかりません"
    echo "     main_simple.pyが動作中の可能性があります"
fi

# クリーンアップ
rm -f /tmp/ai-ft-dashboard.json