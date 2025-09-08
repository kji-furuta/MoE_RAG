#!/bin/bash

# ログクリーンアップスクリプト
# システムは正常稼働中なので、ログファイルのクリーンアップのみ行う

echo "🧹 ログファイルのクリーンアップを開始..."

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ログディレクトリ
LOG_DIRS=(
    "/home/kjifu/MoE_RAG/logs"
    "/home/kjifu/MoE_RAG/docker/logs"
    "/workspace/logs"
)

# クリーンアップ実行
for dir in "${LOG_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "📁 $dir を確認中..."
        
        # 大きなログファイルを検索（1MB以上）
        find "$dir" -name "*.log" -size +1M -type f -exec ls -lh {} \; 2>/dev/null
        
        # 古いログファイルを削除（7日以上前）
        find "$dir" -name "*.log" -mtime +7 -type f -delete 2>/dev/null
        
        # 空のログファイルを削除
        find "$dir" -name "*.log" -empty -type f -delete 2>/dev/null
    fi
done

# Dockerコンテナログのトランケート（サイズが大きい場合のみ）
echo ""
echo "🐳 Dockerコンテナログを確認中..."

# コンテナログサイズ確認
CONTAINERS=("ai-ft-container" "ai-ft-qdrant")
for container in "${CONTAINERS[@]}"; do
    if docker ps | grep -q "$container"; then
        # ログファイルパスを取得
        LOG_PATH=$(docker inspect --format='{{.LogPath}}' "$container" 2>/dev/null)
        if [ -n "$LOG_PATH" ] && [ -f "$LOG_PATH" ]; then
            SIZE=$(du -h "$LOG_PATH" 2>/dev/null | cut -f1)
            echo "  $container: $SIZE"
            
            # 100MB以上の場合はトランケート
            SIZE_BYTES=$(stat -c%s "$LOG_PATH" 2>/dev/null)
            if [ "$SIZE_BYTES" -gt 104857600 ]; then
                echo -e "${YELLOW}  → ログが大きいため、トランケートします${NC}"
                echo "" | sudo tee "$LOG_PATH" > /dev/null
                echo -e "${GREEN}  ✅ トランケート完了${NC}"
            fi
        fi
    fi
done

echo ""
echo -e "${GREEN}✅ ログクリーンアップが完了しました${NC}"
echo ""
echo "💡 ヒント:"
echo "  - Dockerログの詳細度を下げる: docker-compose.ymlでlogging設定を調整"
echo "  - uvicornログを抑制: --log-level warning オプションを追加"
echo "  - 定期的にこのスクリプトを実行: crontab -e で設定"