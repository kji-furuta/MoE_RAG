#!/bin/bash
# モデルダウンロードの進捗を監視するスクリプト

echo "DeepSeek-R1モデルのダウンロード進捗を監視しています..."
echo "Ctrl+Cで終了できます"
echo ""

while true; do
    # ログファイルの最後の行を表示
    if docker exec ai-ft-container test -f /workspace/logs/deepseek_download.log; then
        echo -ne "\r$(docker exec ai-ft-container tail -1 /workspace/logs/deepseek_download.log | tr -d '\n')"
        
        # ダウンロード完了をチェック
        if docker exec ai-ft-container grep -q "モデルのダウンロードが完了しました" /workspace/logs/deepseek_download.log; then
            echo ""
            echo ""
            echo "✅ ダウンロードが完了しました！"
            docker exec ai-ft-container tail -20 /workspace/logs/deepseek_download.log
            break
        fi
        
        # エラーをチェック
        if docker exec ai-ft-container grep -q "モデルのダウンロードに失敗しました" /workspace/logs/deepseek_download.log; then
            echo ""
            echo ""
            echo "❌ ダウンロードに失敗しました"
            docker exec ai-ft-container tail -20 /workspace/logs/deepseek_download.log
            break
        fi
    fi
    
    sleep 2
done