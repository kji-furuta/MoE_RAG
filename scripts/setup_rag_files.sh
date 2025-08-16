#!/bin/bash
echo "📁 RAGシステム用ファイルをコピー中..."

  # 必要なファイルをコンテナにコピー
  docker cp /home/kjifuruta/AI_FT/AI_FT_3/templates/rag.html ai-ft-container:/workspace/templates/rag.html
  docker cp /home/kjifuruta/AI_FT/AI_FT_3/templates/base.html ai-ft-container:/workspace/templates/base.html
  docker cp /home/kjifuruta/AI_FT/AI_FT_3/templates/index.html ai-ft-container:/workspace/templates/index.html
  docker cp /home/kjifuruta/AI_FT/AI_FT_3/app/main_unified.py ai-ft-container:/workspace/app/main_unified.py

  # 権限設定
  docker exec ai-ft-container chown -R ai-user:ai-user /workspace/templates/ /workspace/app/

echo "✅ RAGファイルコピー完了"

docker exec ai-ft-container pkill -f "uvicorn"
sleep 5
docker exec -d ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050
echo "✅ ファイルコピー完了、サーバー再起動中..."