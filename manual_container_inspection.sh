#!/bin/bash
# manual_container_inspection.sh
# コンテナ内部を手動で詳細調査

echo "=== コンテナ内部の手動調査 ==="
echo ""

# 1. コンテナ内のワークスペース構造を確認
echo "1. /workspace ディレクトリの内容:"
docker exec ai-ft-container ls -la /workspace/

echo ""
echo "2. 実際に存在するディレクトリ構造（第1階層）:"
docker exec ai-ft-container find /workspace -maxdepth 1 -type d | sort

echo ""
echo "3. srcディレクトリが存在するか確認:"
docker exec ai-ft-container ls -la /workspace/src/ 2>&1

echo ""
echo "4. 設定ファイルを探す:"
docker exec ai-ft-container find /workspace -name "*.yaml" -o -name "*.yml" | head -10

echo ""
echo "5. Pythonファイルを探す:"
docker exec ai-ft-container find /workspace -name "*.py" | head -20

echo ""
echo "6. requirements.txtの場所:"
docker exec ai-ft-container find /workspace -name "requirements*.txt"

echo ""
echo "7. ボリュームマウントの実際の状態:"
docker inspect ai-ft-container --format='{{json .Mounts}}' | python3 -m json.tool

echo ""
echo "8. コンテナの作業ディレクトリ:"
docker exec ai-ft-container pwd

echo ""
echo "9. 環境変数PYTHONPATH:"
docker exec ai-ft-container printenv PYTHONPATH
