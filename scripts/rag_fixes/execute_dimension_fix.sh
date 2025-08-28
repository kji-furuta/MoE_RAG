#!/bin/bash

# ベクトル次元の統一処理を実行するシェルスクリプト

set -e

echo "========================================="
echo "RAGシステム ベクトル次元統一処理"
echo "========================================="

# 色の定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 作業ディレクトリに移動
cd /home/kjifu/MoE_RAG

# 1. 現状の検証
echo -e "\n${YELLOW}ステップ 1: 現状の検証${NC}"
python3 scripts/rag_fixes/check_vector_dimensions.py || {
    echo -e "${RED}検証で問題が見つかりました。修正を続行します。${NC}"
}

# 2. ユーザー確認
echo -e "\n${YELLOW}ベクトル次元を統一します:${NC}"
echo "  モデル: intfloat/multilingual-e5-large"
echo "  次元数: 1024"
echo ""
read -p "続行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "処理を中止しました"
    exit 1
fi

# 3. 統一処理の実行
echo -e "\n${YELLOW}ステップ 2: ベクトル次元の統一処理${NC}"
python3 scripts/rag_fixes/unify_vector_dimensions.py << EOF
y
EOF

# 4. 完了メッセージ
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ ベクトル次元の統一処理が完了しました${NC}"
echo -e "${GREEN}=========================================${NC}"

echo -e "\n次のステップ:"
echo "1. Dockerコンテナ内でコレクションを再作成:"
echo "   docker exec -it ai-ft-container python /workspace/scripts/rag_fixes/recreate_qdrant_collection.py"
echo ""
echo "2. RAGシステムを再起動:"
echo "   docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh"
