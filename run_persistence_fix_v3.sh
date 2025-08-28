#!/bin/bash

# データ永続性改善の実行スクリプト（v3）

set -e

echo "========================================="
echo "🔧 RAGシステム データ永続性改善 v3"
echo "========================================="

# 色の定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 作業ディレクトリ
WORK_DIR="/home/kjifu/MoE_RAG"
cd $WORK_DIR

# クリーンアップの確認
echo -e "\n🧹 既存データのクリーンアップ"
echo "----------------------------------------"
if [ -d "data/rag_persistent/backups/initial_backup" ]; then
    print_info "既存のinitial_backupが見つかりました"
    rm -rf data/rag_persistent/backups/initial_backup
    print_success "削除完了"
fi

# 永続化ストアの実装（v3を使用）
echo -e "\n🔨 永続化ストアの実装"
echo "----------------------------------------"
python3 scripts/rag_fixes/fix_data_persistence_v3.py

if [ $? -eq 0 ]; then
    print_success "永続化ストア実装完了"
    
    # バックアップディレクトリの内容を確認
    echo -e "\n📁 バックアップディレクトリの確認"
    echo "----------------------------------------"
    if [ -d "data/rag_persistent/backups" ]; then
        echo "バックアップ一覧:"
        ls -la data/rag_persistent/backups/ | head -10
    fi
    
    # データベースサイズの確認
    echo -e "\n💾 データベースサイズ"
    echo "----------------------------------------"
    if [ -f "data/rag_persistent/metadata/persistent_store.db" ]; then
        DB_SIZE=$(du -h data/rag_persistent/metadata/persistent_store.db | cut -f1)
        echo "  persistent_store.db: $DB_SIZE"
    fi
    
else
    print_error "永続化ストア実装失敗"
    exit 1
fi

# 完了メッセージ
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ データ永続性の改善が完了しました${NC}"
echo -e "${GREEN}=========================================${NC}"

echo -e "\n📋 実装された機能:"
echo "  ✅ SQLiteデータベース（文書とベクトル管理）"
echo "  ✅ 自動バックアップ（10文書ごと）"
echo "  ✅ 一意のバックアップ名生成"
echo "  ✅ バックアップからの復元機能"
echo "  ✅ 古いバックアップのクリーンアップ"

echo -e "\n🚀 次のステップ:"
echo "1. バックアップリストの確認:"
echo "   python3 -c \"from scripts.rag_fixes.fix_data_persistence_v3 import PersistentVectorStore; store = PersistentVectorStore(); print('\\n'.join([b['name'] for b in store.list_backups()]))\""
echo ""
echo "2. システムの再起動:"
echo "   ./stop_dev_env.sh && ./start_dev_env.sh"
echo ""
echo "3. Dockerでの動作確認:"
echo "   docker exec -it ai-ft-container python /workspace/scripts/rag_fixes/verify_persistence.py"
