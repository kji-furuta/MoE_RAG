#!/bin/bash

# データ永続性改善の実行スクリプト

set -e

echo "========================================="
echo "🔧 RAGシステム データ永続性改善"
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
cd /home/kjifu/MoE_RAG

# 1. 現状の確認
echo -e "\n📊 ステップ1: 現状の確認"
echo "----------------------------------------"
python3 scripts/rag_fixes/check_data_persistence.py || {
    print_info "データ永続性に問題があります。修正を続行します。"
}

# 2. ユーザー確認
echo -e "\n⚠️  データ永続性の改善を実行します"
echo "以下の機能が追加されます:"
echo "  • 永続化データベース (SQLite + Qdrant)"
echo "  • 自動バックアップシステム"
echo "  • データ復元機能"
echo "  • インクリメンタルバックアップ"
echo ""
read -p "続行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "処理を中止しました"
    exit 1
fi

# 3. 永続化ストアの実装
echo -e "\n🔨 ステップ2: 永続化ストアの実装"
echo "----------------------------------------"
python3 scripts/rag_fixes/fix_data_persistence.py
if [ $? -eq 0 ]; then
    print_success "永続化ストア実装完了"
else
    print_error "永続化ストア実装失敗"
    exit 1
fi

# 4. 既存システムとの統合
echo -e "\n🔗 ステップ3: 既存システムとの統合"
echo "----------------------------------------"
python3 scripts/rag_fixes/integrate_persistence.py
if [ $? -eq 0 ]; then
    print_success "統合完了"
else
    print_error "統合失敗"
    exit 1
fi

# 5. 自動バックアップの設定
echo -e "\n⏰ ステップ4: 自動バックアップの設定"
echo "----------------------------------------"
python3 scripts/rag_fixes/setup_auto_backup.py
if [ $? -eq 0 ]; then
    print_success "自動バックアップ設定完了"
else
    print_error "自動バックアップ設定失敗"
    exit 1
fi

# 6. 完了メッセージ
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ データ永続性の改善が完了しました${NC}"
echo -e "${GREEN}=========================================${NC}"

echo -e "\n📋 実装された機能:"
echo "  1. 永続化データベース"
echo "     - SQLite: メタデータと文書管理"
echo "     - Qdrant: ベクトルインデックス"
echo "     - 重複チェック機能"
echo ""
echo "  2. バックアップシステム"
echo "     - 手動バックアップ: いつでも実行可能"
echo "     - 自動バックアップ: 10文書ごと"
echo "     - スケジュールバックアップ: 日次/週次"
echo ""
echo "  3. データ復元機能"
echo "     - バックアップからの完全復元"
echo "     - インクリメンタル復元"
echo ""

echo -e "\n🚀 次のステップ:"
echo "1. Dockerコンテナで動作確認:"
echo "   docker exec -it ai-ft-container python /workspace/scripts/rag_fixes/verify_persistence.py"
echo ""
echo "2. 既存データの移行（オプション）:"
echo "   docker exec -it ai-ft-container python /workspace/scripts/rag_fixes/migrate_data.py"
echo ""
echo "3. システム再起動:"
echo "   ./stop_dev_env.sh"
echo "   ./start_dev_env.sh"
echo ""

echo -e "\n📁 データ保存場所:"
echo "  永続化データ: data/rag_persistent/"
echo "  バックアップ: data/rag_persistent/backups/"
echo "  メタデータ: data/rag_persistent/metadata/"
