#!/bin/bash

# メタデータ管理統合の実行スクリプト

set -e

echo "========================================="
echo "🔧 メタデータ管理の統合"
echo "========================================="

# 色の定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "\n${BLUE}$1${NC}"
    echo "----------------------------------------"
}

# 作業ディレクトリ
WORK_DIR="/home/kjifu/MoE_RAG"
cd $WORK_DIR

# 1. 現状の確認
print_header "ステップ1: 現状の確認"

# MetadataManagerの確認
if [ -f "metadata/metadata.db" ]; then
    DB_SIZE=$(du -h metadata/metadata.db | cut -f1)
    print_info "MetadataManager DB: $DB_SIZE"
    
    # レコード数の確認
    python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('metadata/metadata.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM document_metadata')
    count = cursor.fetchone()[0]
    print(f'  レコード数: {count}')
    conn.close()
except Exception as e:
    print(f'  エラー: {e}')
" || print_info "  MetadataManagerにアクセスできません"
else
    print_info "MetadataManager DBが存在しません"
fi

# Qdrantの確認
if [ -d "data/qdrant" ]; then
    print_info "Qdrant データディレクトリ: 存在"
    
    # meta.jsonの確認
    if [ -f "data/qdrant/meta.json" ]; then
        python3 -c "
import json
with open('data/qdrant/meta.json', 'r') as f:
    meta = json.load(f)
    for collection in meta.get('collections', {}):
        print(f'  コレクション: {collection}')
"
    fi
else
    print_info "Qdrantディレクトリが存在しません"
fi

# 2. 統合ストアの初期化
print_header "ステップ2: 統合ストアの初期化"

python3 scripts/rag_fixes/unified_metadata_store.py
if [ $? -eq 0 ]; then
    print_success "統合ストアの初期化完了"
else
    print_error "統合ストアの初期化失敗"
    exit 1
fi

# 3. 既存データの移行
print_header "ステップ3: 既存データの移行"

python3 scripts/rag_fixes/integrate_metadata.py
if [ $? -eq 0 ]; then
    print_success "データ移行完了"
else
    print_error "データ移行失敗"
    exit 1
fi

# 4. 移行結果の確認
print_header "ステップ4: 移行結果の確認"

# 統合データベースのサイズ
if [ -f "data/unified_metadata/unified_store.db" ]; then
    DB_SIZE=$(du -h data/unified_metadata/unified_store.db | cut -f1)
    print_success "統合DB作成: $DB_SIZE"
    
    # 統計情報の表示
    python3 -c "
from scripts.rag_fixes.unified_metadata_store import UnifiedMetadataStore
store = UnifiedMetadataStore()
stats = store.get_statistics()

print(f'')
print(f'📊 統合後の統計:')
print(f'  総文書数: {stats[\"total_documents\"]}')
print(f'  ベクトル化済み: {stats[\"vectorized_documents\"]}')
print(f'  ベクトル化率: {stats[\"vectorization_rate\"]:.1f}%')

if stats['status_counts']:
    print(f'')
    print(f'  ステータス別:')
    for status, count in stats['status_counts'].items():
        print(f'    {status}: {count}件')

if stats['category_counts']:
    print(f'')
    print(f'  カテゴリ別:')
    for category, count in stats['category_counts'].items():
        print(f'    {category}: {count}件')
"
fi

# 5. レポートの確認
print_header "ステップ5: 移行レポート"

if [ -f "data/unified_metadata/migration_report.json" ]; then
    print_success "移行レポート生成完了"
    echo ""
    echo "レポート内容（抜粋）:"
    python3 -c "
import json
with open('data/unified_metadata/migration_report.json', 'r') as f:
    report = json.load(f)
    print(f'  タイムスタンプ: {report[\"timestamp\"]}')
    print(f'  総文書数: {report[\"total_documents\"]}')
    print(f'  ベクトル化率: {report[\"vectorization_rate\"]:.1f}%')
"
fi

# 完了メッセージ
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ メタデータ管理の統合が完了しました${NC}"
echo -e "${GREEN}=========================================${NC}"

echo -e "\n📋 実装された改善:"
echo "  ✅ 統一されたID管理（UUIDベース）"
echo "  ✅ メタデータとベクトルの一元管理"
echo "  ✅ 重複排除（content_hashベース）"
echo "  ✅ IDマッピング（後方互換性）"
echo "  ✅ 統合検索API"

echo -e "\n🚀 次のステップ:"
echo "1. システムの再起動:"
echo "   ./stop_dev_env.sh && ./start_dev_env.sh"
echo ""
echo "2. 統合APIのテスト:"
echo "   python3 scripts/rag_fixes/test_unified_api.py"
echo ""
echo "3. 古いシステムの無効化（オプション）:"
echo "   # MetadataManagerとQdrantの直接使用を停止"
echo "   # 統合APIを経由するよう既存コードを更新"

echo -e "\n📁 データ保存場所:"
echo "  統合DB: data/unified_metadata/unified_store.db"
echo "  ベクトル: data/unified_metadata/vectors/"
echo "  レポート: data/unified_metadata/migration_report.json"
