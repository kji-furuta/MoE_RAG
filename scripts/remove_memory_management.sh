#!/bin/bash

# メモリ管理システム関連のファイルと参照を削除

echo "================================"
echo "メモリ管理システムの削除"
echo "================================"

cd /home/kjifu/MoE_RAG || exit 1

# 1. メモリ管理関連ファイルを削除
echo "1. メモリ管理関連ファイルを削除中..."

# 既に削除されているかもしれないが、念のため確認して削除
rm -f src/core/memory_manager.py 2>/dev/null
rm -f src/core/quantization_manager.py 2>/dev/null
rm -f src/core/docker_memory_patch.py 2>/dev/null
rm -f app/memory_optimized_loader_v2.py 2>/dev/null
rm -f scripts/migrate_memory_management.py 2>/dev/null
rm -f scripts/test_memory_management.py 2>/dev/null
rm -f MEMORY_MANAGEMENT_IMPROVEMENT.md 2>/dev/null

# coreディレクトリが空なら削除
if [ -d src/core ] && [ -z "$(ls -A src/core 2>/dev/null)" ]; then
    rmdir src/core 2>/dev/null
fi

echo "✓ メモリ管理ファイルを削除しました"

# 2. app/memory_optimized_loader.pyからメモリ管理システムへの参照を削除
echo "2. memory_optimized_loader.pyを確認中..."

if [ -f app/memory_optimized_loader.py ]; then
    # メモリ管理システムへの参照があるか確認
    if grep -q "from src.core.memory_manager import" app/memory_optimized_loader.py 2>/dev/null; then
        echo "   メモリ管理システムへの参照を削除中..."
        
        # バックアップを作成
        cp app/memory_optimized_loader.py app/memory_optimized_loader.py.bak
        
        # メモリ管理システムへの参照を削除し、元の実装に戻す
        sed -i '/from src.core.memory_manager import/d' app/memory_optimized_loader.py
        sed -i '/from src.core.quantization_manager import/d' app/memory_optimized_loader.py
        sed -i '/memory_manager = get_memory_manager/d' app/memory_optimized_loader.py
        sed -i '/quantization_manager = QuantizationManager/d' app/memory_optimized_loader.py
        
        echo "✓ 参照を削除しました"
    else
        echo "✓ メモリ管理システムへの参照はありません"
    fi
fi

# 3. main_unified.pyからメモリ管理システムへの参照を削除
echo "3. main_unified.pyを確認中..."

if [ -f app/main_unified.py ]; then
    if grep -q "docker_memory_patch" app/main_unified.py 2>/dev/null; then
        echo "   Docker memory patchへの参照を削除中..."
        
        # バックアップを作成
        cp app/main_unified.py app/main_unified.py.bak
        
        # Docker memory patch関連のコードを削除
        sed -i '/from src.core.docker_memory_patch import/d' app/main_unified.py
        sed -i '/patch_memory_manager_for_docker/d' app/main_unified.py
        
        echo "✓ 参照を削除しました"
    else
        echo "✓ Docker memory patchへの参照はありません"
    fi
fi

# 4. その他のファイルからメモリ管理への参照を確認
echo "4. その他のファイルを確認中..."

# grepで参照を探す
echo "   残存する参照を検索中..."
REFS=$(grep -r "memory_manager\|MemoryManager\|quantization_manager\|QuantizationManager" \
    --include="*.py" \
    --exclude-dir=".git" \
    --exclude-dir="__pycache__" \
    --exclude-dir="models" \
    --exclude-dir="outputs" \
    --exclude="*.bak" \
    app/ src/ scripts/ 2>/dev/null | \
    grep -v "# Memory management" | \
    grep -v "memory_optimized_loader.py.bak" || true)

if [ -n "$REFS" ]; then
    echo "⚠️ 以下のファイルにまだ参照が残っています："
    echo "$REFS" | cut -d: -f1 | sort -u
else
    echo "✓ メモリ管理システムへの参照はすべて削除されました"
fi

# 5. 環境変数の設定を元に戻す
echo "5. 環境変数設定を確認中..."

# CUDA_LAUNCH_BLOCKINGを1に設定（元の設定）
if [ -f app/memory_optimized_loader.py ]; then
    if grep -q 'CUDA_LAUNCH_BLOCKING.*=.*"0"' app/memory_optimized_loader.py 2>/dev/null; then
        sed -i 's/os.environ\["CUDA_LAUNCH_BLOCKING"\] = "0"/os.environ["CUDA_LAUNCH_BLOCKING"] = "1"/g' app/memory_optimized_loader.py
        echo "✓ CUDA_LAUNCH_BLOCKINGを1に設定しました"
    fi
fi

# 6. 不要なバックアップファイルを削除
echo "6. クリーンアップ中..."
find . -name "*.bak" -type f -delete 2>/dev/null
find . -name "*_backup.py" -type f -delete 2>/dev/null
echo "✓ クリーンアップ完了"

echo ""
echo "================================"
echo "削除完了"
echo "================================"
echo ""
echo "メモリ管理システムは完全に削除されました。"
echo ""
echo "次のコマンドでDockerを再起動してください："
echo "  cd docker"
echo "  docker-compose restart"
echo ""
echo "または新しくビルドする場合："
echo "  docker-compose down"
echo "  docker-compose build"
echo "  docker-compose up -d"
echo ""
