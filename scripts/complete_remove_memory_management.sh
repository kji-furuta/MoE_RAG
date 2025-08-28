#!/bin/bash

# メモリ管理システムを完全に削除

echo "================================"
echo "メモリ管理システムの完全削除"
echo "================================"

# 作業ディレクトリに移動
cd /home/kjifu/MoE_RAG || exit 1

# 1. src/core内のメモリ管理ファイルを削除
echo "1. src/core内のメモリ管理ファイルを削除..."
if [ -d src/core ]; then
    rm -f src/core/memory_manager.py
    rm -f src/core/quantization_manager.py
    rm -f src/core/docker_memory_patch.py
    rm -rf src/core/__pycache__
    
    # coreディレクトリが空なら削除
    if [ -z "$(ls -A src/core 2>/dev/null)" ]; then
        rmdir src/core
        echo "   ✓ src/coreディレクトリを削除しました"
    fi
else
    echo "   ✓ src/coreディレクトリは存在しません"
fi

# 2. app内のメモリ管理関連ファイルを削除
echo "2. app内のメモリ管理関連ファイルを削除..."
rm -f app/memory_optimized_loader_v2.py
rm -f app/main_unified_docker.py
rm -f app/main_unified_lightweight.py
echo "   ✓ appディレクトリ内のメモリ管理ファイルを削除しました"

# 3. scripts内のメモリ管理関連ファイルを削除
echo "3. scripts内のメモリ管理関連ファイルを削除..."
rm -f scripts/migrate_memory_management.py
rm -f scripts/test_memory_management.py
rm -f scripts/rollback_memory_changes.sh
rm -f scripts/remove_memory_management.sh
rm -f scripts/start_web_interface_docker.sh
rm -f scripts/start_highspec.sh
rm -f scripts/diagnose_docker_resources.sh
rm -f scripts/check_docker_resources.sh
echo "   ✓ scriptsディレクトリ内のメモリ管理ファイルを削除しました"

# 4. docker内のメモリ管理関連ファイルを削除
echo "4. docker内のメモリ管理関連ファイルを削除..."
rm -f docker/entrypoint.sh
rm -f docker/entrypoint_lightweight.sh
rm -f docker/docker-compose-highspec.yml
rm -f docker/wslconfig.example
echo "   ✓ dockerディレクトリ内のメモリ管理ファイルを削除しました"

# 5. その他のファイルを削除
echo "5. その他のメモリ管理関連ファイルを削除..."
rm -f src/rag/config/lightweight_config.py
rm -f MEMORY_MANAGEMENT_IMPROVEMENT.md
echo "   ✓ その他のメモリ管理ファイルを削除しました"

# 6. Dockerfileを元に戻す
echo "6. Dockerfileを元の状態に復元..."
if [ -f docker/Dockerfile ]; then
    # エントリーポイント関連の行を削除して元に戻す
    sed -i '/# エントリーポイントスクリプトをコピー/,/CMD \[\]/d' docker/Dockerfile 2>/dev/null || true
    
    # 元の末尾を追加（既に存在しない場合のみ）
    if ! grep -q "CMD \[\"/bin/bash\"\]" docker/Dockerfile; then
        cat >> docker/Dockerfile << 'EOF'

# 作業ユーザーを切り替え
USER ai-user

# ポートを公開
EXPOSE 8888 6006 8050 8051

# デフォルトコマンド（bashシェルを起動）
CMD ["/bin/bash"]
EOF
    fi
    echo "   ✓ Dockerfileを復元しました"
fi

# 7. docker-compose.ymlを元に戻す
echo "7. docker-compose.ymlを確認..."
if [ -f docker/docker-compose.yml ]; then
    # メモリ管理関連の環境変数を削除
    sed -i '/DOCKER_CONTAINER=true/d' docker/docker-compose.yml 2>/dev/null || true
    sed -i '/RAG_DISABLE_AUTO_INIT=true/d' docker/docker-compose.yml 2>/dev/null || true
    sed -i '/RAG_EMBEDDING_MODEL=sentence-transformers/d' docker/docker-compose.yml 2>/dev/null || true
    sed -i '/RAG_USE_CPU=true/d' docker/docker-compose.yml 2>/dev/null || true
    sed -i '/RAG_LIGHTWEIGHT_MODE=true/d' docker/docker-compose.yml 2>/dev/null || true
    sed -i '/MEMORY_MANAGER_INITIALIZED=1/d' docker/docker-compose.yml 2>/dev/null || true
    echo "   ✓ docker-compose.ymlをクリーンアップしました"
fi

# 8. バックアップファイルを削除
echo "8. バックアップファイルを削除..."
find . -name "*.bak" -type f -delete 2>/dev/null
find . -name "*_backup*" -type f -delete 2>/dev/null
echo "   ✓ バックアップファイルを削除しました"

# 9. Pythonキャッシュを削除
echo "9. Pythonキャッシュを削除..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ Pythonキャッシュを削除しました"

# 10. 最終確認
echo ""
echo "================================"
echo "削除結果の確認"
echo "================================"

# メモリ管理関連のファイルが残っていないか確認
echo "残存ファイル確認:"
REMAINING=$(find . -type f \( \
    -name "*memory_manager*" -o \
    -name "*quantization_manager*" -o \
    -name "*memory_optimized_loader_v2*" -o \
    -name "*docker_memory_patch*" -o \
    -name "*entrypoint*.sh" -o \
    -name "*lightweight*" -o \
    -name "*highspec*" \
    \) 2>/dev/null | grep -v ".git" | grep -v "__pycache__" | grep -v ".bak")

if [ -n "$REMAINING" ]; then
    echo "⚠️ 以下のファイルがまだ残っています:"
    echo "$REMAINING"
else
    echo "✅ メモリ管理関連のファイルはすべて削除されました"
fi

# src/coreディレクトリの状態確認
if [ -d src/core ]; then
    echo ""
    echo "src/coreディレクトリの内容:"
    ls -la src/core/
else
    echo ""
    echo "✅ src/coreディレクトリは削除されました"
fi

echo ""
echo "================================"
echo "削除完了"
echo "================================"
echo ""
echo "メモリ管理システムは完全に削除されました。"
echo ""
echo "Dockerコンテナを再起動してください："
echo "  cd docker"
echo "  docker-compose down"
echo "  docker-compose up -d"
echo ""
echo "その後、通常の起動スクリプトを実行："
echo "  docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh"
echo ""
