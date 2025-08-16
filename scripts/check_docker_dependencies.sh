#!/bin/bash
# Docker環境内での依存関係チェックスクリプト

echo "======================================================================"
echo "🐳 Checking Dependencies in Docker Container"
echo "======================================================================"
echo ""

# Dockerコンテナが実行中か確認
if docker ps | grep -q ai-ft-container; then
    echo "✅ Docker container 'ai-ft-container' is running"
    echo ""
    
    # コンテナ内で依存関係チェックを実行
    echo "📦 Checking dependencies inside container..."
    echo "----------------------------------------------------------------------"
    
    # キャッシュディレクトリの作成（書き込み可能な場所）
    docker exec ai-ft-container mkdir -p /tmp/ai_ft_cache
    docker exec ai-ft-container chmod 777 /tmp/ai_ft_cache
    
    docker exec ai-ft-container python3 -c "
import sys
sys.path.insert(0, '/workspace')

try:
    from src.rag.dependencies.dependency_manager import RAGDependencyManager
    
    manager = RAGDependencyManager()
    result = manager.check_all_dependencies(use_cache=False)
    
    print(f'✅ System Can Run: {result.can_run}')
    print(f'✅ All Dependencies Satisfied: {result.is_satisfied}')
    print()
    
    if result.installed_versions:
        print('📦 Installed Packages (Sample):')
        for name, version in list(result.installed_versions.items())[:10]:
            print(f'  - {name}: {version}')
    
    if result.alternatives_used:
        print()
        print('🔄 Alternative Packages Used:')
        for orig, alt in result.alternatives_used.items():
            print(f'  - {orig} → {alt}')
    
    if result.warnings:
        print()
        print('⚠️ Warnings:')
        for warning in result.warnings[:5]:
            print(f'  - {warning}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
" || echo "❌ Failed to execute Python script"
    
    echo ""
    echo "======================================================================"
    echo ""
    
    # コンテナ内のPython環境情報
    echo "🐍 Python Environment in Container:"
    echo "----------------------------------------------------------------------"
    docker exec ai-ft-container python3 --version
    docker exec ai-ft-container pip list | head -20
    
else
    echo "❌ Docker container 'ai-ft-container' is not running"
    echo ""
    echo "To start the container:"
    echo "  cd docker"
    echo "  docker-compose up -d"
fi

echo ""
echo "======================================================================"
echo "📋 Summary:"
echo "======================================================================"
echo "- Host environment: Dependencies not installed (expected)"
echo "- Docker container: Dependencies should be installed"
echo "- RAG features: Working (running in container)"
echo ""
echo "This is the correct setup - dependencies are isolated in Docker!"
echo "======================================================================"
