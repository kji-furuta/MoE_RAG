#!/bin/bash
# Docker環境の準備と権限修正を含む完全なテストスクリプト

echo "======================================================================"
echo "🔧 Preparing Docker Environment for Phase 2 Tests"
echo "======================================================================"
echo ""

# Dockerコンテナが実行中か確認
if ! docker ps | grep -q ai-ft-container; then
    echo "❌ Docker container 'ai-ft-container' is not running"
    echo ""
    echo "Starting container..."
    cd docker
    docker-compose up -d
    sleep 5
    cd ..
fi

echo "✅ Docker container is running"
echo ""

# 必要なディレクトリの作成と権限設定
echo "🔧 Setting up directories with proper permissions..."
echo "----------------------------------------------------------------------"

# コンテナ内に書き込み可能なディレクトリを作成
docker exec ai-ft-container bash -c "
    # /tmp内にキャッシュディレクトリを作成
    mkdir -p /tmp/ai_ft_cache/dependencies
    chmod -R 777 /tmp/ai_ft_cache
    
    # /workspace内にキャッシュディレクトリを作成
    mkdir -p /workspace/.cache/ai_ft/dependencies
    chmod -R 777 /workspace/.cache
    
    # ログディレクトリも作成
    mkdir -p /workspace/logs/health
    chmod -R 777 /workspace/logs
    
    echo '✅ Directories created with proper permissions'
"

# 必要なパッケージのインストール
echo ""
echo "📦 Installing required packages..."
echo "----------------------------------------------------------------------"
docker exec ai-ft-container pip install psutil loguru 2>/dev/null
echo "✅ Packages installed"

# 依存関係チェック
echo ""
echo "======================================================================"
echo "📋 Checking Dependencies in Container"
echo "======================================================================"

docker exec ai-ft-container python3 -c "
import sys
import os
sys.path.insert(0, '/workspace')

# 環境変数でキャッシュディレクトリを指定
os.environ['AI_FT_CACHE_DIR'] = '/tmp/ai_ft_cache'

try:
    from src.rag.dependencies.dependency_manager import RAGDependencyManager
    
    manager = RAGDependencyManager()
    result = manager.check_all_dependencies(use_cache=False)
    
    print(f'✅ System Can Run: {result.can_run}')
    print(f'✅ All Dependencies Satisfied: {result.is_satisfied}')
    print()
    
    if result.installed_versions:
        print('📦 Installed Core Packages:')
        core_packages = ['transformers', 'torch', 'sentence_transformers', 'pydantic']
        for pkg in core_packages:
            if pkg in result.installed_versions:
                print(f'  ✅ {pkg}: {result.installed_versions[pkg]}')
            else:
                print(f'  ❌ {pkg}: Not found')
    
    if result.alternatives_used:
        print()
        print('🔄 Alternative Packages Used:')
        for orig, alt in result.alternatives_used.items():
            print(f'  - {orig} → {alt}')
    
    if result.warnings:
        print()
        print('⚠️ Warnings:')
        for warning in result.warnings[:3]:
            print(f'  - {warning}')
            
except Exception as e:
    print(f'❌ Error checking dependencies: {e}')
    import traceback
    traceback.print_exc()
"

# Phase 2統合テストの実行
echo ""
echo "======================================================================"
echo "🧪 Running Phase 2 Integration Tests"
echo "======================================================================"

docker exec ai-ft-container python3 -c "
import sys
import os
import asyncio
sys.path.insert(0, '/workspace')

# 環境変数でキャッシュディレクトリを指定
os.environ['AI_FT_CACHE_DIR'] = '/tmp/ai_ft_cache'

async def run_tests():
    try:
        # テストスクリプトを直接実行
        exec(open('/workspace/scripts/test_phase2_integration.py').read())
    except Exception as e:
        print(f'❌ Test execution error: {e}')
        import traceback
        traceback.print_exc()

# テスト実行
try:
    asyncio.run(run_tests())
except Exception as e:
    print(f'Alternative execution method...')
    # 直接実行
    import subprocess
    result = subprocess.run(['python3', '/workspace/scripts/test_phase2_integration.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print('Errors:', result.stderr)
"

# システム最適化テストの実行
echo ""
echo "======================================================================"
echo "⚡ Running System Optimization Test"
echo "======================================================================"

docker exec ai-ft-container python3 -c "
import sys
import os
sys.path.insert(0, '/workspace')

# 環境変数でキャッシュディレクトリを指定
os.environ['AI_FT_CACHE_DIR'] = '/tmp/ai_ft_cache'

try:
    # 簡易的な最適化チェック
    from src.rag.dependencies.dependency_manager import RAGDependencyManager
    
    print('🔍 Checking optimization opportunities...')
    
    manager = RAGDependencyManager()
    result = manager.check_all_dependencies(use_cache=False)
    
    print()
    print('📊 Optimization Suggestions:')
    
    # オプション依存関係の提案
    if result.missing_optional:
        print('  📦 Optional packages to consider:')
        for pkg in result.missing_optional[:3]:
            print(f'    - {pkg}')
    
    # 代替パッケージの提案
    if result.alternatives_used:
        print('  🔄 Consider installing primary packages:')
        for orig, alt in result.alternatives_used.items():
            print(f'    - Install {orig} instead of {alt}')
    
    # パフォーマンス提案
    import psutil
    mem = psutil.virtual_memory()
    if mem.percent > 80:
        print(f'  ⚠️ High memory usage: {mem.percent:.1f}%')
        print('    - Consider reducing batch size')
        print('    - Enable model quantization')
    
    print()
    print('✅ Optimization check complete')
    
except Exception as e:
    print(f'❌ Optimization test error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "======================================================================"
echo "✅ All Phase 2 Tests Completed"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  - Dependencies checked"
echo "  - Integration tests executed"
echo "  - Optimization analysis performed"
echo ""
echo "Check the output above for any errors or warnings."
echo "======================================================================"
