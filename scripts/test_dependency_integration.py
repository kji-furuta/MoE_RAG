#!/usr/bin/env python3
"""
依存関係管理システムの統合テスト

このスクリプトは、新しく実装した依存関係管理機能が
正しく動作することを確認します。
"""

import sys
import os
from pathlib import Path
import json
import time

# プロジェクトのルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.dependencies.dependency_manager import (
    RAGDependencyManager,
    DependencyLevel,
    DependencyCheckResult
)


def test_basic_functionality():
    """基本機能のテスト"""
    print("=" * 60)
    print("1. Testing Basic Functionality")
    print("=" * 60)
    
    try:
        # マネージャーの初期化
        manager = RAGDependencyManager()
        print("✅ Manager initialized successfully")
        
        # 依存関係の定義確認
        deps = manager.dependencies
        print(f"✅ Found {len(deps)} dependency definitions")
        
        # 各レベルの依存関係をカウント
        core_count = sum(1 for d in deps.values() if d.level == DependencyLevel.CORE)
        infra_count = sum(1 for d in deps.values() if d.level == DependencyLevel.INFRASTRUCTURE)
        optional_count = sum(1 for d in deps.values() if d.level == DependencyLevel.OPTIONAL)
        
        print(f"   - Core: {core_count}")
        print(f"   - Infrastructure: {infra_count}")
        print(f"   - Optional: {optional_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_dependency_check():
    """依存関係チェックのテスト"""
    print("\n" + "=" * 60)
    print("2. Testing Dependency Check")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        # キャッシュなしでチェック
        print("Checking dependencies (no cache)...")
        start_time = time.time()
        result = manager.check_all_dependencies(use_cache=False)
        elapsed = time.time() - start_time
        
        print(f"✅ Check completed in {elapsed:.2f} seconds")
        
        # 結果の表示
        print(f"\nCheck Results:")
        print(f"  - Can run: {result.can_run}")
        print(f"  - All satisfied: {result.is_satisfied}")
        print(f"  - Missing core: {len(result.missing_core)}")
        print(f"  - Missing infrastructure: {len(result.missing_infrastructure)}")
        print(f"  - Missing optional: {len(result.missing_optional)}")
        print(f"  - Warnings: {len(result.warnings)}")
        print(f"  - Alternatives used: {len(result.alternatives_used)}")
        
        # インストール済みバージョンの表示（一部）
        if result.installed_versions:
            print(f"\nSample installed versions:")
            for name, version in list(result.installed_versions.items())[:5]:
                print(f"  - {name}: {version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency check test failed: {e}")
        return False


def test_report_generation():
    """レポート生成のテスト"""
    print("\n" + "=" * 60)
    print("3. Testing Report Generation")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        # 各形式でレポート生成
        formats = ["text", "json", "markdown"]
        
        for format in formats:
            print(f"\nGenerating {format} report...")
            report = manager.get_dependency_report(format=format)
            
            if format == "json":
                # JSONの妥当性チェック
                data = json.loads(report)
                print(f"✅ JSON report is valid with {len(data)} keys")
            else:
                # レポートサイズの確認
                lines = report.split('\n')
                print(f"✅ {format.capitalize()} report generated ({len(lines)} lines)")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False


def test_cache_functionality():
    """キャッシュ機能のテスト"""
    print("\n" + "=" * 60)
    print("4. Testing Cache Functionality")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        # キャッシュなしでチェック（初回）
        print("First check (no cache)...")
        start_time = time.time()
        result1 = manager.check_all_dependencies(use_cache=False)
        time_no_cache = time.time() - start_time
        
        # キャッシュありでチェック（2回目）
        print("Second check (with cache)...")
        start_time = time.time()
        result2 = manager.check_all_dependencies(use_cache=True)
        time_with_cache = time.time() - start_time
        
        print(f"\nPerformance comparison:")
        print(f"  - Without cache: {time_no_cache:.3f} seconds")
        print(f"  - With cache: {time_with_cache:.3f} seconds")
        
        if time_with_cache < time_no_cache:
            speedup = time_no_cache / time_with_cache
            print(f"✅ Cache improved performance by {speedup:.1f}x")
        else:
            print("ℹ️ Cache did not improve performance (might be first run)")
        
        # キャッシュクリアテスト
        manager._clear_cache()
        print("✅ Cache cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache functionality test failed: {e}")
        return False


def test_version_comparison():
    """バージョン比較のテスト"""
    print("\n" + "=" * 60)
    print("5. Testing Version Comparison")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        test_cases = [
            ("2.0.0", "1.0.0", 1, "2.0.0 > 1.0.0"),
            ("1.0.0", "2.0.0", -1, "1.0.0 < 2.0.0"),
            ("1.0.0", "1.0.0", 0, "1.0.0 == 1.0.0"),
            ("1.2.3", "1.2.0", 1, "1.2.3 > 1.2.0"),
            ("2.0.1", "2.0.0", 1, "2.0.1 > 2.0.0"),
        ]
        
        all_passed = True
        for v1, v2, expected, description in test_cases:
            result = manager._compare_versions(v1, v2)
            if result == expected:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} (got {result}, expected {expected})")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Version comparison test failed: {e}")
        return False


def test_alternative_packages():
    """代替パッケージのテスト"""
    print("\n" + "=" * 60)
    print("6. Testing Alternative Packages")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        # Qdrantの代替パッケージを確認
        qdrant_dep = manager.dependencies.get("qdrant")
        if qdrant_dep and qdrant_dep.alternatives:
            print(f"Qdrant alternatives: {qdrant_dep.alternatives}")
            
            # 代替パッケージの存在チェック
            for alt in qdrant_dep.alternatives:
                exists = manager._check_package(alt)
                status = "✅ Available" if exists else "❌ Not available"
                print(f"  - {alt}: {status}")
        
        # PDF処理の代替パッケージを確認
        pdf_dep = manager.dependencies.get("pdf_processor")
        if pdf_dep and pdf_dep.alternatives:
            print(f"\nPDF processor alternatives: {pdf_dep.alternatives}")
            
            for alt in pdf_dep.alternatives:
                exists = manager._check_package(alt)
                status = "✅ Available" if exists else "❌ Not available"
                print(f"  - {alt}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alternative packages test failed: {e}")
        return False


def test_service_checks():
    """サービスチェックのテスト"""
    print("\n" + "=" * 60)
    print("7. Testing Service Checks")
    print("=" * 60)
    
    try:
        manager = RAGDependencyManager()
        
        # Qdrantサービスチェック
        print("Checking Qdrant service...")
        qdrant_running = manager._check_qdrant_service()
        
        if qdrant_running:
            print("✅ Qdrant service is running")
        else:
            print("ℹ️ Qdrant service is not running (this is OK for testing)")
        
        # spaCyモデルチェック
        print("\nChecking spaCy Japanese model...")
        spacy_model = manager._check_spacy_model()
        
        if spacy_model:
            print("✅ spaCy Japanese model is installed")
        else:
            print("ℹ️ spaCy Japanese model is not installed (this is OK for testing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Service checks test failed: {e}")
        return False


def main():
    """メイン処理"""
    print("🔍 RAG Dependency Manager Integration Test")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    # テスト実行
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Dependency Check", test_dependency_check),
        ("Report Generation", test_report_generation),
        ("Cache Functionality", test_cache_functionality),
        ("Version Comparison", test_version_comparison),
        ("Alternative Packages", test_alternative_packages),
        ("Service Checks", test_service_checks),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            failed += 1
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        sys.exit(1)
