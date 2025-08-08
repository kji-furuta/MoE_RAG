#!/usr/bin/env python3
"""
RAG依存関係チェック・管理スクリプト

使用方法:
    python check_rag_dependencies.py --check
    python check_rag_dependencies.py --install
    python check_rag_dependencies.py --report markdown
"""

import sys
import os
from pathlib import Path

# プロジェクトのルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.dependencies.dependency_manager import (
    RAGDependencyManager,
    DependencyLevel
)


def main():
    """メイン処理"""
    import argparse
    
    # パーサーの設定
    parser = argparse.ArgumentParser(
        description="RAG System Dependency Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 依存関係をチェック
  %(prog)s --check
  
  # 不足している依存関係をインストール
  %(prog)s --install
  
  # コア依存関係のみインストール
  %(prog)s --install --level core
  
  # Markdown形式でレポート生成
  %(prog)s --report markdown > dependencies.md
  
  # JSON形式でレポート生成
  %(prog)s --report json > dependencies.json
  
  # ドライラン（実際にインストールせずにコマンドを表示）
  %(prog)s --install --dry-run
        """
    )
    
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check all dependencies"
    )
    
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Install missing dependencies"
    )
    
    parser.add_argument(
        "--report", "-r",
        choices=["text", "json", "markdown"],
        default="text",
        help="Generate dependency report in specified format"
    )
    
    parser.add_argument(
        "--level", "-l",
        choices=["core", "infrastructure", "optional", "all"],
        default="all",
        help="Dependency level to install (default: all)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be installed without actually installing"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached check results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # ログレベルの設定
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # マネージャーの初期化
    manager = RAGDependencyManager()
    
    # チェックまたはレポート生成
    if args.check or not args.install:
        print("Checking dependencies...\n")
        result = manager.check_all_dependencies(use_cache=not args.no_cache)
        
        # レポート生成
        report = manager.get_dependency_report(format=args.report)
        print(report)
        
        # 終了コード
        if not result.can_run:
            print("\n⚠️ System cannot run with current dependencies!")
            sys.exit(1)
        else:
            print("\n✅ System can run with current dependencies.")
            sys.exit(0)
    
    # インストール
    if args.install:
        # レベルの決定
        if args.level == "all":
            level = None
        else:
            level = DependencyLevel[args.level.upper()]
        
        print(f"Installing dependencies (level: {args.level})...\n")
        
        # 現在の状態を確認
        result = manager.check_all_dependencies(use_cache=False)
        
        if result.is_satisfied and args.level == "all":
            print("✅ All dependencies are already satisfied!")
            sys.exit(0)
        
        # インストール実行
        install_results = manager.install_missing_dependencies(
            level=level,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print("\n🔍 Dry run completed. No packages were actually installed.")
        else:
            # 結果表示
            print("\n" + "=" * 50)
            print("Installation Results:")
            print("=" * 50)
            
            success_count = 0
            fail_count = 0
            
            for package, success in install_results.items():
                if success is None:  # dry run
                    status = "⏭️"
                elif success:
                    status = "✅"
                    success_count += 1
                else:
                    status = "❌"
                    fail_count += 1
                print(f"  {status} {package}")
            
            if not args.dry_run:
                print(f"\nSummary: {success_count} succeeded, {fail_count} failed")
                
                # 最終チェック
                print("\nPerforming final check...")
                final_result = manager.check_all_dependencies(use_cache=False)
                
                if final_result.can_run:
                    print("✅ System is ready to run!")
                    sys.exit(0)
                else:
                    print("⚠️ Some critical dependencies are still missing.")
                    print("Please check the report above for details.")
                    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        sys.exit(1)
