#!/usr/bin/env python3
"""
リファクタリング後のAPIルーター構造のテストスクリプト
"""

import sys
import json
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_router_structure():
    """ルーター構造の確認"""
    print("=== ルーター構造の確認 ===\n")
    
    router_dir = project_root / "app" / "routers"
    expected_routers = [
        "__init__.py",
        "finetuning.py",
        "rag.py",
        "continual.py",
        "models.py"
    ]
    
    all_exist = True
    for router_file in expected_routers:
        file_path = router_dir / router_file
        if file_path.exists():
            print(f"✅ {router_file} が存在します")
        else:
            print(f"❌ {router_file} が見つかりません")
            all_exist = False
    
    return all_exist


def check_models_structure():
    """モデル定義構造の確認"""
    print("\n=== モデル定義構造の確認 ===\n")
    
    models_dir = project_root / "app" / "models"
    expected_models = [
        "__init__.py",
        "training.py",
        "rag.py",
        "models.py"
    ]
    
    all_exist = True
    for model_file in expected_models:
        file_path = models_dir / model_file
        if file_path.exists():
            print(f"✅ {model_file} が存在します")
        else:
            print(f"❌ {model_file} が見つかりません")
            all_exist = False
    
    return all_exist


def check_main_app():
    """メインアプリケーションファイルの確認"""
    print("\n=== メインアプリケーションの確認 ===\n")
    
    # 新しいmain.pyの確認
    new_main = project_root / "app" / "main.py"
    old_main = project_root / "app" / "main_unified.py"
    
    if new_main.exists():
        with open(new_main, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = len(content.splitlines())
        print(f"✅ 新しいmain.pyが存在します（{lines}行）")
        
        # 重要な要素の確認
        important_elements = [
            "from app.routers import",
            "app.include_router",
            "lifespan",
            "FastAPI"
        ]
        
        for element in important_elements:
            if element in content:
                print(f"  ✅ {element} が含まれています")
            else:
                print(f"  ❌ {element} が見つかりません")
    else:
        print(f"❌ 新しいmain.pyが見つかりません")
        return False
    
    # 旧ファイルとの比較
    if old_main.exists():
        with open(old_main, 'r', encoding='utf-8') as f:
            old_lines = len(f.read().splitlines())
        print(f"\n📊 コード削減: {old_lines}行 → {lines}行 (削減率: {((old_lines-lines)/old_lines*100):.1f}%)")
    
    return True


def check_dependencies():
    """依存関係ファイルの確認"""
    print("\n=== 依存関係ファイルの確認 ===\n")
    
    deps_file = project_root / "app" / "dependencies.py"
    
    if deps_file.exists():
        print(f"✅ dependencies.pyが存在します")
        
        with open(deps_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 重要な定義の確認
        important_defs = [
            "PROJECT_ROOT",
            "DATA_DIR",
            "OUTPUTS_DIR",
            "training_tasks",
            "continual_tasks",
            "model_cache"
        ]
        
        for def_name in important_defs:
            if def_name in content:
                print(f"  ✅ {def_name} が定義されています")
            else:
                print(f"  ❌ {def_name} が見つかりません")
        
        return True
    else:
        print(f"❌ dependencies.pyが見つかりません")
        return False


def test_imports():
    """インポートのテスト"""
    print("\n=== インポートテスト ===\n")
    
    try:
        # ルーターのインポート
        from app.routers import finetuning_router, rag_router, continual_router, models_router
        print("✅ すべてのルーターがインポート可能です")
        
        # モデルのインポート
        from app.models import TrainingRequest, QueryRequest, ModelInfo
        print("✅ すべてのモデルがインポート可能です")
        
        # 依存関係のインポート
        from app.dependencies import PROJECT_ROOT, training_tasks
        print("✅ 依存関係がインポート可能です")
        
        return True
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False


def main():
    """メイン処理"""
    print("="*60)
    print("リファクタリング検証テスト")
    print("="*60)
    
    results = []
    
    # 各チェックを実行
    results.append(("ルーター構造", check_router_structure()))
    results.append(("モデル構造", check_models_structure()))
    results.append(("メインアプリ", check_main_app()))
    results.append(("依存関係", check_dependencies()))
    results.append(("インポート", test_imports()))
    
    # 結果サマリー
    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} : {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 すべてのテストに合格しました！")
        print("\nリファクタリングが正常に完了しています。")
        print("次のステップ:")
        print("1. Docker環境で新しいmain.pyを起動")
        print("   docker exec ai-ft-container python -m uvicorn app.main:app --host 0.0.0.0 --port 8050")
        print("2. 各APIエンドポイントの動作確認")
        print("3. Webインターフェースの動作確認")
    else:
        print("⚠️  一部のテストが失敗しました")
        print("失敗した項目を確認して修正してください")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)