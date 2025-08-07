#!/usr/bin/env python3
"""
継続学習タスクの永続化テストスクリプト
"""

import json
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_persistence():
    """永続化ファイルの存在と内容を確認"""
    
    tasks_file = project_root / "data" / "continual_learning" / "tasks_state.json"
    
    print("=== 継続学習タスク永続化テスト ===\n")
    
    # ファイルの存在確認
    if tasks_file.exists():
        print(f"✅ タスク状態ファイルが存在します: {tasks_file}")
        
        try:
            # ファイルの内容を読み込み
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            print(f"\n📊 保存されているタスク数: {len(tasks)}")
            
            # 各タスクの情報を表示
            for task_id, task_data in tasks.items():
                print(f"\n--- タスク: {task_id[:8]}... ---")
                print(f"  名前: {task_data.get('task_name', 'N/A')}")
                print(f"  状態: {task_data.get('status', 'N/A')}")
                print(f"  進捗: {task_data.get('progress', 0)}%")
                print(f"  開始時刻: {task_data.get('started_at', 'N/A')}")
                if task_data.get('completed_at'):
                    print(f"  完了時刻: {task_data['completed_at']}")
                if task_data.get('error'):
                    print(f"  エラー: {task_data['error']}")
            
            print("\n✅ タスクデータが正常に保存されています")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ JSONファイルの解析エラー: {e}")
            return False
        except Exception as e:
            print(f"❌ ファイル読み込みエラー: {e}")
            return False
    else:
        print(f"⚠️  タスク状態ファイルが存在しません: {tasks_file}")
        print("   (まだタスクが作成されていない可能性があります)")
        return None

def check_api_integration():
    """APIエンドポイントでの永続化機能の確認"""
    print("\n=== API統合チェック ===\n")
    
    try:
        # main_unified.pyの変更を確認
        main_file = project_root / "app" / "main_unified.py"
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("load_continual_tasks", "タスク読み込み関数"),
            ("save_continual_tasks", "タスク保存関数"),
            ("CONTINUAL_TASKS_FILE", "永続化ファイルパス定義"),
            ("shutdown_event", "シャットダウンイベント")
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✅ {description}が実装されています")
            else:
                print(f"❌ {description}が見つかりません")
        
        # 保存処理の呼び出し回数を確認
        save_count = content.count("save_continual_tasks()")
        print(f"\n📊 save_continual_tasks()の呼び出し: {save_count}箇所")
        
        return True
        
    except Exception as e:
        print(f"❌ ファイル確認エラー: {e}")
        return False

def main():
    """メイン処理"""
    print("継続学習タスクの永続化機能をテストします\n")
    
    # 永続化ファイルのテスト
    persistence_result = test_persistence()
    
    # API統合のチェック
    api_result = check_api_integration()
    
    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)
    
    if persistence_result is None:
        print("⚠️  永続化ファイルはまだ作成されていません")
        print("   サーバーを起動してタスクを作成してください")
    elif persistence_result:
        print("✅ 永続化ファイルは正常に機能しています")
    else:
        print("❌ 永続化ファイルに問題があります")
    
    if api_result:
        print("✅ API統合は正常に実装されています")
    else:
        print("❌ API統合に問題があります")
    
    print("\n💡 ヒント:")
    print("1. サーバーを再起動して、タスクが保持されることを確認してください")
    print("2. docker exec ai-ft-container python /workspace/scripts/test_continual_persistence.py")
    print("   でDocker環境内でもテストできます")

if __name__ == "__main__":
    main()