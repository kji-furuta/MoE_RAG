#!/usr/bin/env python3
"""
MoEトレーニング履歴永続化テストスクリプト
"""

import sys
import json
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

def test_moe_persistence():
    """MoE履歴永続化のテスト"""
    
    print("=" * 60)
    print("MoEトレーニング履歴永続化テスト")
    print("=" * 60)
    
    # 1. 履歴ファイルの確認
    history_file = Path("data/moe_training/training_history.json")
    
    if history_file.exists():
        print(f"\n✓ 履歴ファイルが存在します: {history_file}")
        
        # 履歴ファイルの内容を読み込み
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        print(f"  保存されているタスク数: {len(history_data)}")
        
        for task_id, task_data in history_data.items():
            print(f"\n  タスクID: {task_id}")
            print(f"    状態: {task_data.get('status', 'unknown')}")
            print(f"    タイプ: {task_data['config'].get('training_type', 'unknown')}")
            print(f"    エキスパート: {task_data['config'].get('experts', [])}")
            print(f"    ベースモデル: {task_data['config'].get('base_model', 'unknown')}")
            
    else:
        print(f"\n✗ 履歴ファイルが存在しません: {history_file}")
    
    # 2. APIエンドポイントのテスト
    print("\n" + "=" * 60)
    print("APIエンドポイントテスト")
    print("=" * 60)
    
    import requests
    
    try:
        # 履歴取得API
        response = requests.get("http://localhost:8050/api/moe/training/history")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API履歴取得成功")
            print(f"  履歴数: {data.get('total', 0)}")
            
            if data.get('history'):
                for task in data['history'][:3]:  # 最初の3件のみ表示
                    print(f"\n  タスク: {task['task_id']}")
                    print(f"    状態: {task['status']}")
                    print(f"    進捗: {task['progress']}%")
        else:
            print(f"\n✗ API履歴取得失敗: ステータスコード {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("\n✗ APIサーバーに接続できません（ポート8050）")
    except Exception as e:
        print(f"\n✗ APIテスト失敗: {e}")
    
    # 3. モジュールインポートのテスト
    print("\n" + "=" * 60)
    print("モジュールインポートテスト")
    print("=" * 60)
    
    try:
        from app.moe_training_endpoints import training_tasks, HISTORY_FILE_PATH
        
        print(f"\n✓ モジュールインポート成功")
        print(f"  メモリ内のタスク数: {len(training_tasks)}")
        print(f"  履歴ファイルパス: {HISTORY_FILE_PATH}")
        
        # メモリ内のタスクを表示
        for task_id, task in list(training_tasks.items())[:3]:
            print(f"\n  メモリ内タスク: {task_id}")
            print(f"    状態: {task.status}")
            print(f"    設定: {task.config.training_type}")
            
    except ImportError as e:
        print(f"\n✗ モジュールインポート失敗: {e}")
    except Exception as e:
        print(f"\n✗ エラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    test_moe_persistence()