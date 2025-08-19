#!/usr/bin/env python3
"""
MoEトレーニング履歴の再起動後の永続性テスト
"""

import sys
import json
import requests
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

def test_moe_restart():
    """再起動後のMoE履歴確認"""
    
    print("=" * 60)
    print("MoE履歴再起動テスト")
    print("=" * 60)
    
    # 1. 保存された履歴ファイルの確認
    history_file = Path("data/moe_training/training_history.json")
    saved_tasks = {}
    
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            saved_tasks = json.load(f)
        
        print(f"\n✓ 保存された履歴ファイル:")
        print(f"  タスク数: {len(saved_tasks)}")
        for task_id in saved_tasks:
            task = saved_tasks[task_id]
            print(f"  - {task_id}: {task['status']} ({task['config']['training_type']})")
    
    # 2. API経由で履歴を確認
    print("\n" + "-" * 40)
    print("API経由の履歴確認:")
    
    try:
        response = requests.get("http://localhost:8050/api/moe/training/history")
        
        if response.status_code == 200:
            data = response.json()
            api_tasks = data.get('history', [])
            
            print(f"  APIから取得したタスク数: {len(api_tasks)}")
            
            # 保存されたタスクとAPIタスクの比較
            api_task_ids = {task['task_id'] for task in api_tasks}
            saved_task_ids = set(saved_tasks.keys())
            
            if api_task_ids == saved_task_ids:
                print("  ✓ すべてのタスクが正しく読み込まれています")
            else:
                missing = saved_task_ids - api_task_ids
                extra = api_task_ids - saved_task_ids
                
                if missing:
                    print(f"  ✗ 読み込まれなかったタスク: {missing}")
                if extra:
                    print(f"  ? APIのみに存在するタスク: {extra}")
                    
    except requests.exceptions.ConnectionError:
        print("  ✗ APIサーバーに接続できません")
        print("  サーバーを起動してください: docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh")
    
    # 3. RAGシステムでの表示確認
    print("\n" + "-" * 40)
    print("RAGシステムでのMoEモデル表示確認:")
    
    try:
        # MoE履歴をRAG UIで確認するためのヒント
        print("  ブラウザで以下を確認してください:")
        print("  1. http://localhost:8050/rag にアクセス")
        print("  2. 「RAGシステム設定」タブを開く")
        print("  3. 「使用するAIモデル」のドロップダウンを確認")
        print("  4. 「MoE（Mixture of Experts）モデル」セクションに以下が表示されるはず:")
        
        for task_id, task in saved_tasks.items():
            if task['status'] == 'completed':
                experts = task['config'].get('experts', [])
                print(f"     - MoE - {task['config']['training_type']} ({', '.join(experts) if experts else '全エキスパート'})")
                
    except Exception as e:
        print(f"  エラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print("\n再起動後も履歴が保持されることを確認するには:")
    print("1. サーバーを停止: Ctrl+C または docker restart ai-ft-container")
    print("2. サーバーを再起動: docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh")
    print("3. このスクリプトを再実行: python3 scripts/test_moe_restart.py")


if __name__ == "__main__":
    test_moe_restart()