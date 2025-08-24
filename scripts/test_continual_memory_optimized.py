#!/usr/bin/env python3
"""
継続学習のメモリ最適化テスト
"""

import requests
import time
import json

def test_continual_learning():
    """継続学習のメモリ最適化をテスト"""
    
    base_url = "http://localhost:8050"
    
    # 1. 継続学習を開始（メモリ効率化を有効化）
    print("継続学習タスクを開始（メモリ効率化有効）...")
    config = {
        "base_model": "outputs/フルファインチューニング_20250819_111844",
        "task_name": "test_memory_optimized",
        "use_previous_tasks": True,
        "ewc_lambda": 5000,
        "epochs": 1,  # テスト用に1エポックのみ
        "learning_rate": 2e-5,
        "use_memory_efficient": True  # メモリ効率化を有効化
    }
    
    response = requests.post(
        f"{base_url}/api/continual-learning/train",
        json={"config": config}
    )
    
    if response.status_code != 200:
        print(f"エラー: {response.text}")
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"タスクID: {task_id}")
    
    # 2. ステータスをモニタリング
    print("\nステータスをモニタリング中...")
    max_wait = 300  # 最大5分待機
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{base_url}/api/continual-learning/tasks")
        if response.status_code == 200:
            tasks = response.json()
            
            # 該当タスクを探す
            current_task = None
            for task in tasks:
                if task.get("task_id") == task_id:
                    current_task = task
                    break
            
            if current_task:
                status = current_task.get("status")
                message = current_task.get("message", "")
                progress = current_task.get("progress", 0)
                
                print(f"ステータス: {status} | 進捗: {progress:.1f}% | メッセージ: {message}")
                
                if status == "completed":
                    print("\n✅ 継続学習が正常に完了しました（メモリ最適化有効）")
                    return True
                elif status == "failed":
                    error = current_task.get("error", "不明なエラー")
                    print(f"\n❌ 継続学習が失敗しました: {error}")
                    return False
        
        time.sleep(5)
    
    print("\n⏱️ タイムアウト: タスクが時間内に完了しませんでした")
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("継続学習メモリ最適化テスト")
    print("=" * 50)
    
    success = test_continual_learning()
    
    if success:
        print("\n✅ テスト成功: メモリ最適化が正しく動作しています")
    else:
        print("\n❌ テスト失敗: ログを確認してください")