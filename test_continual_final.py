#!/usr/bin/env python3
"""
継続学習の最終テスト - メモリ最適化とデータ処理の修正確認
"""

import requests
import time
import json

def test_continual_learning():
    base_url = "http://localhost:8050"
    
    # 1. 継続学習タスクを開始
    print("=" * 50)
    print("継続学習テスト（4bit量子化 + データ処理修正）")
    print("=" * 50)
    
    config = {
        "base_model": "outputs/フルファインチューニング_20250819_111844",
        "task_name": "test_final_fix",
        "use_previous_tasks": True,
        "ewc_lambda": 5000,
        "epochs": 1,  # テスト用に1エポック
        "learning_rate": 2e-5,
        "use_memory_efficient": True  # メモリ効率化有効
    }
    
    print(f"設定: {json.dumps(config, ensure_ascii=False, indent=2)}")
    print("\n継続学習を開始...")
    
    # APIエンドポイントを直接指定
    response = requests.post(
        f"{base_url}/api/continual/train",
        json={"config": config}
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"エラー: {response.text}")
        return False
    
    result = response.json()
    task_id = result.get("task_id")
    print(f"タスクID: {task_id}")
    print(f"メッセージ: {result.get('message')}")
    
    # 2. タスクのステータスをモニタリング
    print("\nステータスをモニタリング中...")
    print("-" * 50)
    
    max_wait = 300  # 最大5分待機
    start_time = time.time()
    last_message = ""
    
    while time.time() - start_time < max_wait:
        try:
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
                    status = current_task.get("status", "unknown")
                    message = current_task.get("message", "")
                    progress = current_task.get("progress", 0)
                    
                    # メッセージが変わった場合のみ出力
                    if message != last_message:
                        print(f"[{time.strftime('%H:%M:%S')}] {status.upper()} | {progress:.0f}% | {message}")
                        last_message = message
                    
                    if status == "completed":
                        print("\n" + "=" * 50)
                        print("✅ 継続学習が正常に完了しました！")
                        print("=" * 50)
                        
                        # 成功の詳細を表示
                        if current_task.get("output_path"):
                            print(f"出力パス: {current_task.get('output_path')}")
                        if current_task.get("completed_at"):
                            print(f"完了時刻: {current_task.get('completed_at')}")
                        
                        return True
                        
                    elif status == "failed":
                        print("\n" + "=" * 50)
                        print("❌ 継続学習が失敗しました")
                        print("=" * 50)
                        
                        error = current_task.get("error", "不明なエラー")
                        print(f"エラー: {error}")
                        
                        # エラーの詳細分析
                        if "CUDA" in error or "memory" in error:
                            print("\n💡 メモリエラーが発生しています。以下を確認してください：")
                            print("  1. GPUメモリが十分に空いているか")
                            print("  2. 4bit量子化が正しく適用されているか")
                            print("  3. バッチサイズが適切か")
                        elif "dict" in error or "PathLike" in error:
                            print("\n💡 データ処理エラーが発生しています。training_dataの形式を確認してください。")
                        
                        return False
        except Exception as e:
            print(f"モニタリング中のエラー: {e}")
        
        time.sleep(3)
    
    print("\n⏱️ タイムアウト: タスクが時間内に完了しませんでした")
    return False

if __name__ == "__main__":
    success = test_continual_learning()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 すべての修正が正常に動作しています！")
        print("継続学習でメモリ最適化とデータ処理が正しく機能しました。")
    else:
        print("⚠️ まだ問題が残っています。ログを確認してください。")
    print("=" * 50)