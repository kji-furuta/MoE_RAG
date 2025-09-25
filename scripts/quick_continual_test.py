#!/usr/bin/env python3
"""
継続学習の簡易テスト
"""

import requests
import json
import time

# APIベースURL
BASE_URL = "http://localhost:8050"

# テスト用データセットの作成
test_data = [
    {"text": "道路設計の基本は安全性と効率性です。"},
    {"text": "設計速度は道路の幾何構造を決定する重要な要素です。"}
]

# データをファイルに保存
with open("/tmp/mini_test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# テスト設定（最小限）
config = {
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "mini_test",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "use_memory_efficient": True,
    "use_previous_tasks": False,  # 前のタスクを使わない
    "ewc_lambda": 1000.0
}

print("=" * 50)
print("継続学習簡易テスト")
print("=" * 50)
print(f"設定: {json.dumps(config, ensure_ascii=False, indent=2)}")

# タスク開始
print("\n1. タスクを開始...")
files = {
    'dataset': ('mini_test.jsonl', open('/tmp/mini_test.jsonl', 'rb'), 'application/json'),
}
data = {
    'config': json.dumps(config)
}

try:
    response = requests.post(
        f"{BASE_URL}/api/continual-learning/start",
        files=files,
        data=data,
        timeout=10  # 10秒でタイムアウト
    )

    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"✅ タスク開始成功: {task_id}")

        # タスク状態を確認（5秒待つ）
        time.sleep(5)

        print("\n2. タスク状態を確認...")
        status_response = requests.get(
            f"{BASE_URL}/api/continual-learning/status/{task_id}"
        )

        if status_response.status_code == 200:
            status = status_response.json()
            print(f"状態: {status.get('status')}")
            print(f"進捗: {status.get('progress')}%")
            if 'messages' in status:
                print("メッセージ:")
                for msg in status['messages'][-5:]:
                    print(f"  - {msg}")
        else:
            print(f"❌ 状態確認失敗: {status_response.status_code}")

    else:
        print(f"❌ タスク開始失敗: {response.status_code}")
        print(f"エラー: {response.text}")

except requests.exceptions.Timeout:
    print("⏱️ タイムアウト（バックグラウンドで処理中）")
except Exception as e:
    print(f"❌ エラー: {e}")

print("\n" + "=" * 50)
print("テスト完了")
print("タスク一覧: http://localhost:8050/api/continual-learning/tasks")
print("=" * 50)