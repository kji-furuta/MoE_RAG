#!/usr/bin/env python3
"""
LoRAを使用した継続学習テスト（メモリ効率的）
"""

import requests
import json
import time

# APIベースURL
BASE_URL = "http://localhost:8050"

# テスト用データセットの作成
test_data = [
    {"text": "道路設計速度は道路の幾何構造を決定する最も重要な要素です。設計速度が高いほど、より緩やかな曲線と勾配が必要になります。"},
    {"text": "横断勾配は道路の排水性能を確保するために設置されます。標準的には直線部で1.5〜2.0%の勾配を設けます。"},
    {"text": "縦断勾配の最大値は設計速度によって制限されます。設計速度60km/hでは最大勾配は7%となります。"}
]

# データをファイルに保存
with open("/tmp/lora_test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# LoRAを使用した継続学習設定
config = {
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "road_design_lora",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 5e-5,  # LoRA用に学習率を上げる
    "use_memory_efficient": True,
    "use_previous_tasks": False,
    "ewc_lambda": 1000.0,
    "use_lora": True,  # LoRAを使用
    "lora_r": 8,       # LoRAランク
    "lora_alpha": 16,  # LoRAアルファ
    "lora_dropout": 0.1
}

print("=" * 60)
print("LoRAを使用した継続学習テスト")
print("=" * 60)
print("設定:")
print(json.dumps(config, ensure_ascii=False, indent=2))

# タスク開始
print("\n1. タスクを開始...")
files = {
    'dataset': ('lora_test.jsonl', open('/tmp/lora_test.jsonl', 'rb'), 'application/json'),
}
data = {
    'config': json.dumps(config)
}

try:
    response = requests.post(
        f"{BASE_URL}/api/continual-learning/start",
        files=files,
        data=data,
        timeout=10
    )

    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"✅ タスク開始成功")
        print(f"   タスクID: {task_id}")
        print(f"   メッセージ: {result.get('message')}")

        # タスク状態を定期的に確認
        print("\n2. タスク進捗を監視...")
        for i in range(6):  # 30秒間監視
            time.sleep(5)

            status_response = requests.get(
                f"{BASE_URL}/api/continual-learning/status/{task_id}"
            )

            if status_response.status_code == 200:
                status = status_response.json()
                print(f"\n[{i*5+5}秒後]")
                print(f"  状態: {status.get('status')}")
                print(f"  進捗: {status.get('progress')}%")

                if status.get('status') == 'completed':
                    print("✅ タスク完了！")
                    if 'metrics' in status:
                        print(f"  メトリクス: {status['metrics']}")
                    break
                elif status.get('status') == 'failed':
                    print(f"❌ タスク失敗")
                    if 'error' in status:
                        print(f"  エラー: {status['error']}")
                    break
                else:
                    if 'messages' in status and status['messages']:
                        print("  最新メッセージ:")
                        for msg in status['messages'][-3:]:
                            print(f"    - {msg}")
            else:
                print(f"❌ 状態確認失敗: {status_response.status_code}")

    else:
        print(f"❌ タスク開始失敗: {response.status_code}")
        print(f"レスポンス: {response.text[:200]}")

except requests.exceptions.Timeout:
    print("⏱️ タイムアウト（バックグラウンドで処理中）")
    print("タスク一覧を確認してください: http://localhost:8050/api/continual-learning/tasks")
except Exception as e:
    print(f"❌ エラー: {e}")

print("\n" + "=" * 60)
print("テスト完了")
print("\n💡 ヒント:")
print("• タスク一覧: curl http://localhost:8050/api/continual-learning/tasks")
print("• モデル一覧: curl http://localhost:8050/api/continual-learning/models")
print("• Web UI: http://localhost:8050/continual")
print("=" * 60)