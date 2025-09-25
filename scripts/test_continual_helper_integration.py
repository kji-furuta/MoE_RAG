#!/usr/bin/env python3
"""
継続学習ヘルパー統合テスト
"""

import requests
import json
import time
import sys
from pathlib import Path

# APIベースURL
BASE_URL = "http://localhost:8050"

# テスト用データセットの作成（道路工学に関連）
test_data = [
    {"text": "道路設計速度80km/hの最小曲線半径は280mです。この基準は安全性と快適性を考慮して設定されています。"},
    {"text": "横断勾配は排水性能を確保するために重要です。標準的な直線部では1.5〜2.0%の勾配を設けます。"},
    {"text": "縦断勾配の最大値は設計速度によって制限されます。設計速度60km/hでは最大勾配は7%となります。"},
    {"text": "道路幅員の決定では、車線幅、路肩幅、中央分離帯幅を考慮する必要があります。"},
    {"text": "インターチェンジのランプ設計では、本線からの減速と加速を考慮した適切な長さが必要です。"}
]

# データをファイルに保存
dataset_path = "/tmp/helper_integration_test.jsonl"
with open(dataset_path, "w", encoding="utf-8") as f:
    for item in test_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print("=" * 70)
print("継続学習ヘルパー統合テスト")
print("=" * 70)

# テスト1: 小規模モデル (7B) での継続学習
print("\n【テスト1: 小規模モデル (7B) での継続学習】")
print("-" * 50)

config_7b = {
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "road_design_7b_test",
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "use_memory_efficient": True,
    "use_previous_tasks": False,
    "ewc_lambda": 1000.0
}

print(f"設定: {json.dumps(config_7b, ensure_ascii=False, indent=2)}")

# タスク開始
print("\n1. 7Bモデルのタスクを開始...")
files = {
    'dataset': ('helper_integration_test.jsonl', open(dataset_path, 'rb'), 'application/json'),
}
data = {
    'config': json.dumps(config_7b)
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
        task_id_7b = result.get("task_id")
        print(f"✅ タスク開始成功")
        print(f"   タスクID: {task_id_7b}")
        print(f"   メッセージ: {result.get('message')}")

        # 進捗監視（30秒）
        print("\n2. タスク進捗を監視...")
        for i in range(6):
            time.sleep(5)

            status_response = requests.get(
                f"{BASE_URL}/api/continual-learning/status/{task_id_7b}"
            )

            if status_response.status_code == 200:
                status = status_response.json()
                print(f"\n[{i*5+5}秒後]")
                print(f"  状態: {status.get('status')}")
                print(f"  進捗: {status.get('progress')}%")

                if status.get('status') == 'completed':
                    print("✅ 7Bモデルのタスク完了！")
                    break
                elif status.get('status') == 'failed':
                    print(f"❌ タスク失敗: {status.get('error')}")
                    break
                else:
                    if 'messages' in status and status['messages']:
                        print("  最新メッセージ:")
                        for msg in status['messages'][-2:]:
                            print(f"    - {msg}")
            else:
                print(f"❌ 状態確認失敗: {status_response.status_code}")

    else:
        print(f"❌ タスク開始失敗: {response.status_code}")
        print(f"レスポンス: {response.text[:200]}")

except requests.exceptions.Timeout:
    print("⏱️ タイムアウト（バックグラウンドで処理中）")
except Exception as e:
    print(f"❌ エラー: {e}")

# テスト2: LoRA を使用した効率的な継続学習
print("\n" + "=" * 70)
print("【テスト2: LoRA を使用した効率的な継続学習】")
print("-" * 50)

config_lora = {
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "road_design_lora_test",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 5e-5,
    "use_memory_efficient": True,
    "use_previous_tasks": False,
    "ewc_lambda": 1000.0,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1
}

print(f"設定: {json.dumps(config_lora, ensure_ascii=False, indent=2)}")

# タスク開始
print("\n1. LoRAタスクを開始...")
files = {
    'dataset': ('helper_integration_test.jsonl', open(dataset_path, 'rb'), 'application/json'),
}
data = {
    'config': json.dumps(config_lora)
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
        task_id_lora = result.get("task_id")
        print(f"✅ LoRAタスク開始成功")
        print(f"   タスクID: {task_id_lora}")

        # 状態確認（10秒後）
        time.sleep(10)
        status_response = requests.get(
            f"{BASE_URL}/api/continual-learning/status/{task_id_lora}"
        )

        if status_response.status_code == 200:
            status = status_response.json()
            print(f"\n[10秒後の状態]")
            print(f"  状態: {status.get('status')}")
            print(f"  進捗: {status.get('progress')}%")
            if 'messages' in status and status['messages']:
                print("  メッセージ:")
                for msg in status['messages'][-3:]:
                    print(f"    - {msg}")

    else:
        print(f"❌ タスク開始失敗: {response.status_code}")

except Exception as e:
    print(f"❌ エラー: {e}")

# テスト3: システム情報の確認
print("\n" + "=" * 70)
print("【テスト3: システム情報の確認】")
print("-" * 50)

# タスク一覧
print("\n1. アクティブなタスク一覧:")
try:
    response = requests.get(f"{BASE_URL}/api/continual-learning/tasks")
    if response.status_code == 200:
        tasks = response.json()
        for task in tasks[-3:]:  # 最新3件を表示
            print(f"  - ID: {task['id'][:8]}...")
            print(f"    タイプ: {task['type']}")
            print(f"    状態: {task['status']}")
            print(f"    進捗: {task.get('progress', 0)}%")
            print()
    else:
        print(f"  取得失敗: {response.status_code}")
except Exception as e:
    print(f"  エラー: {e}")

# モデル一覧
print("\n2. 利用可能なモデル:")
try:
    response = requests.get(f"{BASE_URL}/api/continual-learning/models")
    if response.status_code == 200:
        models = response.json()
        for model in models[:5]:  # 最初の5件を表示
            print(f"  - {model['name']}")
            print(f"    タイプ: {model['type']}")
            if 'created_at' in model:
                print(f"    作成日時: {model['created_at']}")
            print()
    else:
        print(f"  取得失敗: {response.status_code}")
except Exception as e:
    print(f"  エラー: {e}")

# タスク履歴
print("\n3. タスク履歴:")
try:
    response = requests.get(f"{BASE_URL}/api/continual-learning/tasks/history")
    if response.status_code == 200:
        history = response.json()
        if history:
            for task in history[-3:]:  # 最新3件を表示
                print(f"  - タスク: {task.get('task_name', 'Unknown')}")
                print(f"    モデル: {task.get('model_path', 'Unknown')}")
                print(f"    作成日時: {task.get('created_at', 'Unknown')}")
                print()
        else:
            print("  履歴なし")
    else:
        print(f"  取得失敗: {response.status_code}")
except Exception as e:
    print(f"  エラー: {e}")

print("\n" + "=" * 70)
print("統合テスト完了")
print("=" * 70)
print("\n💡 確認事項:")
print("1. メモリアロケータが正しく設定されているか")
print("2. オフロードディレクトリが作成・削除されているか")
print("3. 量子化モデルの検出が機能しているか")
print("4. エラーハンドリングが適切に行われているか")
print("5. リソースのクリーンアップが実行されているか")
print("\n📊 次のステップ:")
print("• docker logs ai-ft-container --tail 100 | grep -E 'offload|memory|quantization'")
print("• ls -la /tmp/continual_offload_*  # オフロードディレクトリの確認")
print("• nvidia-smi  # GPU メモリ使用状況")
print("=" * 70)