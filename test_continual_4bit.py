import requests
import time
import json

base_url = "http://localhost:8050"

# 継続学習を開始
print("継続学習タスクを開始（4bit量子化有効）...")
config = {
    "base_model": "outputs/フルファインチューニング_20250819_111844",
    "task_name": "test_4bit_optimized",
    "use_previous_tasks": True,
    "ewc_lambda": 5000,
    "epochs": 1,
    "learning_rate": 2e-5,
    "use_memory_efficient": True
}

response = requests.post(
    f"{base_url}/api/continual/train",
    json={"config": config}
)

print(f"Response status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    result = response.json()
    task_id = result["task_id"]
    print(f"タスクID: {task_id}")
    
    # ステータスを確認
    for i in range(60):
        time.sleep(5)
        response = requests.get(f"{base_url}/api/continual-learning/tasks")
        if response.status_code == 200:
            tasks = response.json()
            for task in tasks:
                if task.get("task_id") == task_id:
                    print(f"Status: {task.get('status')} | Progress: {task.get('progress')}% | Message: {task.get('message')}")
                    if task.get('status') in ['completed', 'failed']:
                        if task.get('status') == 'failed':
                            print(f"Error: {task.get('error')}")
                        exit(0)
                    break