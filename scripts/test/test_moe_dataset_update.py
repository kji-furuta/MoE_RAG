#!/usr/bin/env python3
"""
MoE Dataset Update Functionality Test Script
MoEデータセット更新機能のテストスクリプト
"""

import requests
import json
import sys
from pathlib import Path
import time

# API endpoint base URL
BASE_URL = "http://localhost:8050"

def test_dataset_stats():
    """データセット統計情報の取得テスト"""
    print("\n=== Testing Dataset Stats API ===")
    
    datasets = ["civil_engineering", "road_design"]
    
    for dataset in datasets:
        try:
            response = requests.get(f"{BASE_URL}/api/moe/dataset/stats/{dataset}")
            if response.status_code == 200:
                stats = response.json()
                print(f"\n✅ Dataset: {dataset}")
                print(f"   Samples: {stats.get('sample_count', 0)}")
                print(f"   Experts: {stats.get('expert_distribution', 'N/A')}")
                print(f"   Last Updated: {stats.get('last_updated', 'N/A')}")
                print(f"   File Size: {stats.get('file_size', 0)} bytes")
            else:
                print(f"❌ Failed to get stats for {dataset}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting stats for {dataset}: {str(e)}")

def test_dataset_update():
    """データセット更新テスト"""
    print("\n=== Testing Dataset Update API ===")
    
    # テスト用のサンプルデータを作成
    test_data = [
        {
            "expert_domain": "road_design",
            "question": "テスト質問1: 道路の設計速度について",
            "answer": "テスト回答1: 設計速度は道路構造令に基づいて決定されます。",
            "keywords": ["設計速度", "道路構造令"],
            "difficulty": "beginner"
        },
        {
            "expert_domain": "structural",
            "question": "テスト質問2: 橋梁の設計について",
            "answer": "テスト回答2: 橋梁設計は構造計算と耐震設計が重要です。",
            "keywords": ["橋梁", "構造計算"],
            "difficulty": "intermediate"
        }
    ]
    
    # JSONLファイルとして保存
    test_file_path = Path("test_dataset.jsonl")
    with open(test_file_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    try:
        # ファイルをアップロード
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_dataset.jsonl', f, 'application/x-jsonlines')}
            data = {'dataset_name': 'road_design'}
            
            response = requests.post(f"{BASE_URL}/api/moe/dataset/update", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Dataset update successful!")
                print(f"   Backup Path: {result.get('backup_path', 'None')}")
                print(f"   Sample Count: {result.get('sample_count', 0)}")
                print(f"   Validation: {result.get('validation_result', 'N/A')}")
            else:
                print(f"❌ Dataset update failed: {response.status_code}")
                print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error updating dataset: {str(e)}")
    
    finally:
        # クリーンアップ
        if test_file_path.exists():
            test_file_path.unlink()

def test_dataset_download():
    """データセットダウンロードテスト"""
    print("\n=== Testing Dataset Download API ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/moe/dataset/download/road_design")
        
        if response.status_code == 200:
            # ダウンロードしたデータの最初の数行を表示
            lines = response.text.strip().split('\n')[:3]
            print("✅ Dataset download successful!")
            print(f"   Total size: {len(response.content)} bytes")
            print(f"   First 3 lines:")
            for i, line in enumerate(lines, 1):
                if len(line) > 100:
                    print(f"   {i}. {line[:100]}...")
                else:
                    print(f"   {i}. {line}")
        else:
            print(f"❌ Dataset download failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")

def test_moe_training_apis():
    """MoEトレーニングAPIのテスト"""
    print("\n=== Testing MoE Training APIs ===")
    
    # トレーニング開始テスト
    training_config = {
        "training_type": "demo",
        "base_model": "cyberagent/open-calm-small",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0001,
        "warmup_steps": 10,
        "save_steps": 50,
        "dataset": "road_design",
        "experts": ["road_design", "structural"]
    }
    
    try:
        # トレーニング開始
        response = requests.post(f"{BASE_URL}/api/moe/training/start", json=training_config)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            print(f"✅ Training started with task_id: {task_id}")
            
            # ステータス確認
            time.sleep(2)
            status_response = requests.get(f"{BASE_URL}/api/moe/training/status/{task_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   Status: {status.get('status')}")
                print(f"   Progress: {status.get('progress', 0)}%")
            
            # GPU状態確認
            gpu_response = requests.get(f"{BASE_URL}/api/moe/training/gpu-status")
            if gpu_response.status_code == 200:
                gpu_info = gpu_response.json()
                if gpu_info.get('gpus'):
                    print(f"   GPU Available: {len(gpu_info['gpus'])} device(s)")
                else:
                    print("   GPU: Not available")
        else:
            print(f"❌ Training start failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error in training APIs: {str(e)}")

def main():
    """メインテスト実行"""
    print("=" * 50)
    print("MoE Dataset Update Functionality Test")
    print("=" * 50)
    
    # サーバーの確認
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("⚠️ Warning: Server may not be fully operational")
    except:
        print("❌ Error: Cannot connect to server at", BASE_URL)
        print("Please ensure the server is running: docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050")
        sys.exit(1)
    
    # 各テストを実行
    test_dataset_stats()
    test_dataset_update()
    test_dataset_download()
    test_moe_training_apis()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()