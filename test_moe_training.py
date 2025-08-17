#!/usr/bin/env python3
"""
MoE Training API Test Script
トレーニングAPIのテストスクリプト
"""

import asyncio
import aiohttp
import json
import time

API_BASE = "http://localhost:8050"

async def test_gpu_status():
    """GPU状態確認テスト"""
    print("\n=== GPU Status Test ===")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE}/api/moe/training/gpu-status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✅ GPU Status Retrieved")
                    if "gpus" in data and data["gpus"]:
                        for gpu in data["gpus"]:
                            print(f"  GPU {gpu['id']}: {gpu['name']}")
                            print(f"    Memory: {gpu.get('memory_used', 'N/A')}MB / {gpu.get('memory_total', 'N/A')}MB")
                            print(f"    Load: {gpu.get('gpu_load', 'N/A')}%")
                    else:
                        print("  No GPU information available")
                else:
                    print(f"❌ Failed: Status {resp.status}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_start_training():
    """トレーニング開始テスト"""
    print("\n=== Start Training Test ===")
    
    training_config = {
        "training_type": "demo",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "warmup_steps": 50,
        "save_steps": 100,
        "dataset": "demo",
        "experts": ["structural", "road"]
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE}/api/moe/training/start",
                json=training_config
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Training Started: Task ID = {data['task_id']}")
                    return data['task_id']
                else:
                    print(f"❌ Failed: Status {resp.status}")
                    print(await resp.text())
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None

async def test_get_status(task_id):
    """トレーニングステータス確認テスト"""
    print(f"\n=== Get Training Status Test (Task: {task_id}) ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE}/api/moe/training/status/{task_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Status Retrieved")
                    print(f"  Status: {data['status']}")
                    print(f"  Progress: {data['progress']}%")
                    print(f"  Current Epoch: {data['current_epoch']}")
                    print(f"  Current Loss: {data['current_loss']}")
                    if data['logs']:
                        print(f"  Latest log: {data['logs'][-1] if data['logs'] else 'No logs'}")
                    return data
                else:
                    print(f"❌ Failed: Status {resp.status}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None

async def test_stop_training(task_id):
    """トレーニング停止テスト"""
    print(f"\n=== Stop Training Test (Task: {task_id}) ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{API_BASE}/api/moe/training/stop/{task_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Training Stopped: {data['message']}")
                else:
                    print(f"❌ Failed: Status {resp.status}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_get_history():
    """トレーニング履歴取得テスト"""
    print("\n=== Get Training History Test ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE}/api/moe/training/history?limit=5") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ History Retrieved: {data['total']} tasks")
                    for task in data['history'][:3]:  # 最初の3件のみ表示
                        print(f"  Task {task['task_id'][:8]}...")
                        print(f"    Status: {task['status']}")
                        print(f"    Type: {task['config']['training_type']}")
                        print(f"    Experts: {', '.join(task['config']['experts'])}")
                else:
                    print(f"❌ Failed: Status {resp.status}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """メインテスト実行"""
    print("=" * 50)
    print("MoE Training API Test")
    print("=" * 50)
    
    # GPU状態確認
    await test_gpu_status()
    
    # トレーニング履歴確認
    await test_get_history()
    
    # トレーニング開始
    task_id = await test_start_training()
    
    if task_id:
        # 少し待機
        await asyncio.sleep(3)
        
        # ステータス確認
        status = await test_get_status(task_id)
        
        # 進捗モニタリング（10秒間）
        print("\n=== Monitoring Progress (10 seconds) ===")
        for i in range(5):
            await asyncio.sleep(2)
            status = await test_get_status(task_id)
            if status:
                print(f"  [{i+1}/5] Progress: {status['progress']}%, Status: {status['status']}")
                if status['status'] in ['completed', 'failed']:
                    break
        
        # トレーニング停止
        if status and status['status'] == 'running':
            await test_stop_training(task_id)
    
    print("\n" + "=" * 50)
    print("Test Completed")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())