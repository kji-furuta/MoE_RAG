#!/usr/bin/env python3
"""
LoRA to Ollama変換機能のテストスクリプト
"""

import requests
import json
import time
import sys

def test_lora_to_ollama():
    """LoRA to Ollama API機能をテスト"""
    
    base_url = "http://localhost:8050"
    
    print("=" * 60)
    print("LoRA to Ollama Conversion Test")
    print("=" * 60)
    
    # テストデータ
    test_data = {
        "base_model_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "base_model_name": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "output_model_name": "test-deepseek-32b-finetuned"
    }
    
    print("\n1. Starting LoRA to Ollama conversion...")
    print(f"   Base Model: {test_data['base_model_name']}")
    print(f"   Output Name: {test_data['output_model_name']}")
    
    try:
        # APIリクエスト送信
        response = requests.post(
            f"{base_url}/api/apply-lora-to-ollama",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"\n✅ Conversion started successfully!")
            print(f"   Task ID: {task_id}")
            print(f"   Message: {result.get('message')}")
            
            # ステータスチェック
            print("\n2. Checking conversion status...")
            max_attempts = 60  # 最大2分待つ
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(2)
                status_response = requests.get(f"{base_url}/api/training-status/{task_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    print(f"   [{attempt*2}s] Status: {status.get('status')} - {status.get('message')}")
                    
                    if status.get('status') == 'completed':
                        print(f"\n✅ Conversion completed successfully!")
                        if status.get('model_name'):
                            print(f"   Model Name: {status.get('model_name')}")
                            print(f"\n   To use the model:")
                            print(f"   ollama run {status.get('model_name')} \"Your question here\"")
                        break
                        
                    elif status.get('status') == 'error':
                        print(f"\n❌ Conversion failed!")
                        print(f"   Error: {status.get('message')}")
                        break
                        
                attempt += 1
                
            if attempt >= max_attempts:
                print("\n⚠️ Timeout: Conversion is taking longer than expected")
                
        else:
            print(f"\n❌ Failed to start conversion")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Cannot connect to the server")
        print("   Please ensure the server is running on port 8050")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    test_lora_to_ollama()