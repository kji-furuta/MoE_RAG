#!/usr/bin/env python3
"""
LoRAモデル動的読み込みとUIの統合テスト
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8050"

def test_lora_models_endpoint():
    """LoRAモデルリストAPIのテスト"""
    print("=" * 60)
    print("1. Testing LoRA models listing endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/rag/list-lora-models")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Found {data['count']} LoRA models:")
        
        for model in data['models']:
            status = "⭐ RECOMMENDED" if model['recommended'] else ""
            print(f"  - {model['name']}: {model['base_model']} ({model['size_mb']:.2f} MB) {status}")
        
        return data['models']
    except Exception as e:
        print(f"❌ Failed to list LoRA models: {e}")
        return []

def test_quantization_start(lora_path: str, model_name: str = "test-model"):
    """量子化開始APIのテスト"""
    print("\n" + "=" * 60)
    print("2. Testing quantization start endpoint...")
    
    # Form dataとして送信
    form_data = {
        "lora_path": lora_path,
        "quantization_level": "Q4_K_M",
        "model_name": model_name
    }
    
    print(f"  Form data: {json.dumps(form_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/rag/quantize-model",
            data=form_data  # jsonではなくdataを使用
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Quantization started: Task ID = {data['task_id']}")
        return data['task_id']
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Failed to start quantization: {e}")
        return None

def test_quantization_status(task_id: str):
    """量子化ステータスAPIのテスト"""
    print("\n" + "=" * 60)
    print("3. Testing quantization status endpoint...")
    
    try:
        for _ in range(5):  # 5秒間だけステータスをチェック
            response = requests.get(f"{BASE_URL}/rag/quantization-status/{task_id}")
            response.raise_for_status()
            
            data = response.json()
            print(f"  Status: {data['status']}")
            
            if 'progress' in data:
                print(f"  Progress: {data['progress']}")
            if 'message' in data:
                print(f"  Message: {data['message']}")
            
            if data['status'] in ['completed', 'failed']:
                if data['status'] == 'completed':
                    print(f"✅ Quantization completed successfully!")
                else:
                    print(f"❌ Quantization failed: {data.get('error', 'Unknown error')}")
                break
            
            time.sleep(1)
        
        return data
    except Exception as e:
        print(f"❌ Failed to check status: {e}")
        return None

def test_ui_elements():
    """UI要素の存在確認"""
    print("\n" + "=" * 60)
    print("4. Testing UI elements presence...")
    
    try:
        response = requests.get(f"{BASE_URL}/rag")
        response.raise_for_status()
        
        html = response.text
        
        # 必要な要素が存在するか確認
        elements = [
            ("LoRA model dropdown", 'id="loraModel"'),
            ("Quantization level dropdown", 'id="quantizationLevel"'),
            ("Model name input", 'id="modelName"'),
            ("Quantize button", 'onclick="startQuantization()"'),
            ("loadLoRAModels function", 'loadLoRAModels'),
            ("Progress div", 'id="quantizationProgress"'),
            ("Result div", 'id="quantizationResult"')
        ]
        
        all_found = True
        for name, search_str in elements:
            if search_str in html:
                print(f"  ✅ {name}: Found")
            else:
                print(f"  ❌ {name}: Not found")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"❌ Failed to test UI elements: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("LoRA Model Dynamic Loading and UI Integration Test")
    print("=" * 60)
    
    # 1. LoRAモデルリストをテスト
    models = test_lora_models_endpoint()
    
    # 2. UI要素の存在確認
    ui_ok = test_ui_elements()
    
    # 3. 量子化フローのテスト（モデルが存在する場合のみ）
    if models:
        # 推奨モデルまたは最初のモデルを使用
        test_model = next((m for m in models if m['recommended']), models[0])
        print(f"\n  Using model for test: {test_model['name']}")
        
        task_id = test_quantization_start(
            test_model['path'],
            f"test-{test_model['name'].split('_')[-1]}"
        )
        
        if task_id:
            test_quantization_status(task_id)
    else:
        print("\n⚠️  No LoRA models found, skipping quantization test")
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  - LoRA models found: {len(models)}")
    print(f"  - UI elements OK: {ui_ok}")
    print("=" * 60)

if __name__ == "__main__":
    main()