#!/usr/bin/env python3
"""
LoRAモデルドロップダウンテスト
APIが正しくファインチューニング済みモデルを返すか確認
"""

import json
import requests
from pathlib import Path

def test_api_response():
    """APIレスポンスをテスト"""
    
    print("="*60)
    print("LoRAモデルドロップダウンAPIテスト")
    print("="*60)
    
    # APIエンドポイント
    url = "http://localhost:8050/rag/list-lora-models"
    
    try:
        print(f"\nAPI呼び出し: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ APIレスポンス成功")
            print(f"総モデル数: {data.get('count', 0)}")
            
            # サマリー情報
            if data.get('summary'):
                print("\n[サマリー]")
                summary = data['summary']
                print(f"  総数: {summary.get('total', 0)}")
                print(f"  使用可能: {summary.get('ready_to_use', 0)}")
                print(f"  LoRAアダプター: {summary.get('lora_adapters', 0)}")
                print(f"  マージ済み: {summary.get('merged_models', 0)}")
                print(f"  継続学習: {summary.get('continual_models', 0)}")
                print(f"  GGUF: {summary.get('gguf_models', 0)}")
            
            # モデル詳細
            if data.get('models'):
                print("\n[検出されたモデル]")
                
                # タイプごとにグループ化して表示
                model_by_type = {}
                for model in data['models']:
                    model_type = model.get('type', 'unknown')
                    if model_type not in model_by_type:
                        model_by_type[model_type] = []
                    model_by_type[model_type].append(model)
                
                type_labels = {
                    'ollama_ready': '✅ Ollama登録済み',
                    'gguf_model': '📦 GGUF形式',
                    'merged_model': '🔀 マージ済み',
                    'lora_adapter': '🎯 LoRAアダプター',
                    'continual_model': '📚 継続学習',
                    'auto': '🔍 自動検出'
                }
                
                for model_type, models in model_by_type.items():
                    print(f"\n{type_labels.get(model_type, model_type)}:")
                    for model in models[:3]:  # 各タイプ最大3個表示
                        print(f"  - {model.get('display_name', model.get('name', 'Unknown'))}")
                        print(f"    パス: {model.get('path', 'Unknown')}")
                        if model.get('base_model'):
                            print(f"    ベース: {model['base_model']}")
                        if model.get('needs_processing'):
                            print(f"    要処理: {model['needs_processing']}")
                        if model.get('recommended'):
                            print(f"    ⭐ 推奨")
            else:
                print("\n⚠️ モデルが見つかりません")
                print("  自動検出モードが使用されます")
            
            # JSONレスポンスをファイルに保存（デバッグ用）
            debug_file = Path("debug_lora_api_response.json")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\nデバッグ用JSONを保存: {debug_file}")
            
        else:
            print(f"\n❌ APIエラー: ステータスコード {response.status_code}")
            print(f"レスポンス: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 接続エラー")
        print("サーバーが起動していることを確認してください:")
        print("docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050")
    except Exception as e:
        print(f"\n❌ エラー: {e}")

def create_test_models():
    """テスト用のモデルを作成"""
    
    print("\n" + "="*60)
    print("テスト用モデル作成")
    print("="*60)
    
    outputs_dir = Path("/workspace/outputs")
    
    # テスト用LoRAモデル
    test_lora = outputs_dir / "lora_test_deepseek"
    test_lora.mkdir(parents=True, exist_ok=True)
    
    adapter_config = {
        "base_model_name_or_path": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
    
    with open(test_lora / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"✅ テストLoRAモデル作成: {test_lora}")
    
    # テスト用マージ済みモデル
    test_merged = outputs_dir / "merged_model"
    test_merged.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model_type": "qwen2",
        "hidden_size": 4096,
        "num_hidden_layers": 32
    }
    
    with open(test_merged / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ テストマージモデル作成: {test_merged}")
    
    # テスト用GGUFファイル
    test_gguf = outputs_dir / "test_model.gguf"
    test_gguf.touch()
    print(f"✅ テストGGUFファイル作成: {test_gguf}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-test":
        create_test_models()
    
    test_api_response()