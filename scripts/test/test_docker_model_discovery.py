#!/usr/bin/env python3
"""
Docker環境でのモデル探索テスト
WSL2 + Dockerコンテナ環境でモデルリストが正しく取得できるかテスト
"""

import os
import sys
import json
import requests
from pathlib import Path

def test_model_listing():
    """モデルリストAPIをテスト"""
    
    print("="*60)
    print("Docker環境モデル探索テスト")
    print("="*60)
    
    # 環境確認
    print("\n[環境確認]")
    print(f"現在のディレクトリ: {os.getcwd()}")
    print(f"Python実行パス: {sys.executable}")
    
    # outputsディレクトリの確認
    outputs_paths = [
        "/workspace/outputs",  # Dockerコンテナ内
        "./outputs",           # 相対パス
        "/home/kjifu/MoE_RAG/outputs"  # WSL2ホスト
    ]
    
    print("\n[outputsディレクトリ検索]")
    for path in outputs_paths:
        p = Path(path)
        if p.exists():
            print(f"✅ 存在: {path}")
            # ディレクトリ内容を確認
            try:
                items = list(p.iterdir())[:5]  # 最初の5個
                if items:
                    print(f"   内容: {[item.name for item in items]}")
                else:
                    print(f"   空のディレクトリ")
            except PermissionError:
                print(f"   アクセス権限なし")
        else:
            print(f"❌ 存在しない: {path}")
    
    # APIテスト
    print("\n[APIテスト]")
    api_url = "http://localhost:8050/rag/list-lora-models"
    
    try:
        print(f"API呼び出し: {api_url}")
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ APIレスポンス成功")
            print(f"   モデル数: {data.get('count', 0)}")
            
            # モデル詳細を表示
            if data.get('models'):
                print("\n[検出されたモデル]")
                for i, model in enumerate(data['models'][:5], 1):  # 最初の5個
                    print(f"{i}. {model.get('display_name', model.get('name', 'Unknown'))}")
                    print(f"   タイプ: {model.get('type', 'Unknown')}")
                    print(f"   パス: {model.get('path', 'Unknown')}")
                    print(f"   処理要否: {model.get('needs_processing', 'Unknown')}")
                    print()
            else:
                print("⚠️ モデルが見つかりません")
            
            # サマリー表示
            if data.get('summary'):
                print("\n[サマリー]")
                summary = data['summary']
                print(f"総数: {summary.get('total', 0)}")
                print(f"使用可能: {summary.get('ready_to_use', 0)}")
                print(f"LoRAアダプター: {summary.get('lora_adapters', 0)}")
                print(f"マージ済み: {summary.get('merged_models', 0)}")
                print(f"GGUF: {summary.get('gguf_models', 0)}")
        else:
            print(f"❌ APIエラー: ステータスコード {response.status_code}")
            print(f"   エラー内容: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 接続エラー: APIサーバーが起動していません")
        print("   以下のコマンドでサーバーを起動してください:")
        print("   docker exec ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050")
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    # 直接Pythonモジュールをテスト（Dockerコンテナ内で実行時）
    print("\n[Pythonモジュール直接テスト]")
    try:
        sys.path.insert(0, '/workspace')
        from src.utils.model_discovery import ModelDiscovery
        
        discovery = ModelDiscovery()
        print(f"✅ ModelDiscoveryインポート成功")
        print(f"   探索ディレクトリ: {discovery.outputs_dir}")
        
        # モデル探索実行
        all_models = discovery.find_all_models()
        
        for model_type, models in all_models.items():
            if models:
                print(f"   {model_type}: {len(models)} モデル")
                
    except ImportError as e:
        print(f"⚠️ ModelDiscoveryインポート失敗: {e}")
        print("   フォールバック方式が使用されます")
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("\n" + "="*60)
    print("テスト完了")
    print("="*60)

def create_dummy_lora_model():
    """テスト用のダミーLoRAモデルを作成"""
    
    print("\n[テスト用LoRAモデル作成]")
    
    outputs_dir = Path("/workspace/outputs")
    if not outputs_dir.exists():
        print(f"outputsディレクトリを作成: {outputs_dir}")
        outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # ダミーLoRAディレクトリ
    dummy_lora = outputs_dir / "lora_test_model"
    dummy_lora.mkdir(exist_ok=True)
    
    # adapter_config.json作成
    adapter_config = {
        "base_model_name_or_path": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    }
    
    with open(dummy_lora / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"✅ テスト用LoRAモデル作成: {dummy_lora}")

def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker環境モデル探索テスト")
    parser.add_argument("--create-dummy", action="store_true", 
                       help="テスト用ダミーモデルを作成")
    
    args = parser.parse_args()
    
    if args.create_dummy:
        create_dummy_lora_model()
    
    test_model_listing()

if __name__ == "__main__":
    main()