#!/usr/bin/env python3
"""
継続学習の修正テストスクリプト
"""

import json
import os
from pathlib import Path

def test_model_path_resolution():
    """モデルパスの解決をテスト"""
    
    # テストケース1: LoRAアダプタパス
    base_model_path = "outputs/lora_20250904_101907"
    
    print(f"入力パス: {base_model_path}")
    print(f"パスが存在するか: {os.path.exists(base_model_path)}")
    print(f"'/'が含まれるか: {'/' in base_model_path}")
    
    # ファインチューニング済みモデルの場合はベースモデル情報を取得
    actual_base_model = base_model_path
    if os.path.exists(base_model_path):
        print(f"ファインチューニング済みモデルのパス: {base_model_path}")
        
        # training_info.jsonからベースモデル情報を取得
        training_info_path = Path(base_model_path) / "training_info.json"
        if training_info_path.exists():
            try:
                with open(training_info_path, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                    actual_base_model = training_info.get("base_model", base_model_path)
                    print(f"✅ ベースモデル情報を取得: {actual_base_model}")
                    
                    # LoRAアダプタ情報も表示
                    print(f"  - training_method: {training_info.get('training_method')}")
                    print(f"  - LoRA r: {training_info.get('r')}")
                    print(f"  - LoRA alpha: {training_info.get('lora_alpha')}")
            except Exception as e:
                print(f"❌ training_info.jsonの読み込みに失敗: {e}")
                actual_base_model = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
        else:
            print(f"❌ training_info.jsonが見つかりません")
            actual_base_model = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    
    print(f"\n最終的なベースモデル: {actual_base_model}")
    print(f"既存LoRAアダプタパス: {base_model_path if os.path.exists(base_model_path) else None}")
    
    return actual_base_model

if __name__ == "__main__":
    print("=== 継続学習のモデルパス解決テスト ===\n")
    
    # 実際のパスでテスト
    if os.path.exists("outputs/lora_20250904_101907"):
        resolved_model = test_model_path_resolution()
        
        print("\n=== 推奨される継続学習アプローチ ===")
        print("1. ベースモデルをロード:", resolved_model)
        print("2. 既存LoRAアダプタを適用: outputs/lora_20250904_101907")
        print("3. 新しいタスクで継続学習")
        print("4. 結果を新しいLoRAアダプタとして保存")
    else:
        print("❌ テスト用のLoRAアダプタが見つかりません")