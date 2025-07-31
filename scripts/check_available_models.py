#!/usr/bin/env python3
"""
利用可能なモデルをチェックするスクリプト
HuggingFaceキャッシュとローカルディレクトリを確認
"""

import os
import json
from pathlib import Path

def check_model_availability():
    """モデルの利用可能性をチェック"""
    
    # チェックするディレクトリ
    base_dir = Path("/home/kjifu/AI_finet")
    hf_cache = base_dir / "hf_cache" / "hub"
    local_models = base_dir / "models"
    
    available_models = []
    
    # 1. ローカルにダウンロード済みのモデル
    # CALM3-22B
    if (base_dir / "calm3-22b" / "model.safetensors.index.json").exists():
        available_models.append("cyberagent/calm3-22b-chat")
        print("✓ CALM3-22B - ローカルに完全ダウンロード済み")
    
    # 2. HuggingFaceキャッシュをチェック
    if hf_cache.exists():
        for model_dir in hf_cache.glob("models--*"):
            model_name = str(model_dir.name).replace("models--", "").replace("--", "/")
            
            # スナップショットディレクトリの存在確認
            snapshots = model_dir / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                available_models.append(model_name)
                print(f"✓ {model_name} - HFキャッシュに存在")
    
    # 3. 特定のモデルの詳細チェック
    specific_checks = {
        "Qwen/Qwen2.5-14B-Instruct": hf_cache / "models--Qwen--Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct": hf_cache / "models--Qwen--Qwen2.5-32B-Instruct",
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese": hf_cache / "models--cyberagent--DeepSeek-R1-Distill-Qwen-32B-Japanese"
    }
    
    print("\n=== 詳細チェック ===")
    for model_name, model_path in specific_checks.items():
        if model_path.exists():
            snapshots = model_path / "snapshots"
            if snapshots.exists():
                snapshot_dirs = list(snapshots.iterdir())
                if snapshot_dirs:
                    print(f"✓ {model_name}")
                    for snap in snapshot_dirs[:1]:  # 最初のスナップショットのみ表示
                        files = list(snap.glob("*.safetensors"))
                        if files:
                            print(f"  - スナップショット: {snap.name}")
                            print(f"  - モデルファイル数: {len(files)}")
    
    # 結果を保存
    result = {
        "available_models": sorted(list(set(available_models))),
        "total_available": len(set(available_models))
    }
    
    with open(base_dir / "available_models.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== サマリー ===")
    print(f"利用可能なモデル数: {result['total_available']}")
    print("\n利用可能なモデル:")
    for model in sorted(set(available_models)):
        print(f"  - {model}")
    
    return result

if __name__ == "__main__":
    check_model_availability()