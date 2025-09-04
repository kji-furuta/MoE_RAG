#!/usr/bin/env python3
"""
LoRAアダプタをベースモデルにマージして継続学習用の完全なモデルを作成
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_to_base(lora_path: str, output_path: str):
    """
    LoRAアダプタをベースモデルにマージ
    
    Args:
        lora_path: LoRAアダプタのパス (例: outputs/lora_20250904_101907)
        output_path: マージ済みモデルの出力パス
    """
    
    # training_info.jsonからベースモデル情報を取得
    training_info_path = Path(lora_path) / "training_info.json"
    if not training_info_path.exists():
        raise FileNotFoundError(f"training_info.jsonが見つかりません: {training_info_path}")
    
    with open(training_info_path, 'r') as f:
        training_info = json.load(f)
    
    base_model_name = training_info.get("base_model")
    if not base_model_name:
        raise ValueError("training_info.jsonにbase_model情報がありません")
    
    logger.info(f"ベースモデル: {base_model_name}")
    logger.info(f"LoRAアダプタ: {lora_path}")
    
    # ベースモデルをロード（量子化なし、継続学習のため）
    logger.info("ベースモデルをロード中...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # LoRAアダプタをロードしてマージ
    logger.info("LoRAアダプタをロード中...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    logger.info("モデルをマージ中...")
    model = model.merge_and_unload()
    
    # マージ済みモデルを保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"マージ済みモデルを保存中: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # メタ情報を保存
    meta_info = {
        "original_base_model": base_model_name,
        "merged_lora_adapter": lora_path,
        "model_type": "merged_lora",
        "training_method": "continual_ready",
        "description": "LoRAアダプタをマージ済み。継続学習に使用可能"
    }
    
    with open(output_path / "merge_info.json", 'w') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ マージ完了！")
    logger.info(f"継続学習には以下のモデルパスを使用してください: {output_path}")
    
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRAアダプタをマージして継続学習用モデルを作成")
    parser.add_argument("--lora-path", type=str, required=True,
                       help="LoRAアダプタのパス (例: outputs/lora_20250904_101907)")
    parser.add_argument("--output-path", type=str, default=None,
                       help="出力パス（デフォルト: outputs/merged_[timestamp]）")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"outputs/merged_{timestamp}"
    
    try:
        merge_lora_to_base(args.lora_path, args.output_path)
    except Exception as e:
        logger.error(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()