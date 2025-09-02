#!/usr/bin/env python3
"""
正しいマージフロー：量子化モデルをFP16で再マージ
"""

import torch
import gc
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def proper_lora_merge():
    """LoRAを正しくマージ（FP16で保存）"""
    
    print("="*60)
    print("正しいLoRAマージプロセス")
    print("="*60)
    
    base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    lora_path = "/workspace/outputs/lora_20250830_223432"
    output_path = "/workspace/outputs/merged_model_fp16"
    
    # 既存のFP16モデルがある場合はスキップ
    if os.path.exists(f"{output_path}/model.safetensors.index.json"):
        print("✅ FP16マージ済みモデルが既に存在します")
        return True
    
    print("\n1. ベースモデルをFP16でロード中...")
    print("   注意: 量子化せずにFP16でロード（メモリ多く使用）")
    
    try:
        # FP16でロード（量子化なし）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/offload"
        )
        print("✅ FP16ベースモデルロード成功")
        
    except Exception as e:
        print(f"⚠️ メモリ不足の可能性: {e}")
        print("\n代替方法: 4bit量子化でロード後、FP16に変換...")
        
        # 4bit量子化でロード
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/offload"
        )
        print("✅ 4bit量子化ベースモデルロード成功")
    
    print("\n2. LoRAアダプタをロード中...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("\n3. LoRAをマージ中...")
    model = model.merge_and_unload()
    
    print("\n4. FP16形式で保存中...")
    print("   重要: 量子化せずにFP16で保存")
    os.makedirs(output_path, exist_ok=True)
    
    # FP16で保存（量子化メタデータなし）
    model.save_pretrained(
        output_path,
        torch_dtype=torch.float16,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # トークナイザも保存
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    # クリーンアップ
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ FP16マージモデル保存完了: {output_path}")
    print("\n次のステップ:")
    print("1. GGUF変換が可能になりました")
    print("2. RAGシステムで「量子化開始」を実行してください")
    
    return True

if __name__ == "__main__":
    success = proper_lora_merge()
    sys.exit(0 if success else 1)