#!/usr/bin/env python3
"""
ローカルPC（RTX A5000 x2）用のLoRAマージスクリプト
デュアルGPU環境で32Bモデルを効率的に処理
"""

import torch
import gc
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def clear_memory():
    """GPUメモリをクリア"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def merge_lora_dual_gpu():
    """デュアルGPU環境でLoRAをマージ"""
    
    # パス設定
    base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    lora_path = "/workspace/outputs/lora_20250830_223432"
    output_path = "/workspace/outputs/merged_model"
    
    print("=" * 60)
    print("ローカルPC デュアルGPU LoRAマージスクリプト")
    print("=" * 60)
    
    # GPU情報表示
    if torch.cuda.is_available():
        print(f"\n利用可能なGPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # メモリクリア
    clear_memory()
    
    # 4bit量子化設定（メモリ効率重視）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # デバイスマップを手動で設定（2GPU用）
    device_map = {
        # 最初の15レイヤーをGPU0に
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        # 残りのレイヤーをGPU1に
        "model.layers.15": 1,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 1,
        "model.layers.23": 1,
        "model.layers.24": 1,
        "model.layers.25": 1,
        "model.layers.26": 1,
        "model.layers.27": 1,
        "model.layers.28": 1,
        "model.layers.29": 1,
        "model.layers.30": 1,
        "model.layers.31": 1,
        "model.norm": 1,
        "lm_head": 1,
    }
    
    print("\n1. ベースモデルをロード中（4bit量子化、デュアルGPU）...")
    try:
        # オプション1: 自動デバイスマップ（推奨）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",  # 自動的に2つのGPUに分散
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "23GB", 1: "23GB"},  # 各GPUの最大メモリを指定
            offload_folder="/tmp/offload"  # 必要に応じてCPUオフロード
        )
        print("✅ 自動デバイスマップでロード成功")
        
    except Exception as e:
        print(f"⚠️ 自動マップ失敗: {e}")
        print("手動デバイスマップを試行中...")
        
        # オプション2: 手動デバイスマップ
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ 手動デバイスマップでロード成功")
    
    # メモリ使用状況表示
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: 割当 {allocated:.1f}GB / 予約 {reserved:.1f}GB")
    
    print("\n2. LoRAアダプタをロード中...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("\n3. LoRAをマージ中...")
    print("   （この処理には20-30分かかる場合があります）")
    
    # メモリ効率的なマージ
    try:
        model = model.merge_and_unload()
        print("✅ マージ完了")
    except Exception as e:
        print(f"⚠️ 標準マージ失敗: {e}")
        print("レイヤーごとのマージを試行中...")
        
        # レイヤーごとにマージ（メモリ効率的）
        from tqdm import tqdm
        for name, module in tqdm(model.named_modules(), desc="レイヤーマージ"):
            if hasattr(module, 'merge_and_unload'):
                module.merge_and_unload()
                clear_memory()
    
    print("\n4. マージ済みモデルを保存中...")
    print("   注意: 4bit量子化されたマージモデルを保存します")
    print("   FP16への変換はGGUF変換時に行われます")
    os.makedirs(output_path, exist_ok=True)
    
    # 4bit量子化のまま保存（dtype変換は不可）
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    print("\n5. トークナイザを保存中...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    # クリーンアップ
    del model, base_model
    clear_memory()
    
    print(f"\n✅ 完了！マージ済みモデルを保存しました: {output_path}")
    print("\n次のステップ:")
    print("1. GGUF変換: python scripts/convert_to_gguf.py")
    print("2. 量子化: python scripts/quantize_model_for_ollama.py")

if __name__ == "__main__":
    merge_lora_dual_gpu()