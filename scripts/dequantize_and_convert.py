#!/usr/bin/env python3
"""
量子化されたマージモデルを脱量子化してGGUF変換する
4bit量子化モデルをFP16に戻してから変換
"""

import os
import sys
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def dequantize_model():
    """量子化モデルを脱量子化してFP16で保存"""
    
    print("="*60)
    print("量子化モデルの脱量子化")
    print("="*60)
    
    input_path = "/workspace/outputs/merged_model"
    output_path = "/workspace/outputs/merged_model_fp16"
    
    # 既存のFP16モデルがあるか確認
    if os.path.exists(output_path):
        print(f"✅ FP16モデルが既に存在: {output_path}")
        return output_path
    
    print("\n1. 量子化済みモデルをロード中...")
    print("   注意: これは量子化された状態のモデルです")
    
    try:
        # 量子化モデルを通常のFP16としてロード（脱量子化）
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPUで処理
            trust_remote_code=True,
            ignore_mismatched_sizes=True  # サイズ不一致を無視
        )
        
        print("✅ モデルロード成功")
        
    except Exception as e:
        print(f"❌ 通常ロード失敗: {e}")
        print("\n代替方法: ベースモデルから再構築...")
        
        # ベースモデルとLoRAを再度マージ（FP16で）
        from peft import PeftModel
        
        base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
        lora_path = "/workspace/outputs/lora_20250830_223432"
        
        print("ベースモデルをFP16でロード中...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("LoRAアダプタをロード中...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.float16
        )
        
        print("マージ中...")
        model = model.merge_and_unload()
        
        # CPUに移動
        model = model.to("cpu")
    
    print("\n2. FP16形式で保存中...")
    os.makedirs(output_path, exist_ok=True)
    
    # FP16で保存（量子化なし）
    model.save_pretrained(
        output_path,
        torch_dtype=torch.float16,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # トークナイザも保存
    print("3. トークナイザを保存中...")
    tokenizer = AutoTokenizer.from_pretrained(
        input_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    # クリーンアップ
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ FP16モデル保存完了: {output_path}")
    return output_path

def convert_to_gguf(model_path):
    """FP16モデルをGGUF形式に変換"""
    
    print("\n" + "="*60)
    print("GGUF変換")
    print("="*60)
    
    gguf_path = "/workspace/outputs/model-f16.gguf"
    
    if os.path.exists(gguf_path):
        print(f"✅ GGUF既に存在: {gguf_path}")
        return gguf_path
    
    # llama.cppが存在するか確認
    if not os.path.exists("/workspace/llama.cpp"):
        print("❌ llama.cppが見つかりません")
        print("先にllama.cppをセットアップしてください")
        return None
    
    print("GGUF変換中...")
    import subprocess
    
    cmd = f"""cd /workspace/llama.cpp && python convert_hf_to_gguf.py \
        {model_path} \
        --outfile {gguf_path} \
        --outtype f16"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ GGUF変換失敗: {result.stderr}")
        return None
    
    print(f"✅ GGUF変換成功: {gguf_path}")
    return gguf_path

def main():
    """メイン処理"""
    
    print("量子化モデルの脱量子化とGGUF変換")
    print("="*60)
    
    # Step 1: 脱量子化
    fp16_model_path = dequantize_model()
    
    if not fp16_model_path:
        print("❌ 脱量子化に失敗しました")
        return 1
    
    # Step 2: GGUF変換
    gguf_path = convert_to_gguf(fp16_model_path)
    
    if not gguf_path:
        print("❌ GGUF変換に失敗しました")
        return 1
    
    print("\n" + "="*60)
    print("✅ 完了！")
    print(f"FP16モデル: {fp16_model_path}")
    print(f"GGUFファイル: {gguf_path}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())