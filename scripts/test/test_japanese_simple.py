#!/usr/bin/env python3
"""
日本語モデルの簡易テスト
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_japanese_model():
    """日本語モデルの簡易テスト"""
    
    print("=== 日本語モデルテスト ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # テスト用の小さい日本語モデル
    model_name = "rinna/japanese-gpt2-small"  # 小さいモデルでテスト
    
    print(f"\nLoading model: {model_name}")
    
    try:
        # トークナイザーをロード
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # モデルをロード（メモリ節約のため8bit量子化）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        print("✓ Model loaded successfully")
        
        # パラメータ数を表示
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count / 1e6:.1f}M")
        
        # テキスト生成テスト
        test_prompts = [
            "こんにちは、今日は",
            "人工知能とは",
            "東京の天気は"
        ]
        
        print("\n=== Generation Test ===")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # デバイスを合わせる
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # デコード
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_japanese_model()