#!/usr/bin/env python3
"""
保存されたLoRAモデルの読み込みと利用例
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_and_use_trained_model():
    print("🔄 保存されたLoRAモデルの読み込みと利用")
    print("=" * 50)
    
    # 保存されたモデルのパス
    base_model_name = "distilgpt2"
    adapter_path = "/workspace/lora_demo_20250626_074248"
    
    print(f"📚 ベースモデル: {base_model_name}")
    print(f"📁 アダプターパス: {adapter_path}")
    
    try:
        # Step 1: ベースモデルとトークナイザーの読み込み
        print("\n🔧 Step 1: ベースモデル読み込み")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ ベースモデル読み込み完了")
        
        # Step 2: LoRAアダプターの読み込み
        print("\n🎯 Step 2: LoRAアダプター読み込み")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ LoRAアダプター読み込み完了")
        
        # Step 3: テスト生成
        print("\n🧪 Step 3: ファインチューニング済みモデルでテスト")
        
        test_inputs = [
            "Hello, how are",
            "The weather is",
            "I love machine",
            "RTX A5000 is",
            "Fine-tuning is"
        ]
        
        model.eval()
        
        for test_input in test_inputs:
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"入力: '{test_input}'")
            print(f"出力: '{generated_text}'")
            print("-" * 40)
        
        # Step 4: モデル情報表示
        print("\n📊 Step 4: モデル情報")
        model.print_trainable_parameters()
        
        # Step 5: インタラクティブテスト
        print("\n💬 Step 5: インタラクティブテスト")
        print("（'quit'で終了）")
        
        while True:
            try:
                user_input = input("\n入力テキスト: ")
                if user_input.lower() == 'quit':
                    break
                
                inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"生成: {generated_text}")
                
            except KeyboardInterrupt:
                break
        
        print("\n✅ モデル利用完了！")
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_base_vs_finetuned():
    """ベースモデルとファインチューニング済みモデルの比較"""
    print("\n🔍 ベースモデル vs ファインチューニング済みモデル比較")
    print("=" * 60)
    
    base_model_name = "distilgpt2"
    adapter_path = "/workspace/lora_demo_20250626_074248"
    
    try:
        # ベースモデル
        print("📚 ベースモデル読み込み...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # ファインチューニング済みモデル
        print("🎯 ファインチューニング済みモデル読み込み...")
        finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 比較テスト
        test_input = "Hello, how are"
        inputs = tokenizer(test_input, return_tensors="pt").to(base_model.device)
        
        print(f"\n🧪 比較テスト: '{test_input}'")
        print("-" * 40)
        
        # ベースモデルでの生成
        base_model.eval()
        with torch.no_grad():
            outputs = base_model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        base_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"ベースモデル: {base_text}")
        
        # ファインチューニング済みモデルでの生成
        finetuned_model.eval()
        with torch.no_grad():
            outputs = finetuned_model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        finetuned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"ファインチューニング済み: {finetuned_text}")
        
        print("\n✅ 比較完了")
        
    except Exception as e:
        print(f"❌ 比較エラー: {e}")

def show_model_files():
    """保存されたモデルファイルの詳細"""
    print("\n📁 保存されたモデルファイルの詳細")
    print("=" * 50)
    
    adapter_path = "/workspace/lora_demo_20250626_074248"
    
    try:
        import json
        
        # adapter_config.jsonの内容
        config_path = f"{adapter_path}/adapter_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print("⚙️ LoRA設定:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        
        # ファイルサイズ情報
        print("\n📊 ファイルサイズ:")
        import subprocess
        result = subprocess.run(['ls', '-lh', adapter_path], capture_output=True, text=True)
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ ファイル情報取得エラー: {e}")

def main():
    print("🎯 ファインチューニング済みモデルの活用ガイド")
    print("=" * 60)
    
    # モデルファイル情報
    show_model_files()
    
    # モデル読み込みと使用
    success = load_and_use_trained_model()
    
    if success:
        # ベースモデルとの比較
        compare_base_vs_finetuned()
    
    print("\n" + "=" * 60)
    print("💡 活用方法まとめ")
    print("=" * 60)
    print("1. 保存場所: /workspace/lora_demo_YYYYMMDD_HHMMSS/")
    print("2. 読み込み: PeftModel.from_pretrained(base_model, adapter_path)")
    print("3. 利用: 通常のTransformersモデルと同様")
    print("4. 配布: adapter_model.safetensorsのみ送付可能（約1.6MB）")
    print("5. 本番運用: APIサーバーやWebアプリに統合可能")

if __name__ == "__main__":
    main()