#!/usr/bin/env python3
"""
RTX A5000 x2 実践ファインチューニングガイド
動作確認済みのシンプルな手順
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from datetime import datetime

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = []
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            })
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]

def main():
    print("🚀 RTX A5000 x2 実践ファインチューニングガイド")
    print("=" * 60)
    
    # GPU確認
    print("🔧 GPU環境確認:")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ 利用可能GPU: {device_count}台")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        print("❌ CUDAが利用できません")
        return False
    
    print("\n📚 推奨実行手順:")
    print("=" * 40)
    
    print("\n🥇 **方法1: 軽量モデルで学習（推奨）**")
    print("   モデル: distilgpt2")
    print("   メモリ使用量: ~2GB")
    print("   実行時間: 5-10分")
    print("   実行コマンド:")
    print("   ```")
    print("   docker exec ai-ft-container python /workspace/run_distilgpt2_lora.py")
    print("   ```")
    
    print("\n🥈 **方法2: 日本語モデルで学習（中級）**")
    print("   モデル: stabilityai/japanese-stablelm-3b-4e1t-instruct")
    print("   メモリ使用量: ~8GB")
    print("   実行時間: 15-30分")
    print("   注意: HuggingFace認証が必要な場合があります")
    
    print("\n🥉 **方法3: 大規模モデル（上級）**")
    print("   モデル: 7B-13Bモデル")
    print("   メモリ使用量: 16-32GB")
    print("   手法: QLoRA推奨")
    print("   注意: Model Parallelism使用")
    
    print("\n" + "=" * 60)
    print("🎯 今すぐ試せる方法1の実装:")
    print("=" * 60)
    
    return demo_simple_lora()

def demo_simple_lora():
    """シンプルなLoRAデモ（distilgpt2使用）"""
    
    try:
        print("\n🚀 方法1: DistilGPT-2 LoRAファインチューニング開始")
        
        # モデルとトークナイザー
        model_name = "distilgpt2"
        print(f"📚 モデル読み込み: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ モデル読み込み完了")
        
        # LoRA設定
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # 小さめのランク
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # GPT-2のモジュール
        )
        
        model = get_peft_model(model, peft_config)
        print("⚙️ LoRA設定完了")
        model.print_trainable_parameters()
        
        # 訓練データ（英語で簡単に）
        train_texts = [
            "Hello, how are you today? I am doing well, thank you.",
            "What is the weather like? The weather is sunny and warm.",
            "Can you help me? Of course, I would be happy to help.",
            "Thank you very much. You are welcome, anytime.",
            "Good morning! Good morning, have a great day!"
        ]
        
        print(f"📝 訓練データ: {len(train_texts)}件")
        
        # データセット準備
        dataset = SimpleTextDataset(train_texts, tokenizer, max_length=128)
        
        # 訓練設定
        output_dir = f"./outputs/distilgpt2_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=1,
            save_steps=50,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # データコレクター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("🏋️ 訓練開始...")
        
        # 訓練前テスト
        print("\n--- 訓練前テスト ---")
        test_input = "Hello, how are you"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        before_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"入力: {test_input}")
        print(f"出力: {before_text}")
        
        # 訓練実行
        trainer.train()
        
        print("\n✅ 訓練完了！")
        
        # 訓練後テスト
        print("\n--- 訓練後テスト ---")
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        after_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"入力: {test_input}")
        print(f"出力: {after_text}")
        
        # モデル保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n💾 モデル保存: {output_dir}")
        print("✅ ファインチューニング完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_advanced_options():
    """上級者向けオプション"""
    print("\n🔥 上級者向けオプション:")
    print("=" * 40)
    
    print("\n📊 **マルチGPU最適化:**")
    print("```python")
    print("# Model Parallelism (大規模モデル用)")
    print("model = AutoModelForCausalLM.from_pretrained(")
    print("    model_name,")
    print("    device_map='auto',  # 自動GPU配置")
    print("    torch_dtype=torch.float16,")
    print("    load_in_8bit=True  # メモリ効率化")
    print(")")
    print("```")
    
    print("\n💎 **QLoRA設定 (13B+モデル用):**")
    print("```python")
    print("from transformers import BitsAndBytesConfig")
    print("")
    print("bnb_config = BitsAndBytesConfig(")
    print("    load_in_4bit=True,")
    print("    bnb_4bit_quant_type='nf4',")
    print("    bnb_4bit_compute_dtype=torch.float16")
    print(")")
    print("")
    print("peft_config = LoraConfig(")
    print("    r=8,  # 小さめのランク")
    print("    lora_alpha=16,")
    print("    target_modules=['q_proj', 'v_proj'],")
    print("    lora_dropout=0.1")
    print(")")
    print("```")
    
    print("\n🎯 **実際のデータ使用例:**")
    print("```python")
    print("# CSVファイルから読み込み")
    print("import pandas as pd")
    print("df = pd.read_csv('your_data.csv')")
    print("train_texts = df['text'].tolist()")
    print("")
    print("# JSONファイルから読み込み")
    print("import json")
    print("with open('data.json', 'r') as f:")
    print("    data = json.load(f)")
    print("train_texts = [item['text'] for item in data]")
    print("```")

def show_troubleshooting():
    """トラブルシューティング"""
    print("\n🔧 トラブルシューティング:")
    print("=" * 40)
    
    print("\n❌ **よくあるエラーと解決法:**")
    print("\n1. CUDA Out of Memory:")
    print("   解決法: batch_sizeを1に、gradient_accumulation_stepsを増やす")
    print("   ```python")
    print("   per_device_train_batch_size=1,")
    print("   gradient_accumulation_steps=8")
    print("   ```")
    
    print("\n2. モデル読み込みエラー:")
    print("   解決法: より軽量なモデルを使用")
    print("   ```python")
    print("   model_name = 'distilgpt2'  # 最も軽量")
    print("   ```")
    
    print("\n3. トークナイザーエラー:")
    print("   解決法: use_fast=Falseを追加")
    print("   ```python")
    print("   tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)")
    print("   ```")
    
    print("\n✅ **成功の確認方法:**")
    print("• 訓練Loss が下がっている")
    print("• 生成テキストが改善されている")
    print("• ファイルが正常に保存されている")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            show_advanced_options()
            show_troubleshooting()
            
            print("\n" + "=" * 60)
            print("🎉 ファインチューニング実践ガイド完了！")
            print("=" * 60)
            print("\n次のステップ:")
            print("• より大きなモデルに挑戦")
            print("• 独自データセットでの学習")
            print("• マルチGPU設定の最適化")
            print("• 量子化モデルでの学習")
            
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()