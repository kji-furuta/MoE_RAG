#!/usr/bin/env python3
"""
動作確認済みLoRAファインチューニングデモ
RTX A5000 x2環境
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

# WandBを無効化
os.environ["WANDB_DISABLED"] = "true"

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
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
    print("🚀 RTX A5000 x2 動作確認済みLoRAデモ")
    print("=" * 50)
    
    # GPU確認
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ GPU: {device_count}台利用可能")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    # Step 1: モデル準備
    print("\n📚 Step 1: モデル準備")
    model_name = "distilgpt2"
    print(f"使用モデル: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ モデル読み込み完了")
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False
    
    # Step 2: LoRA設定
    print("\n⚙️ Step 2: LoRA設定")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    print("✅ LoRA適用完了")
    model.print_trainable_parameters()
    
    # Step 3: 訓練データ
    print("\n📝 Step 3: 訓練データ準備")
    train_texts = [
        "Hello, how are you today?",
        "The weather is beautiful.",
        "I love machine learning.",
        "RTX A5000 is powerful.",
        "Fine-tuning is exciting."
    ]
    
    dataset = SimpleTextDataset(train_texts, tokenizer)
    print(f"✅ データセット: {len(train_texts)}件")
    
    # Step 4: 訓練設定
    print("\n📊 Step 4: 訓練設定")
    output_dir = f"./lora_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=1,
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to=[],  # WandB無効化
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("✅ トレーナー準備完了")
    
    # Step 5: 訓練前テスト
    print("\n🧪 Step 5: 訓練前テスト")
    test_input = "Hello, how are"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=15,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    before_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"入力: {test_input}")
    print(f"訓練前出力: {before_text}")
    
    # Step 6: 訓練実行
    print("\n🏋️ Step 6: 訓練実行")
    print("訓練開始...")
    
    try:
        trainer.train()
        print("✅ 訓練完了！")
        
    except Exception as e:
        print(f"❌ 訓練エラー: {e}")
        return False
    
    # Step 7: 訓練後テスト
    print("\n🎯 Step 7: 訓練後テスト")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=15,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    after_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"入力: {test_input}")
    print(f"訓練後出力: {after_text}")
    
    # Step 8: モデル保存
    print("\n💾 Step 8: モデル保存")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 保存完了: {output_dir}")
        
    except Exception as e:
        print(f"❌ 保存エラー: {e}")
    
    print("\n🎉 LoRAファインチューニング完了！")
    
    return True

def show_summary():
    """完了後の要約"""
    print("\n" + "=" * 60)
    print("📋 実行完了 - ファインチューニング手順まとめ")
    print("=" * 60)
    
    print("\n✅ 今回実行した内容:")
    print("• DistilGPT-2モデルの読み込み")
    print("• LoRA (Low-Rank Adaptation) の適用")
    print("• 小規模データセットでの訓練")
    print("• 訓練前後の生成テスト")
    print("• モデルの保存")
    
    print("\n🚀 RTX A5000 x2での最適化ポイント:")
    print("• device_map='auto' でマルチGPU活用")
    print("• fp16=True でメモリ効率化")
    print("• gradient_accumulation_steps でバッチサイズ調整")
    print("• LoRAで訓練可能パラメータを0.5%に削減")
    
    print("\n🎯 次のステップ:")
    print("1. より大きなモデル (3B-8B) に挑戦")
    print("2. 日本語データでファインチューニング")
    print("3. QLoRA で13B+モデルを試す")
    print("4. 独自データセットでの学習")
    
    print("\n💡 実用的な応用例:")
    print("• Q&Aシステムの構築")
    print("• 対話システムのカスタマイズ")
    print("• 文書要約システム")
    print("• 特定ドメインの言語モデル")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            show_summary()
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()