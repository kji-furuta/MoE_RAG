#!/usr/bin/env python3
"""
簡単なLoRAファインチューニングチュートリアル
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from datetime import datetime

class SimpleQADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def main():
    print("🚀 RTX A5000 x2 簡単LoRAファインチューニング")
    print("=" * 50)
    
    # GPU確認
    device_count = torch.cuda.device_count()
    print(f"✅ 利用可能GPU: {device_count}台")
    
    if device_count >= 2:
        print("   GPU 0: RTX A5000 (24GB)")
        print("   GPU 1: RTX A5000 (24GB)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用デバイス: {device}")
    
    # Step 1: モデルとトークナイザーの準備
    print("\n📚 Step 1: モデル読み込み")
    model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
    
    print(f"モデル: {model_name}")
    print("読み込み中...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # 自動的にGPUに配置
            trust_remote_code=True
        )
        
        print("✅ モデル読み込み完了")
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        print("フォールバックモデルを使用...")
        
        model_name = "rinna/japanese-gpt-neox-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ フォールバックモデル読み込み完了")
    
    # Step 2: LoRA設定
    print("\n⚙️ Step 2: LoRA設定")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("✅ LoRA設定完了")
    
    # Step 3: 訓練データ準備
    print("\n📝 Step 3: 訓練データ準備")
    
    train_texts = [
        "ユーザー: 日本の首都はどこですか？\nシステム: 日本の首都は東京です。",
        "ユーザー: 富士山の高さは何メートルですか？\nシステム: 富士山の高さは3,776メートルです。",
        "ユーザー: 今日はいい天気ですね。\nシステム: そうですね。とても良い天気で気持ちがいいです。",
        "ユーザー: ありがとうございました。\nシステム: どういたしまして。お役に立てて嬉しいです。",
        "ユーザー: おはようございます。\nシステム: おはようございます。今日も一日よろしくお願いします。",
    ]
    
    dataset = SimpleQADataset(train_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"✅ 訓練データ: {len(train_texts)}件")
    
    # Step 4: 訓練設定
    print("\n📊 Step 4: 訓練設定")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 3
    
    print(f"✅ エポック数: {num_epochs}")
    print(f"✅ 学習率: 3e-4")
    print(f"✅ バッチサイズ: 1")
    
    # Step 5: 訓練実行
    print("\n🏋️ Step 5: 訓練実行")
    
    model.train()
    total_loss = 0
    step = 0
    
    for epoch in range(num_epochs):
        print(f"\nエポック {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            step += 1
            
            if step % 2 == 0:
                print(f"  ステップ {step}: Loss = {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"エポック {epoch + 1} 平均Loss: {avg_epoch_loss:.4f}")
    
    print("\n✅ 訓練完了！")
    
    # Step 6: テスト生成
    print("\n🧪 Step 6: テスト生成")
    
    model.eval()
    test_input = "ユーザー: こんにちは\nシステム:"
    
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"入力: {test_input}")
    print(f"出力: {response}")
    
    # Step 7: モデル保存
    print("\n💾 Step 7: モデル保存")
    
    output_dir = f"./lora_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ モデル保存完了: {output_dir}")
    
    return True

def show_next_steps():
    print("\n🎯 次のステップ:")
    print("=" * 30)
    print("1. より大きなモデルを試す:")
    print("   - elyza/Llama-3-ELYZA-JP-8B")
    print("   - stabilityai/japanese-stablelm-3b-4e1t-instruct")
    
    print("\n2. QLoRAで大規模モデル:")
    print("   - 13B以上のモデル")
    print("   - 4bit量子化使用")
    
    print("\n3. マルチGPU設定:")
    print("   - device_map='auto'")
    print("   - Model Parallelism")
    
    print("\n4. 独自データセット:")
    print("   - CSVファイルから読み込み")
    print("   - より多くの訓練データ")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            show_next_steps()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()