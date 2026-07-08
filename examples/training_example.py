#!/usr/bin/env python3
"""
ファインチューニングの使用例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.japanese_model import JapaneseModel
from src.training.full_finetuning import FullFinetuningTrainer
from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from src.training.quantization import QuantizationOptimizer
from src.training.training_utils import TrainingConfig


def example_full_finetuning():
    """フルファインチューニングの例"""
    print("=== Full Fine-tuning Example ===")
    
    # モデルの初期化
    model = JapaneseModel(
        model_name="stabilityai/japanese-stablelm-3b-4e1t-instruct",
        load_in_8bit=True  # メモリ節約
    )
    
    # トレーニング設定
    config = TrainingConfig(
        learning_rate=2e-5,
        batch_size=2,
        gradient_accumulation_steps=8,
        num_epochs=3,
        warmup_steps=100,
        eval_steps=100,
        save_steps=500,
        output_dir="./outputs/full_finetuning",
        fp16=True,
        gradient_checkpointing=True
    )
    
    # サンプルデータ
    train_texts = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さは何メートルですか？\n回答: 富士山の高さは3,776メートルです。",
        "質問: 日本の人口は約何人ですか？\n回答: 日本の人口は約1億2,500万人です。"
    ] * 50  # データを増やす
    
    # トレーナーの初期化
    trainer = FullFinetuningTrainer(
        model=model,
        config=config
    )
    
    # トレーニング実行
    trained_model = trainer.train(train_texts=train_texts)
    
    print("Full fine-tuning completed!")
    return trained_model


def example_lora_finetuning():
    """LoRAファインチューニングの例"""
    print("=== LoRA Fine-tuning Example ===")
    
    # モデルの初期化
    model = JapaneseModel(
        model_name="stabilityai/japanese-stablelm-3b-4e1t-instruct"
    )
    
    # LoRA設定
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        use_qlora=False
    )
    
    # トレーニング設定
    training_config = TrainingConfig(
        learning_rate=3e-4,  # LoRAは高めの学習率
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=5,
        output_dir="./outputs/lora_finetuning",
        fp16=True
    )
    
    # サンプルデータ
    train_texts = [
        "以下の文章を要約してください。\n" +
        "人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。" +
        "機械学習、深層学習、自然言語処理などの技術を含みます。\n" +
        "要約: AIは人間の知能を模倣する技術で、機械学習や自然言語処理を含みます。",
        
        "次の質問に答えてください。\n" +
        "質問: 機械学習とは何ですか？\n" +
        "回答: 機械学習は、データから自動的にパターンを学習するAIの手法です。"
    ] * 100
    
    # トレーナーの初期化
    trainer = LoRAFinetuningTrainer(
        model=model,
        lora_config=lora_config,
        training_config=training_config
    )
    
    # トレーニング実行
    trained_model = trainer.train(train_texts=train_texts)
    
    print("LoRA fine-tuning completed!")
    return trained_model


def example_qlora_finetuning():
    """QLoRAファインチューニングの例"""
    print("=== QLoRA Fine-tuning Example ===")
    
    # モデルの初期化（量子化なしで初期化）
    model = JapaneseModel(
        model_name="elyza/Llama-3-ELYZA-JP-8B"  # より大きなモデルでQLoRAのメリットを活用
    )
    
    # QLoRA設定
    qlora_config = LoRAConfig(
        r=8,  # QLoRAでは小さめのrank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        use_qlora=True,
        qlora_4bit=True  # 4bit量子化
    )
    
    # トレーニング設定
    training_config = TrainingConfig(
        learning_rate=2e-4,
        batch_size=1,  # QLoRAでも大きなモデルは小さいバッチサイズ
        gradient_accumulation_steps=16,
        num_epochs=3,
        output_dir="./outputs/qlora_finetuning",
        fp16=True,
        gradient_checkpointing=True
    )
    
    # 対話形式のデータ
    train_texts = [
        "<|user|>こんにちは<|endoftext|>\n<|assistant|>こんにちは！何かお手伝いできることはありますか？<|endoftext|>",
        "<|user|>今日の天気はどうですか？<|endoftext|>\n<|assistant|>申し訳ございませんが、リアルタイムの天気情報にはアクセスできません。お住まいの地域の天気予報サイトをご確認ください。<|endoftext|>",
        "<|user|>プログラミングについて教えてください<|endoftext|>\n<|assistant|>プログラミングは、コンピュータに実行させたい処理を、プログラミング言語を使って記述することです。Python、JavaScript、Javaなど様々な言語があります。<|endoftext|>"
    ] * 50
    
    # トレーナーの初期化
    trainer = LoRAFinetuningTrainer(
        model=model,
        lora_config=qlora_config,
        training_config=training_config
    )
    
    # トレーニング実行
    trained_model = trainer.train(train_texts=train_texts)
    
    print("QLoRA fine-tuning completed!")
    return trained_model


def example_quantization():
    """量子化の例"""
    print("=== Quantization Example ===")
    
    model_name = "stabilityai/japanese-stablelm-3b-4e1t-instruct"
    
    # 量子化オプティマイザーの初期化
    quantizer = QuantizationOptimizer(model_name)
    
    # 8bit量子化
    print("Performing 8-bit quantization...")
    quantized_8bit = quantizer.quantize_to_8bit(
        output_dir="./outputs/quantized_8bit"
    )
    
    # 4bit量子化
    print("Performing 4-bit quantization...")
    quantized_4bit = quantizer.quantize_to_4bit(
        output_dir="./outputs/quantized_4bit"
    )
    
    print("Quantization completed!")
    
    # 量子化モデルのロード例
    print("Loading quantized models...")
    model_8bit, tokenizer_8bit = QuantizationOptimizer.load_quantized_model(
        "./outputs/quantized_8bit"
    )
    
    model_4bit, tokenizer_4bit = QuantizationOptimizer.load_quantized_model(
        "./outputs/quantized_4bit"
    )
    
    return model_8bit, model_4bit


def example_inference_with_finetuned():
    """ファインチューニング済みモデルでの推論例"""
    print("=== Inference with Fine-tuned Model ===")
    
    # LoRAモデルのロード例
    try:
        model, tokenizer = LoRAFinetuningTrainer.load_lora_model(
            base_model_name="stabilityai/japanese-stablelm-3b-4e1t-instruct",
            lora_adapter_path="./outputs/lora_finetuning/best_lora_model"
        )
        
        # 推論テスト
        prompt = "質問: 機械学習とは何ですか？\n回答:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Could not load fine-tuned model: {e}")
        print("Please run training examples first.")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning examples")
    parser.add_argument("--example", type=str, choices=[
        "full", "lora", "qlora", "quantization", "inference", "all"
    ], default="all", help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example in ["full", "all"]:
        try:
            example_full_finetuning()
        except Exception as e:
            print(f"Full fine-tuning example failed: {e}")
    
    if args.example in ["lora", "all"]:
        try:
            example_lora_finetuning()
        except Exception as e:
            print(f"LoRA fine-tuning example failed: {e}")
    
    if args.example in ["qlora", "all"]:
        try:
            example_qlora_finetuning()
        except Exception as e:
            print(f"QLoRA fine-tuning example failed: {e}")
    
    if args.example in ["quantization", "all"]:
        try:
            example_quantization()
        except Exception as e:
            print(f"Quantization example failed: {e}")
    
    if args.example in ["inference", "all"]:
        try:
            example_inference_with_finetuned()
        except Exception as e:
            print(f"Inference example failed: {e}")


if __name__ == "__main__":
    main()