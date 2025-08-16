#!/usr/bin/env python3
"""
大規模モデル（17B/32B）ファインチューニング例
RTX A5000 x2環境での大規模モデルファインチューニング
"""

import torch
from src.models.japanese_model import JapaneseModel
from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from src.training.training_utils import TrainingConfig
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """大規模モデルファインチューニングのメイン実行"""
    
    print("🚀 大規模モデル（17B/32B）ファインチューニング例")
    print("=" * 60)
    
    # 利用可能な大規模モデルを表示
    large_models = JapaneseModel.list_large_models()
    print(f"📋 利用可能な大規模モデル（17B+）: {len(large_models)}個")
    
    for model_name, config in large_models.items():
        print(f"  • {config['display_name']}")
        print(f"    必要VRAM: {config['min_gpu_memory_gb']}GB")
        print(f"    認証必要: {config.get('requires_auth', False)}")
        print()
    
    # モデルサイズ別の分類を表示
    models_by_size = JapaneseModel.list_models_by_size()
    print("📊 モデルサイズ別分類:")
    for size, models in models_by_size.items():
        if models:
            print(f"  {size}: {len(models)}個")
    
    print("\n" + "=" * 60)
    
    # 例1: 17Bモデル（Phi-3.5-17B）のファインチューニング
    print("🎯 例1: Microsoft Phi-3.5-17B ファインチューニング")
    print("-" * 40)
    
    # 17Bモデルの情報を取得
    model_info = JapaneseModel.get_model_info("microsoft/Phi-3.5-17B-Instruct")
    print(f"モデル情報: {model_info['display_name']}")
    print(f"必要VRAM: {model_info['min_gpu_memory_gb']}GB")
    print(f"推奨設定: {model_info['recommended_training_config']}")
    
    # 17Bモデルの初期化（QLoRA推奨）
    model_17b = JapaneseModel(
        model_name="microsoft/Phi-3.5-17B-Instruct",
        load_in_8bit=True,  # メモリ効率化
        gradient_checkpointing=True,  # 自動有効化
        use_flash_attention=True
    )
    
    # QLoRA設定
    qlora_config = LoRAConfig(
        r=8,                    # 小さいランクでメモリ節約
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        use_qlora=True,         # QLoRA有効
        qlora_4bit=False        # 17Bなら8bitで十分
    )
    
    # トレーニング設定
    training_config = TrainingConfig(
        learning_rate=2e-4,     # 大規模モデル用の低い学習率
        batch_size=2,           # 小さいバッチサイズ
        gradient_accumulation_steps=8,  # 実効バッチサイズ16
        num_epochs=3,           # 少ないエポック数
        output_dir="./outputs/phi3.5_17b_qlora",
        gradient_checkpointing=True,
        fp16=True,
        save_steps=100,
        logging_steps=10
    )
    
    print(f"✅ 17Bモデル設定完了")
    print(f"   LoRAランク: {qlora_config.r}")
    print(f"   バッチサイズ: {training_config.batch_size}")
    print(f"   実効バッチサイズ: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    
    print("\n" + "=" * 60)
    
    # 例2: 32Bモデル（Qwen2.5-32B）のファインチューニング
    print("🎯 例2: Qwen 2.5 32B ファインチューニング")
    print("-" * 40)
    
    # 32Bモデルの情報を取得
    model_info_32b = JapaneseModel.get_model_info("Qwen/Qwen2.5-32B-Instruct")
    print(f"モデル情報: {model_info_32b['display_name']}")
    print(f"必要VRAM: {model_info_32b['min_gpu_memory_gb']}GB")
    print(f"推奨設定: {model_info_32b['recommended_training_config']}")
    
    # 32Bモデルの初期化（4bit QLoRA必須）
    model_32b = JapaneseModel(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        load_in_4bit=True,      # 4bit量子化必須
        gradient_checkpointing=True,
        use_flash_attention=True
    )
    
    # 4bit QLoRA設定
    qlora_config_32b = LoRAConfig(
        r=4,                    # さらに小さいランク
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        use_qlora=True,
        qlora_4bit=True         # 4bit量子化
    )
    
    # 32B用トレーニング設定
    training_config_32b = TrainingConfig(
        learning_rate=2e-4,
        batch_size=1,           # 最小バッチサイズ
        gradient_accumulation_steps=16,  # 実効バッチサイズ16
        num_epochs=2,           # さらに少ないエポック数
        output_dir="./outputs/qwen2.5_32b_qlora",
        gradient_checkpointing=True,
        fp16=True,
        save_steps=50,
        logging_steps=5
    )
    
    print(f"✅ 32Bモデル設定完了")
    print(f"   LoRAランク: {qlora_config_32b.r}")
    print(f"   4bit量子化: {qlora_config_32b.qlora_4bit}")
    print(f"   バッチサイズ: {training_config_32b.batch_size}")
    print(f"   実効バッチサイズ: {training_config_32b.batch_size * training_config_32b.gradient_accumulation_steps}")
    
    print("\n" + "=" * 60)
    
    # サンプルデータ
    train_texts = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さは何メートルですか？\n回答: 富士山の高さは3,776メートルです。",
        "質問: 日本の国花は何ですか？\n回答: 日本の国花は桜です。",
        "質問: 日本の人口は何人ですか？\n回答: 日本の人口は約1億2,500万人です。",
        "質問: 日本の通貨は何ですか？\n回答: 日本の通貨は円（JPY）です。"
    ]
    
    print("📝 サンプルデータ準備完了")
    print(f"   データ数: {len(train_texts)}件")
    
    print("\n" + "=" * 60)
    print("🚀 ファインチューニング実行例")
    print("-" * 40)
    
    # 実際の実行はコメントアウト（時間がかかるため）
    print("""
# 17Bモデルのファインチューニング実行例:
trainer_17b = LoRAFinetuningTrainer(
    model=model_17b,
    lora_config=qlora_config,
    training_config=training_config
)

# トレーニング実行
trained_model_17b = trainer_17b.train(train_texts=train_texts)

# 32Bモデルのファインチューニング実行例:
trainer_32b = LoRAFinetuningTrainer(
    model=model_32b,
    lora_config=qlora_config_32b,
    training_config=training_config_32b
)

# トレーニング実行
trained_model_32b = trainer_32b.train(train_texts=train_texts)
""")
    
    print("=" * 60)
    print("💡 大規模モデルファインチューニングのポイント:")
    print("• 17B+モデルはQLoRA必須")
    print("• 32B+モデルは4bit量子化推奨")
    print("• バッチサイズは1-2に制限")
    print("• Gradient Accumulationで実効バッチサイズを確保")
    print("• Gradient Checkpointingでメモリ節約")
    print("• 学習率は2e-4程度に設定")
    print("• エポック数は2-3回程度")
    
    print("\n🎉 大規模モデルファインチューニング準備完了！")

if __name__ == "__main__":
    main() 