#!/usr/bin/env python3
"""
実践的LoRAファインチューニングチュートリアル
RTX A5000 x2環境最適化済み
"""

import torch
import os
from datetime import datetime
from src.models.japanese_model import JapaneseModel
from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from src.training.training_utils import TrainingConfig

def main():
    print("🚀 RTX A5000 x2 LoRAファインチューニング実践ガイド")
    print("=" * 60)
    
    # GPU確認
    if torch.cuda.device_count() >= 2:
        print(f"✅ {torch.cuda.device_count()}台のGPU検出")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory/(1024**3):.1f}GB)")
    
    # Step 1: モデル選択
    print("\n📚 Step 1: モデル選択")
    print("推奨モデル（RTX A5000 x2環境）:")
    print("• stabilityai/japanese-stablelm-3b-4e1t-instruct (8GB使用)")
    print("• elyza/Llama-3-ELYZA-JP-8B (16GB使用)")
    
    # 軽量モデルから開始
    model_name = "stabilityai/japanese-stablelm-3b-4e1t-instruct"
    print(f"\n選択: {model_name}")
    
    # Step 2: モデル初期化
    print("\n🔧 Step 2: モデル初期化")
    model = JapaneseModel(
        model_name=model_name,
        load_in_8bit=True,  # メモリ効率化
        use_flash_attention=True,  # 高速化
        gradient_checkpointing=True  # メモリ節約
    )
    
    print("✅ モデル設定完了")
    
    # Step 3: 訓練データ準備
    print("\n📝 Step 3: 訓練データ準備")
    
    # 実用的な例：Q&Aシステム
    train_texts = [
        "質問: 東京の人口は何人ですか？\n回答: 東京都の人口は約1,400万人です。",
        "質問: 日本で一番高い山はどこですか？\n回答: 日本で一番高い山は富士山で、標高3,776メートルです。",
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 日本語でこんにちはは何と言いますか？\n回答: 日本語で挨拶する時は「こんにちは」と言います。",
        "質問: 寿司の主な材料は何ですか？\n回答: 寿司の主な材料は酢飯と新鮮な魚介類です。",
        "質問: 桜の季節はいつですか？\n回答: 桜の季節は通常3月下旬から5月上旬です。",
        "質問: 日本の通貨は何ですか？\n回答: 日本の通貨は円（えん）です。",
        "質問: 新幹線の最高速度はどのくらいですか？\n回答: 新幹線の最高速度は路線により異なりますが、最高320km/hです。",
        "質問: 日本の国花は何ですか？\n回答: 日本の国花は桜です。",
        "質問: 味噌汁の主な材料は何ですか？\n回答: 味噌汁の主な材料は味噌、だし、具材（豆腐、わかめなど）です。"
    ]
    
    print(f"✅ 訓練データ: {len(train_texts)}件")
    
    # Step 4: LoRA設定
    print("\n⚙️ Step 4: LoRA設定（RTX A5000 x2最適化）")
    
    lora_config = LoRAConfig(
        r=16,  # LoRAランク（品質と速度のバランス）
        lora_alpha=32,  # アルファ値（通常r*2）
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 対象モジュール
        lora_dropout=0.05,  # ドロップアウト
        use_qlora=False,  # QLoRAは使用しない（8bit量子化で十分）
        bias="none"
    )
    
    print("✅ LoRA設定:")
    print(f"   ランク: {lora_config.r}")
    print(f"   アルファ: {lora_config.lora_alpha}")
    print(f"   対象モジュール: {lora_config.target_modules}")
    
    # Step 5: 訓練設定
    print("\n📊 Step 5: 訓練設定")
    
    output_dir = f"./outputs/lora_qa_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_config = TrainingConfig(
        learning_rate=3e-4,  # LoRA推奨学習率
        batch_size=4,  # RTX A5000で安全なサイズ
        gradient_accumulation_steps=4,  # 実効バッチサイズ16
        num_epochs=5,  # 少ないデータなので多めのエポック
        warmup_steps=50,
        max_grad_norm=1.0,
        eval_steps=10,
        save_steps=50,
        logging_steps=5,
        output_dir=output_dir,
        fp16=True,  # RTX A5000で高速化
        gradient_checkpointing=True,  # メモリ効率化
        ddp=False  # 単一プロセスから開始
    )
    
    print("✅ 訓練設定:")
    print(f"   学習率: {training_config.learning_rate}")
    print(f"   バッチサイズ: {training_config.batch_size}")
    print(f"   実効バッチサイズ: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"   エポック数: {training_config.num_epochs}")
    print(f"   出力ディレクトリ: {output_dir}")
    
    # Step 6: 訓練実行
    print("\n🏋️ Step 6: 訓練実行")
    
    try:
        trainer = LoRAFinetuningTrainer(
            model=model,
            lora_config=lora_config,
            training_config=training_config
        )
        
        print("✅ トレーナー初期化完了")
        print("🚀 訓練開始...")
        
        # 訓練前のテスト
        print("\n--- 訓練前のテスト ---")
        test_input = "質問: 日本の首都はどこですか？"
        
        if hasattr(trainer, 'model') and trainer.model.model is not None:
            response = model.generate_japanese(test_input, max_new_tokens=50)
            print(f"入力: {test_input}")
            print(f"出力: {response}")
        
        # 実際の訓練
        trained_model = trainer.train(train_texts=train_texts)
        
        print("\n✅ 訓練完了！")
        
        # 訓練後のテスト
        print("\n--- 訓練後のテスト ---")
        if trained_model:
            response = model.generate_japanese(test_input, max_new_tokens=50)
            print(f"入力: {test_input}")
            print(f"出力: {response}")
        
        print(f"\n💾 モデル保存場所: {output_dir}")
        print("✅ LoRAファインチューニング完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def usage_tips():
    """使用のヒント"""
    print("\n💡 使用のヒント:")
    print("=" * 30)
    print("1. 初回は軽量モデル（3B）から開始")
    print("2. データが少ない場合はエポック数を増やす")
    print("3. メモリ不足の場合はbatch_sizeを下げる")
    print("4. 学習率は3e-4から始めて調整")
    print("5. 保存されたモデルは後で読み込み可能")
    
    print("\n🔄 次のステップ:")
    print("• より大きなモデル（8B）を試す")
    print("• QLoRAで13B+モデルに挑戦")
    print("• 独自データでファインチューニング")
    print("• マルチGPU設定で高速化")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            usage_tips()
    except KeyboardInterrupt:
        print("\n⚠️ 訓練が中断されました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")