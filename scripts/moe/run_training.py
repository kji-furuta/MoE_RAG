#!/usr/bin/env python3
"""
Run MoE Training
MoEモデルのトレーニング実行スクリプト
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

import torch
import argparse
from pathlib import Path
from src.moe.moe_architecture import create_civil_engineering_moe, MoEConfig
from src.moe.moe_training import MoETrainer, MoETrainingConfig, CivilEngineeringDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="Train MoE model for civil engineering")
    
    # モデル設定
    parser.add_argument("--base_model", type=str, default="cyberagent/calm3-22b-chat",
                      help="Base model name")
    parser.add_argument("--num_experts", type=int, default=8,
                      help="Number of experts")
    parser.add_argument("--num_experts_per_tok", type=int, default=2,
                      help="Number of active experts per token")
    
    # トレーニング設定
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                      help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # パス設定
    parser.add_argument("--data_path", type=str, default="./data/civil_engineering",
                      help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./outputs/moe_civil",
                      help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/moe_civil",
                      help="Checkpoint directory")
    
    # その他
    parser.add_argument("--use_mixed_precision", action="store_true",
                      help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Use gradient checkpointing")
    parser.add_argument("--demo_mode", action="store_true",
                      help="Run in demo mode with smaller model")
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_args()
    
    print("=" * 60)
    print("MoE土木・建設特化モデル トレーニング")
    print("=" * 60)
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"✓ GPU利用可能: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("! GPUが利用できません。CPUモードで実行します。")
    
    # 設定の作成
    config = MoETrainingConfig(
        base_model_name=args.base_model,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        dataset_path=args.data_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        mixed_precision="bf16" if args.use_mixed_precision else None,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # デモモードの設定
    if args.demo_mode:
        print("\n📝 デモモード: 小規模モデルで実行します")
        config.num_epochs = 1
        config.max_seq_length = 128
        config.logging_steps = 1
        config.save_steps = 10
        config.eval_steps = 5
    
    print("\n設定:")
    print(f"  エキスパート数: {config.num_experts}")
    print(f"  アクティブエキスパート: {config.num_experts_per_tok}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  エポック数: {config.num_epochs}")
    
    # モデルの作成
    print("\nモデルを作成中...")
    
    if args.demo_mode:
        # デモ用の小規模モデル
        moe_config = MoEConfig(
            hidden_size=512,  # 小規模
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            domain_specific_routing=True
        )
        from src.moe.moe_architecture import CivilEngineeringMoEModel
        model = CivilEngineeringMoEModel(moe_config, base_model=None)
    else:
        model = create_civil_engineering_moe(
            base_model_name=config.base_model_name,
            num_experts=config.num_experts
        )
    
    print(f"✓ モデル作成完了")
    print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # トークナイザーの準備
    print("\nトークナイザーを準備中...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    except:
        logger.warning("Failed to load tokenizer, using dummy tokenizer")
        # ダミートークナイザー
        class DummyTokenizer:
            vocab_size = 32000  # モデルのembedding層と一致させる
            safe_vocab_size = 30000  # 安全マージン（93.75%）を確保
            
            def __call__(self, text, **kwargs):
                max_length = kwargs.get('max_length', 100)
                # 安全な範囲でのみトークンを生成（0-29999）
                # 特殊トークン用に0-10を予約、通常トークンは11-29999
                return {
                    'input_ids': torch.randint(11, self.safe_vocab_size, (1, max_length)),
                    'attention_mask': torch.ones(1, max_length)
                }
            
            def __len__(self):
                return self.vocab_size  # embedding層のサイズは32000のまま
        tokenizer = DummyTokenizer()
    
    # データセットの準備
    print("\nデータセットを準備中...")
    
    train_dataset = CivilEngineeringDataset(
        data_path=f"{config.dataset_path}/train",
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    val_dataset = CivilEngineeringDataset(
        data_path=f"{config.dataset_path}/val",
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    print(f"✓ データセット準備完了")
    print(f"  トレーニングサンプル: {len(train_dataset)}")
    print(f"  検証サンプル: {len(val_dataset)}")
    
    # トレーナーの初期化
    print("\nトレーナーを初期化中...")
    trainer = MoETrainer(model, config, tokenizer)
    
    # トレーニング実行
    print("\n" + "=" * 60)
    print("トレーニングを開始します...")
    print("=" * 60)
    
    try:
        trainer.train(train_dataset, val_dataset)
        print("\n✓ トレーニングが正常に完了しました！")
    except KeyboardInterrupt:
        print("\n! トレーニングが中断されました")
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("トレーニング終了")
    print("=" * 60)
    
    # 結果の保存場所
    print("\n結果の保存場所:")
    print(f"  モデル: {config.output_dir}")
    print(f"  チェックポイント: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
