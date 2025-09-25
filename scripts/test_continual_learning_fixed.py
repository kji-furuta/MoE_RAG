#!/usr/bin/env python3
"""
継続学習システムの修正版テストスクリプト
EWCFullFinetuningTrainer初期化エラー修正後のテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_continual_learning():
    """継続学習システムのエンドツーエンドテスト"""

    from src.training.continual_learning_pipeline import ContinualLearningPipeline
    from src.training.training_utils import TrainingConfig

    # 設定の準備
    config = TrainingConfig(
        model_name_or_path="outputs/continual_task_1",  # 既存のLoRAファインチューニング済みモデル
        output_dir="outputs/test_continual_fixed",
        num_epochs=1,  # テスト用に1エポックのみ
        batch_size=1,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        ewc_lambda=5000.0,  # EWC有効化
        use_lora=True,  # LoRAを使用
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    # パイプラインの作成
    pipeline = ContinualLearningPipeline(config)
    logger.info("ContinualLearningPipeline created successfully")

    # 新しいタスクのデータ準備（最小限のテストデータ）
    test_data_file = Path("test_data_continual.jsonl")
    with open(test_data_file, "w", encoding="utf-8") as f:
        f.write('{"text": "これはテスト用のデータです。継続学習のテストを行います。"}\n')
        f.write('{"text": "EWCを使用した継続学習により、以前のタスクの知識を保持します。"}\n')
        f.write('{"text": "LoRAアダプターを使用した効率的な学習を実現します。"}\n')

    logger.info(f"Test data created at {test_data_file}")

    try:
        # タスクの実行
        logger.info("Starting continual learning task...")
        result = pipeline.train_task(
            task_id="test_task",
            data_path=str(test_data_file),
            num_epochs=1
        )

        logger.info(f"✅ Continual learning task completed successfully!")
        logger.info(f"Result: {result}")

        # モデルが正しく保存されたか確認
        output_path = Path(config.output_dir)
        if output_path.exists():
            logger.info(f"✅ Output directory created: {output_path}")

            # 保存されたファイルを確認
            saved_files = list(output_path.glob("**/*"))
            logger.info(f"Saved files: {len(saved_files)} files")
            for file in saved_files[:10]:  # 最初の10ファイルのみ表示
                logger.info(f"  - {file.relative_to(output_path)}")

        return True

    except Exception as e:
        logger.error(f"❌ Error during continual learning: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        if test_data_file.exists():
            test_data_file.unlink()
            logger.info("Test data cleaned up")

def test_lora_on_lora():
    """LoRA-on-LoRA継続学習の検証"""

    from src.training.continual_learning_pipeline import ContinualLearningPipeline

    logger.info("=== Testing LoRA-on-LoRA Continual Learning ===")

    # 既存のLoRAモデルパスを確認
    base_lora_path = Path("outputs/continual_task_1")
    if not base_lora_path.exists():
        logger.warning(f"Base LoRA model not found at {base_lora_path}")
        logger.info("Checking for alternative LoRA models...")

        # 他のLoRAモデルを探す
        outputs_dir = Path("outputs")
        lora_models = [p for p in outputs_dir.glob("*/adapter_config.json")]

        if lora_models:
            base_lora_path = lora_models[0].parent
            logger.info(f"Using alternative LoRA model: {base_lora_path}")
        else:
            logger.error("No LoRA models found in outputs directory")
            return False

    # パイプラインの初期化
    pipeline = ContinualLearningPipeline(None)  # configは後で設定

    try:
        # LoRAモデルのロード確認
        logger.info(f"Loading LoRA model from {base_lora_path}")
        model, tokenizer = pipeline.load_finetuned_model(str(base_lora_path))

        # モデルタイプの確認
        model_type = type(model).__name__
        logger.info(f"✅ Model loaded successfully: {model_type}")

        # PeftModelの確認
        if "PeftModel" in model_type:
            logger.info("✅ Successfully loaded as PEFT model (LoRA adapter applied)")

            # アダプター情報の表示
            if hasattr(model, 'peft_config'):
                for adapter_name, config in model.peft_config.items():
                    logger.info(f"  Adapter: {adapter_name}")
                    logger.info(f"    - r: {config.r}")
                    logger.info(f"    - alpha: {config.lora_alpha}")
                    logger.info(f"    - dropout: {config.lora_dropout}")
        else:
            logger.warning(f"Model is not a PEFT model: {model_type}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load LoRA model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""

    logger.info("=" * 60)
    logger.info("Starting Continual Learning System Tests")
    logger.info("=" * 60)

    # GPUメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # テスト実行
    all_passed = True

    # LoRA-on-LoRAテスト
    logger.info("\n" + "=" * 40)
    logger.info("Test 1: LoRA-on-LoRA Loading")
    logger.info("=" * 40)
    if not test_lora_on_lora():
        all_passed = False
        logger.error("LoRA-on-LoRA test failed")

    # エンドツーエンドテスト
    logger.info("\n" + "=" * 40)
    logger.info("Test 2: End-to-End Continual Learning")
    logger.info("=" * 40)
    if not test_continual_learning():
        all_passed = False
        logger.error("End-to-end test failed")

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ All tests PASSED!")
    else:
        logger.error("❌ Some tests FAILED")
    logger.info("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())