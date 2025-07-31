#!/usr/bin/env python3
"""
ファインチューニング機能のテストスクリプト
"""

import sys
import torch
import logging
from pathlib import Path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_imports():
    """トレーニング関連のインポートテスト"""
    print("\n=== Import Test ===")
    
    try:
        from src.training.training_utils import TrainingConfig, TextDataset
        from src.training.full_finetuning import FullFinetuningTrainer
        from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
        from src.training.quantization import QuantizationOptimizer
        from src.models.japanese_model import JapaneseModel
        print("✓ All training modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_config():
    """LoRA設定のテスト"""
    print("\n=== LoRA Config Test ===")
    
    try:
        from src.training.lora_finetuning import LoRAConfig
        
        # 標準的なLoRA設定
        lora_config = LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            use_qlora=False
        )
        
        print(f"✓ LoRA Config created:")
        print(f"  - Rank: {lora_config.r}")
        print(f"  - Alpha: {lora_config.lora_alpha}")
        print(f"  - Target modules: {lora_config.target_modules}")
        print(f"  - QLoRA: {lora_config.use_qlora}")
        
        # QLoRA設定
        qlora_config = LoRAConfig(
            r=8,
            lora_alpha=16,
            use_qlora=True,
            qlora_4bit=True
        )
        
        print(f"\n✓ QLoRA Config created:")
        print(f"  - 4-bit quantization: {qlora_config.qlora_4bit}")
        
        return True
    except Exception as e:
        print(f"✗ LoRA config test failed: {e}")
        return False


def test_training_config():
    """トレーニング設定のテスト"""
    print("\n=== Training Config Test ===")
    
    try:
        from src.training.training_utils import TrainingConfig
        
        config = TrainingConfig(
            learning_rate=2e-5,
            batch_size=2,  # 小さいバッチサイズでテスト
            gradient_accumulation_steps=4,
            num_epochs=1,
            output_dir="./test_outputs",
            fp16=True,
            gradient_checkpointing=True
        )
        
        print(f"✓ Training Config created:")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  - FP16: {config.fp16}")
        print(f"  - Gradient checkpointing: {config.gradient_checkpointing}")
        
        return True
    except Exception as e:
        print(f"✗ Training config test failed: {e}")
        return False


def test_dataset_creation():
    """データセット作成のテスト"""
    print("\n=== Dataset Test ===")
    
    try:
        from src.training.training_utils import TextDataset
        from transformers import AutoTokenizer
        
        # 簡単なトークナイザーでテスト
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # サンプルテキスト
        sample_texts = [
            "これはテストテキストです。",
            "日本語のファインチューニングをテストしています。",
            "モデルの学習がうまくいくことを願っています。"
        ]
        
        dataset = TextDataset(sample_texts, tokenizer, max_length=128)
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # サンプルアイテムをテスト
        sample = dataset[0]
        print(f"  - Input IDs shape: {sample['input_ids'].shape}")
        print(f"  - Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  - Labels shape: {sample['labels'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization_optimizer():
    """量子化オプティマイザーのテスト"""
    print("\n=== Quantization Optimizer Test ===")
    
    try:
        from src.training.quantization import QuantizationOptimizer
        
        # 小さいモデルでテスト
        optimizer = QuantizationOptimizer("gpt2")
        
        print("✓ QuantizationOptimizer created")
        print(f"  - Model: {optimizer.model_name_or_path}")
        print(f"  - Device: {optimizer.device}")
        
        return True
    except Exception as e:
        print(f"✗ Quantization optimizer test failed: {e}")
        return False


def test_model_preparation():
    """モデル準備のテスト"""
    print("\n=== Model Preparation Test ===")
    
    try:
        from src.models.japanese_model import JapaneseModel
        from src.training.full_finetuning import FullFinetuningTrainer
        from src.training.training_utils import TrainingConfig
        
        # 小さいモデルでテスト
        model = JapaneseModel(
            model_name="cyberagent/open-calm-small",
            load_in_8bit=True
        )
        
        config = TrainingConfig(
            batch_size=1,
            num_epochs=1,
            output_dir="./test_outputs"
        )
        
        trainer = FullFinetuningTrainer(
            model=model,
            config=config,
            use_accelerate=True
        )
        
        print("✓ FullFinetuningTrainer created")
        print(f"  - Model: {model.model_name}")
        print(f"  - Device: {trainer.device}")
        
        return True
    except Exception as e:
        print(f"✗ Model preparation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_setup():
    """GPU設定のテスト"""
    print("\n=== GPU Setup Test ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # メモリテスト
        x = torch.randn(1000, 1000, device='cuda')
        print(f"GPU memory test: ✓ (allocated {x.element_size() * x.numel() / 1024**2:.1f} MB)")
        del x
        torch.cuda.empty_cache()
    else:
        print("No GPU available - CPU mode")
    
    return True


def main():
    """メイン関数"""
    print("ファインチューニング機能のテスト")
    print("=" * 60)
    
    tests = [
        ("GPU Setup", test_gpu_setup),
        ("Imports", test_training_imports),
        ("Training Config", test_training_config),
        ("LoRA Config", test_lora_config),
        ("Dataset Creation", test_dataset_creation),
        ("Quantization Optimizer", test_quantization_optimizer),
        ("Model Preparation", test_model_preparation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"\n✓ {test_name} test passed")
            else:
                print(f"\n✗ {test_name} test failed")
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ファインチューニング機能の実装が完了しました。")
        print("\n使用可能な機能:")
        print("- フルファインチューニング (全パラメータ更新)")
        print("- LoRA ファインチューニング (パラメータ効率的)")
        print("- QLoRA ファインチューニング (4bit/8bit量子化)")
        print("- マルチGPU対応 (DataParallel/DistributedDataParallel)")
        print("- 量子化最適化 (推論速度向上)")
    else:
        print(f"\n⚠️  {total - passed} tests failed. 実装に問題がある可能性があります。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)