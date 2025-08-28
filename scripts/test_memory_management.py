#!/usr/bin/env python3
"""
統一メモリ管理システムのテストスクリプト
メモリ管理と量子化設定の動作確認
"""

import sys
import torch
import logging
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

from src.core.memory_manager import get_memory_manager, MemoryManager
from src.core.quantization_manager import (
    get_quantization_config_manager,
    UnifiedQuantizationConfig
)
from app.memory_optimized_loader_v2 import create_model_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_memory_manager():
    """メモリマネージャーのテスト"""
    print("\n" + "="*50)
    print("Testing Memory Manager")
    print("="*50)
    
    # 本番環境モード（デフォルト）
    memory_manager = get_memory_manager(debug_mode=False)
    
    # GPUメモリ情報の取得
    if torch.cuda.is_available():
        gpu_info = memory_manager.get_gpu_memory_info()
        if gpu_info:
            print(f"\nGPU Memory Info:")
            print(f"  Total: {gpu_info.total_gb:.2f} GB")
            print(f"  Allocated: {gpu_info.allocated_gb:.2f} GB")
            print(f"  Free: {gpu_info.free_gb:.2f} GB")
            print(f"  Utilization: {gpu_info.utilization_percent:.1f}%")
    else:
        print("GPU not available")
    
    # メモリ監視
    status = memory_manager.monitor_memory_usage()
    print(f"\nMemory Status:")
    print(f"  GPU: {status.get('gpu', 'N/A')}")
    print(f"  CPU: {status.get('cpu', 'N/A')}")
    
    if status.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in status['recommendations']:
            print(f"  - {rec}")
    
    # メモリクリアのテスト
    print("\nTesting memory clear...")
    memory_manager.clear_gpu_memory(aggressive=False)
    print("Memory cleared (normal)")
    
    memory_manager.clear_gpu_memory(aggressive=True)
    print("Memory cleared (aggressive)")
    
    # モデルサイズの推定テスト
    test_models = [
        "llama-3b",
        "mistral-7b",
        "qwen-14b",
        "calm3-22b",
        "DeepSeek-32b"
    ]
    
    print(f"\nModel Size Estimation:")
    for model_name in test_models:
        size = memory_manager.get_model_size(model_name)
        print(f"  {model_name}: {size.value}")
    
    return True


def test_quantization_configs():
    """量子化設定のテスト"""
    print("\n" + "="*50)
    print("Testing Quantization Configurations")
    print("="*50)
    
    memory_manager = get_memory_manager(debug_mode=False)
    
    # 異なるシナリオでの量子化設定テスト
    test_scenarios = [
        {"model": "llama-3b", "memory_gb": 16, "training": False},
        {"model": "mistral-7b", "memory_gb": 8, "training": False},
        {"model": "qwen-14b", "memory_gb": 24, "training": False},
        {"model": "calm3-22b", "memory_gb": 16, "training": False},
        {"model": "calm3-22b", "memory_gb": 48, "training": True},
        {"model": "DeepSeek-32b", "memory_gb": 24, "training": False},
    ]
    
    for scenario in test_scenarios:
        config = memory_manager.get_optimal_quantization(
            model_name=scenario["model"],
            available_memory_gb=scenario["memory_gb"],
            for_training=scenario["training"]
        )
        
        mode = "Training" if scenario["training"] else "Inference"
        print(f"\n{scenario['model']} ({scenario['memory_gb']}GB, {mode}):")
        print(f"  Quantization: {config.type.value}")
        print(f"  Compute dtype: {config.compute_dtype}")
        print(f"  Double quant: {config.use_double_quant}")
        print(f"  CPU offload: {config.use_cpu_offload}")
        print(f"  Device map: {config.device_map}")
    
    return True


def test_quantization_manager():
    """量子化マネージャーのテスト"""
    print("\n" + "="*50)
    print("Testing Quantization Manager")
    print("="*50)
    
    manager = get_quantization_config_manager()
    
    # プリセットのテスト
    print("\nAvailable Presets:")
    for preset_name in ["cpu", "fp16", "int8", "int4", "int4_double", "int4_offload"]:
        preset = manager.get_preset(preset_name)
        if preset:
            print(f"  {preset_name}: {preset.load_in_4bit=}, {preset.load_in_8bit=}")
    
    # 自動設定のテスト
    print("\nAuto Configuration:")
    
    test_cases = [
        ("mistral-7b", None, False, None),
        ("calm3-22b", 16, False, None),
        ("calm3-22b", 16, True, None),
        ("qwen-14b", None, False, "int8"),
    ]
    
    for model_name, memory_gb, for_training, preset in test_cases:
        config = manager.get_config_for_model(
            model_name,
            memory_gb,
            for_training,
            preset
        )
        
        print(f"\n  Model: {model_name}")
        print(f"    Memory: {memory_gb or 'Auto'} GB")
        print(f"    Training: {for_training}")
        print(f"    Preset: {preset or 'None'}")
        print(f"    -> 4bit: {config.load_in_4bit}, 8bit: {config.load_in_8bit}")
    
    # 設定の保存と読み込みテスト
    print("\nSave/Load Test:")
    test_config = UnifiedQuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    manager.save_config(test_config, "test_config")
    print(f"  Saved config: test_config")
    
    loaded_config = manager.load_config("test_config")
    print(f"  Loaded config: 4bit={loaded_config.load_in_4bit}, type={loaded_config.bnb_4bit_quant_type}")
    
    saved_configs = manager.list_saved_configs()
    print(f"  All saved configs: {saved_configs}")
    
    return True


def test_model_loader():
    """モデルローダーのテスト（実際のモデル読み込みはスキップ）"""
    print("\n" + "="*50)
    print("Testing Model Loader")
    print("="*50)
    
    loader = create_model_loader(debug_mode=False)
    
    # メモリ要件の推定テスト
    print("\nMemory Requirements Estimation:")
    
    test_cases = [
        ("mistral-7b", 1, 2048, False),
        ("mistral-7b", 4, 2048, False),
        ("calm3-22b", 1, 2048, False),
        ("calm3-22b", 1, 2048, True),
    ]
    
    for model_name, batch_size, seq_len, for_training in test_cases:
        requirements = loader.estimate_memory_requirements(
            model_name,
            batch_size,
            seq_len,
            for_training
        )
        
        mode = "Training" if for_training else "Inference"
        print(f"\n{model_name} (BS={batch_size}, Seq={seq_len}, {mode}):")
        
        if for_training:
            print(f"  Model: {requirements.get('model', 0):.1f} GB")
            print(f"  Gradients: {requirements.get('gradients', 0):.1f} GB")
            print(f"  Optimizer: {requirements.get('optimizer', 0):.1f} GB")
            print(f"  Activations: {requirements.get('activations', 0):.1f} GB")
        else:
            print(f"  Model: {requirements.get('model', 0):.1f} GB")
            print(f"  Activations: {requirements.get('activations', 0):.1f} GB")
        
        print(f"  Total: {requirements.get('total', 0):.1f} GB")
        print(f"  Recommendation: {requirements.get('recommended_quantization', 'N/A')}")
    
    return True


def test_training_memory_requirements():
    """トレーニング時のメモリ要件テスト"""
    print("\n" + "="*50)
    print("Testing Training Memory Requirements")
    print("="*50)
    
    memory_manager = get_memory_manager(debug_mode=False)
    
    test_configs = [
        {"model": "mistral-7b", "batch_size": 1, "seq_length": 2048, "grad_accum": 1},
        {"model": "mistral-7b", "batch_size": 1, "seq_length": 2048, "grad_accum": 16},
        {"model": "calm3-22b", "batch_size": 1, "seq_length": 1024, "grad_accum": 8},
        {"model": "DeepSeek-32b", "batch_size": 1, "seq_length": 512, "grad_accum": 32},
    ]
    
    for config in test_configs:
        requirements = memory_manager.get_training_memory_requirements(
            model_name=config["model"],
            batch_size=config["batch_size"],
            sequence_length=config["seq_length"],
            gradient_accumulation_steps=config["grad_accum"]
        )
        
        print(f"\n{config['model']}:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Sequence Length: {config['seq_length']}")
        print(f"  Gradient Accumulation: {config['grad_accum']}")
        print(f"  Effective Batch Size: {config['batch_size'] * config['grad_accum']}")
        print(f"\n  Memory Requirements:")
        print(f"    Model: {requirements['model']:.1f} GB")
        print(f"    Gradients: {requirements['gradients']:.1f} GB")
        print(f"    Optimizer: {requirements['optimizer']:.1f} GB")
        print(f"    Activations: {requirements['activations']:.1f} GB")
        print(f"    Total: {requirements['total']:.1f} GB")
        print(f"  Recommendation: {requirements['recommended_quantization']}")
    
    return True


def main():
    """メインテスト実行"""
    print("="*60)
    print("Unified Memory Management System Test Suite")
    print("="*60)
    
    tests = [
        ("Memory Manager", test_memory_manager),
        ("Quantization Configs", test_quantization_configs),
        ("Quantization Manager", test_quantization_manager),
        ("Model Loader", test_model_loader),
        ("Training Memory Requirements", test_training_memory_requirements),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"Running test: {test_name}")
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {e}")
            results.append((test_name, "ERROR"))
    
    # 結果サマリー
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, result in results:
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    # 全体の成功判定
    all_passed = all(result == "PASS" for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
