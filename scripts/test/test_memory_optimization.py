#!/usr/bin/env python3
"""
メモリ効率的な継続学習のテストスクリプト
"""
import sys
import os
import torch
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.efficient_fisher_manager import EfficientFisherManager
from src.training.dynamic_batch_size import DynamicBatchSizeManager
from src.training.memory_profiler import MemoryProfiler, MemoryOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_efficient_fisher_manager():
    """効率的なFisher行列管理のテスト"""
    logger.info("Testing EfficientFisherManager...")
    
    # ダミーモデルの作成
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(100, 50)
            self.layer2 = torch.nn.Linear(50, 10)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    model = DummyModel()
    
    # ダミーデータローダー
    class DummyDataLoader:
        def __init__(self, size=10):
            self.size = size
            
        def __iter__(self):
            for i in range(self.size):
                yield {
                    'input_ids': torch.randn(4, 100),
                    'labels': torch.randint(0, 10, (4,))
                }
        
        def __len__(self):
            return self.size
    
    # Fisher管理の初期化
    fisher_manager = EfficientFisherManager("outputs/test_fisher")
    
    # Fisher行列の計算
    dataloader = DummyDataLoader()
    fisher_path = fisher_manager.compute_fisher_blockwise(
        model=model,
        dataloader=dataloader,
        task_name="test_task",
        block_size=1000,
        max_batches=5
    )
    
    logger.info(f"Fisher matrix saved to: {fisher_path}")
    
    # Fisher行列のロード
    loaded_fisher = fisher_manager.load_fisher_matrices(["test_task"])
    logger.info(f"Loaded {len(loaded_fisher)} Fisher matrices")
    
    return True


def test_dynamic_batch_size():
    """動的バッチサイズ管理のテスト"""
    logger.info("Testing DynamicBatchSizeManager...")
    
    # マネージャーの初期化
    batch_manager = DynamicBatchSizeManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16,
        target_memory_usage=0.7
    )
    
    # メモリ使用状況のシミュレーション
    memory_usages = [0.5, 0.6, 0.8, 0.9, 0.4, 0.3]
    
    for i, mem_usage in enumerate(memory_usages):
        batch_size = batch_manager.adjust_batch_size(mem_usage)
        logger.info(f"Step {i}: Memory={mem_usage:.1%}, Batch size={batch_size}")
    
    # OOMシミュレーション
    batch_size = batch_manager.handle_oom()
    logger.info(f"After OOM: Batch size={batch_size}")
    
    # 統計情報
    stats = batch_manager.get_statistics()
    logger.info(f"Statistics: {stats}")
    
    return True


def test_memory_profiler():
    """メモリプロファイラーのテスト"""
    logger.info("Testing MemoryProfiler...")
    
    profiler = MemoryProfiler("outputs/test_memory_profile")
    
    # メモリ使用の記録
    profiler.record_memory("Start")
    
    # 何か重い処理のシミュレーション
    data = torch.randn(1000, 1000)
    profiler.record_memory("After allocation")
    
    result = torch.matmul(data, data)
    profiler.record_memory("After computation")
    
    del data, result
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    profiler.record_memory("After cleanup")
    
    # プロファイル結果の保存
    profile_path = profiler.save_profile()
    plot_path = profiler.plot_memory_usage()
    report = profiler.generate_report()
    
    logger.info(f"Profile saved to: {profile_path}")
    logger.info(f"Plot saved to: {plot_path}")
    logger.info("Report:")
    logger.info(report)
    
    return True


def test_memory_optimizer():
    """メモリ最適化のテスト"""
    logger.info("Testing MemoryOptimizer...")
    
    # ダミーモデル
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    # 最適化前のパラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # モデルの最適化
    optimized_model = MemoryOptimizer.optimize_model_for_training(model)
    
    # バッチサイズの推定
    sample_input = {
        'input_ids': torch.randn(100),
        'attention_mask': torch.ones(100)
    }
    
    if torch.cuda.is_available():
        model = model.cuda()
        estimated_batch_size = MemoryOptimizer.estimate_batch_size(model, sample_input)
        logger.info(f"Estimated batch size: {estimated_batch_size}")
    else:
        logger.info("CUDA not available, skipping batch size estimation")
    
    # キャッシュクリア
    MemoryOptimizer.clear_cache_aggressive()
    logger.info("Cache cleared")
    
    return True


def main():
    """すべてのテストを実行"""
    logger.info("Starting memory optimization tests...")
    
    tests = [
        ("EfficientFisherManager", test_efficient_fisher_manager),
        ("DynamicBatchSizeManager", test_dynamic_batch_size),
        ("MemoryProfiler", test_memory_profiler),
        ("MemoryOptimizer", test_memory_optimizer)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
            
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "ERROR"
    
    # 結果のサマリー
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary:")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    # 全体の成功/失敗
    all_passed = all(result == "PASSED" for result in results.values())
    
    if all_passed:
        logger.info("\nAll tests passed! ✅")
        return 0
    else:
        logger.info("\nSome tests failed! ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
