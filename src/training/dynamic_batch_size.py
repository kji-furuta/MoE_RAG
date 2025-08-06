"""
動的バッチサイズ調整
メモリ使用状況に応じてバッチサイズを自動調整
"""
import torch
import gc
import psutil
from typing import Optional, Tuple, Dict
import logging
import time

logger = logging.getLogger(__name__)


class DynamicBatchSizeManager:
    """動的バッチサイズ管理"""
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        target_memory_usage: float = 0.8,  # GPUメモリの80%を目標
        adjustment_factor: float = 0.5     # 調整時の変更率
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.adjustment_factor = adjustment_factor
        
        # メモリ使用履歴
        self.memory_history = []
        self.batch_size_history = []
        self.oom_count = 0
        
        # GPU情報
        if torch.cuda.is_available():
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU total memory: {self.gpu_total_memory / 1e9:.2f} GB")
        else:
            self.gpu_total_memory = 0
    
    def adjust_batch_size(self, memory_usage: Optional[float] = None) -> int:
        """現在のメモリ使用状況に基づいてバッチサイズを調整"""
        if memory_usage is None:
            memory_usage = self.get_current_memory_usage()
        
        self.memory_history.append(memory_usage)
        self.batch_size_history.append(self.current_batch_size)
        
        # メモリ使用率に基づく調整
        if memory_usage > self.target_memory_usage + 0.1:
            # メモリ使用率が高すぎる場合は減らす
            self._decrease_batch_size()
            logger.info(f"Decreased batch size to {self.current_batch_size} "
                       f"(memory usage: {memory_usage:.2%})")
        elif memory_usage < self.target_memory_usage - 0.2:
            # メモリ使用率が低い場合は増やす
            self._increase_batch_size()
            logger.info(f"Increased batch size to {self.current_batch_size} "
                       f"(memory usage: {memory_usage:.2%})")
        
        return self.current_batch_size
    
    def handle_oom(self) -> int:
        """OOM（Out of Memory）エラーを処理"""
        self.oom_count += 1
        logger.warning(f"OOM error #{self.oom_count} - reducing batch size")
        
        # より積極的にバッチサイズを削減
        self.current_batch_size = max(
            self.min_batch_size,
            int(self.current_batch_size * 0.5)
        )
        
        # メモリをクリア
        self.clear_memory()
        
        # 少し待機（GPUが回復するまで）
        time.sleep(2)
        
        return self.current_batch_size
    
    def _decrease_batch_size(self):
        """バッチサイズを減少"""
        new_size = int(self.current_batch_size * (1 - self.adjustment_factor))
        self.current_batch_size = max(self.min_batch_size, new_size)
    
    def _increase_batch_size(self):
        """バッチサイズを増加"""
        new_size = int(self.current_batch_size * (1 + self.adjustment_factor))
        self.current_batch_size = min(self.max_batch_size, new_size)
    
    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用率を取得"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            return allocated / self.gpu_total_memory
        return 0.0
    
    def clear_memory(self):
        """メモリをクリア"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_batch: Dict[str, torch.Tensor],
        max_trials: int = 5
    ) -> int:
        """モデルとサンプルバッチから最適なバッチサイズを推定"""
        logger.info("Estimating optimal batch size...")
        
        device = next(model.parameters()).device
        optimal_size = self.min_batch_size
        
        # バイナリサーチで最適なバッチサイズを探索
        low, high = self.min_batch_size, self.max_batch_size
        
        while low <= high and max_trials > 0:
            mid = (low + high) // 2
            
            try:
                # テストバッチの作成
                test_batch = self._create_test_batch(sample_batch, mid, device)
                
                # メモリクリア
                self.clear_memory()
                
                # フォワードパスとバックワードパスのテスト
                model.zero_grad()
                outputs = model(**test_batch)
                loss = outputs.loss
                loss.backward()
                
                # メモリ使用率の確認
                memory_usage = self.get_current_memory_usage()
                logger.info(f"Batch size {mid}: memory usage {memory_usage:.2%}")
                
                if memory_usage < self.target_memory_usage:
                    optimal_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM with batch size {mid}")
                    high = mid - 1
                    self.clear_memory()
                else:
                    raise e
            
            max_trials -= 1
        
        # 安全マージンを考慮
        self.current_batch_size = int(optimal_size * 0.9)
        logger.info(f"Optimal batch size: {self.current_batch_size}")
        
        return self.current_batch_size
    
    def _create_test_batch(
        self,
        sample_batch: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """テスト用のバッチを作成"""
        test_batch = {}
        
        for key, tensor in sample_batch.items():
            # バッチサイズ分だけ複製
            if tensor.dim() > 0:
                repeat_dims = [batch_size] + [1] * (tensor.dim() - 1)
                test_batch[key] = tensor.repeat(*repeat_dims).to(device)
            else:
                test_batch[key] = tensor.to(device)
        
        return test_batch
    
    def get_statistics(self) -> Dict[str, float]:
        """バッチサイズ調整の統計情報を取得"""
        if not self.memory_history:
            return {}
        
        return {
            "avg_memory_usage": sum(self.memory_history) / len(self.memory_history),
            "max_memory_usage": max(self.memory_history),
            "min_batch_size_used": min(self.batch_size_history) if self.batch_size_history else 0,
            "max_batch_size_used": max(self.batch_size_history) if self.batch_size_history else 0,
            "avg_batch_size": sum(self.batch_size_history) / len(self.batch_size_history) if self.batch_size_history else 0,
            "oom_count": self.oom_count,
            "current_batch_size": self.current_batch_size
        }


class AdaptiveDataLoader:
    """動的バッチサイズに対応したデータローダー"""
    
    def __init__(
        self,
        dataset,
        batch_size_manager: DynamicBatchSizeManager,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        self.dataset = dataset
        self.batch_size_manager = batch_size_manager
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.current_batch_size = batch_size_manager.current_batch_size
        
    def __iter__(self):
        """動的にバッチサイズを調整しながらイテレート"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        current_idx = 0
        
        while current_idx < len(indices):
            # 現在のバッチサイズを取得
            batch_size = self.batch_size_manager.current_batch_size
            
            # バッチのインデックスを取得
            batch_indices = indices[current_idx:current_idx + batch_size]
            
            # バッチデータを作成
            batch_data = self._collate_batch([self.dataset[i] for i in batch_indices])
            
            # メモリ使用状況を監視
            memory_usage = self.batch_size_manager.get_current_memory_usage()
            
            # 次のイテレーション用にバッチサイズを調整
            self.batch_size_manager.adjust_batch_size(memory_usage)
            
            yield batch_data
            
            current_idx += len(batch_indices)
    
    def _collate_batch(self, batch_items):
        """バッチデータを整形"""
        # 簡易的な実装（実際にはより複雑な処理が必要）
        keys = batch_items[0].keys()
        batch = {}
        
        for key in keys:
            values = [item[key] for item in batch_items]
            
            # テンソルの場合はスタック
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        
        return batch
    
    def __len__(self):
        """データローダーの長さ（動的なので概算）"""
        return len(self.dataset) // self.batch_size_manager.current_batch_size
