"""
メモリプロファイリングとモニタリング
継続学習中のメモリ使用状況を監視
"""
import torch
import psutil
import GPUtil
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import time
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """メモリ使用状況のプロファイラー"""
    
    def __init__(self, log_dir: str = "logs/memory_profile"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.cpu_history = []
        self.gpu_history = []
        self.timestamps = []
        self.events = []  # 重要なイベントを記録
        
        self.start_time = time.time()
        
    def record_memory(self, event: Optional[str] = None):
        """現在のメモリ使用状況を記録"""
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        # CPU メモリ
        process = psutil.Process()
        cpu_info = {
            'rss': process.memory_info().rss / 1e9,  # GB
            'vms': process.memory_info().vms / 1e9,  # GB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1e9  # GB
        }
        self.cpu_history.append(cpu_info)
        
        # GPU メモリ
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1e9,  # GB
                    'reserved': torch.cuda.memory_reserved(i) / 1e9,  # GB
                    'free': (torch.cuda.get_device_properties(i).total_memory - 
                            torch.cuda.memory_allocated(i)) / 1e9  # GB
                }
                
                # GPUtil を使用してより詳細な情報を取得
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_info[f'gpu_{i}']['temperature'] = gpus[i].temperature
                        gpu_info[f'gpu_{i}']['utilization'] = gpus[i].memoryUtil * 100
                except:
                    pass
        
        self.gpu_history.append(gpu_info)
        
        # イベントの記録
        if event:
            self.events.append({
                'time': current_time,
                'event': event,
                'cpu_rss': cpu_info['rss'],
                'gpu_allocated': gpu_info.get('gpu_0', {}).get('allocated', 0)
            })
            logger.info(f"Memory at {event}: CPU={cpu_info['rss']:.2f}GB, "
                       f"GPU={gpu_info.get('gpu_0', {}).get('allocated', 0):.2f}GB")
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """メモリ使用状況をプロット"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU メモリ
        cpu_rss = [info['rss'] for info in self.cpu_history]
        cpu_available = [info['available'] for info in self.cpu_history]
        
        ax1.plot(self.timestamps, cpu_rss, label='RSS', color='blue')
        ax1.plot(self.timestamps, cpu_available, label='Available', color='green', linestyle='--')
        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('CPU Memory Usage')
        ax1.legend()
        ax1.grid(True)
        
        # イベントをマーク
        for event in self.events:
            ax1.axvline(x=event['time'], color='red', alpha=0.3, linestyle=':')
            ax1.text(event['time'], max(cpu_rss), event['event'], 
                    rotation=90, fontsize=8, verticalalignment='bottom')
        
        # GPU メモリ
        if self.gpu_history and 'gpu_0' in self.gpu_history[0]:
            gpu_allocated = [info.get('gpu_0', {}).get('allocated', 0) for info in self.gpu_history]
            gpu_reserved = [info.get('gpu_0', {}).get('reserved', 0) for info in self.gpu_history]
            
            ax2.plot(self.timestamps, gpu_allocated, label='Allocated', color='red')
            ax2.plot(self.timestamps, gpu_reserved, label='Reserved', color='orange', linestyle='--')
            ax2.set_ylabel('Memory (GB)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('GPU Memory Usage')
            ax2.legend()
            ax2.grid(True)
            
            # イベントをマーク
            for event in self.events:
                ax2.axvline(x=event['time'], color='red', alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Memory usage plot saved to: {save_path}")
        else:
            save_path = self.log_dir / f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300)
        
        plt.close()
        
        return save_path
    
    def save_profile(self):
        """プロファイル結果を保存"""
        profile_data = {
            'timestamps': self.timestamps,
            'cpu_history': self.cpu_history,
            'gpu_history': self.gpu_history,
            'events': self.events,
            'duration': time.time() - self.start_time
        }
        
        save_path = self.log_dir / f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Memory profile saved to: {save_path}")
        return save_path
    
    def get_peak_memory(self) -> Dict[str, float]:
        """ピークメモリ使用量を取得"""
        peak_cpu = max([info['rss'] for info in self.cpu_history]) if self.cpu_history else 0
        
        peak_gpu = 0
        if self.gpu_history and 'gpu_0' in self.gpu_history[0]:
            peak_gpu = max([info.get('gpu_0', {}).get('allocated', 0) for info in self.gpu_history])
        
        return {
            'peak_cpu_gb': peak_cpu,
            'peak_gpu_gb': peak_gpu,
            'events': len(self.events)
        }
    
    def generate_report(self) -> str:
        """メモリ使用レポートを生成"""
        peak_memory = self.get_peak_memory()
        
        report = f"""
Memory Usage Report
==================
Duration: {time.time() - self.start_time:.2f} seconds
Peak CPU Memory: {peak_memory['peak_cpu_gb']:.2f} GB
Peak GPU Memory: {peak_memory['peak_gpu_gb']:.2f} GB
Number of Events: {peak_memory['events']}

Events:
"""
        for event in self.events:
            report += f"  - {event['event']} at {event['time']:.2f}s: "
            report += f"CPU={event['cpu_rss']:.2f}GB, GPU={event['gpu_allocated']:.2f}GB\n"
        
        # レポートを保存
        report_path = self.log_dir / f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


class MemoryOptimizer:
    """メモリ最適化のヘルパー関数"""
    
    @staticmethod
    def optimize_model_for_training(model: torch.nn.Module) -> torch.nn.Module:
        """トレーニング用にモデルを最適化"""
        # 勾配チェックポイントの有効化
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # 不要なバッファをクリア
        for module in model.modules():
            # バッチノーマライゼーションの統計情報をクリア
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        
        return model
    
    @staticmethod
    def clear_cache_aggressive():
        """積極的なキャッシュクリア"""
        # Python ガベージコレクション
        import gc
        gc.collect()
        gc.collect()
        gc.collect()
        
        # PyTorch キャッシュ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # より積極的なクリア
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    
    @staticmethod
    def estimate_batch_size(model: torch.nn.Module, sample_input: Dict[str, torch.Tensor]) -> int:
        """モデルとサンプル入力から推奨バッチサイズを推定"""
        device = next(model.parameters()).device
        
        # 利用可能なメモリを取得
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / 1e9
        else:
            free_memory_gb = psutil.virtual_memory().available / 1e9
        
        # サンプル入力でメモリ使用量を推定
        model.eval()
        with torch.no_grad():
            # 単一サンプルのメモリ使用量を測定
            MemoryOptimizer.clear_cache_aggressive()
            
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # フォワードパス
            sample_batch = {k: v.unsqueeze(0).to(device) for k, v in sample_input.items()}
            _ = model(**sample_batch)
            
            single_sample_memory = (torch.cuda.memory_allocated() - initial_memory) if torch.cuda.is_available() else 1e8
            single_sample_memory_gb = single_sample_memory / 1e9
        
        # 推奨バッチサイズを計算（安全マージン50%）
        recommended_batch_size = int(free_memory_gb * 0.5 / single_sample_memory_gb)
        recommended_batch_size = max(1, min(recommended_batch_size, 32))  # 1-32の範囲
        
        logger.info(f"Estimated batch size: {recommended_batch_size} "
                   f"(free memory: {free_memory_gb:.2f}GB, "
                   f"single sample: {single_sample_memory_gb:.4f}GB)")
        
        model.train()
        return recommended_batch_size
