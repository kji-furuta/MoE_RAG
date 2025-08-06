"""
メモリ効率的なFisher行列管理
大規模モデルでの継続学習を可能にする最適化実装
"""
import torch
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, List, Optional, Tuple
import logging
import gc
import psutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EfficientFisherManager:
    """メモリ効率的なFisher行列管理"""
    
    def __init__(self, storage_path: str = "outputs/ewc_data/fisher"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.block_info = {}  # ブロック情報を記録
        
    def compute_fisher_blockwise(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        task_name: str,
        block_size: int = 1000000,
        max_batches: int = 100
    ) -> Path:
        """ブロック単位でFisher行列を計算（メモリ効率化）"""
        logger.info(f"Computing Fisher matrix blockwise for task: {task_name}")
        logger.info(f"Block size: {block_size:,} parameters")
        
        model.eval()
        
        # タスク用のディレクトリを作成
        task_path = self.storage_path / task_name
        task_path.mkdir(exist_ok=True)
        
        # パラメータをブロックに分割
        param_blocks = self._create_parameter_blocks(model, block_size)
        logger.info(f"Created {len(param_blocks)} parameter blocks")
        
        # メモリ使用状況の記録
        self._log_memory_usage("Before Fisher computation")
        
        # 各ブロックの処理
        for block_idx, block_params in enumerate(param_blocks):
            logger.info(f"Processing block {block_idx + 1}/{len(param_blocks)}")
            
            # ブロックごとのFisher計算
            block_fisher = self._compute_block_fisher(
                model, dataloader, block_params, max_batches
            )
            
            # HDF5形式で保存（圧縮）
            self._save_fisher_block(task_path, block_idx, block_fisher)
            
            # ブロック情報を記録
            self.block_info[f"block_{block_idx}"] = {
                "params": list(block_fisher.keys()),
                "size": sum(f.numel() for f in block_fisher.values())
            }
            
            # メモリ解放
            del block_fisher
            torch.cuda.empty_cache()
            gc.collect()
            
            self._log_memory_usage(f"After block {block_idx + 1}")
        
        # ブロック情報を保存
        self._save_block_info(task_path)
        
        logger.info(f"Fisher matrix computation completed for task: {task_name}")
        return task_path
    
    def _create_parameter_blocks(
        self, 
        model: torch.nn.Module, 
        block_size: int
    ) -> List[List[Tuple[str, torch.nn.Parameter]]]:
        """パラメータをブロックに分割"""
        blocks = []
        current_block = []
        current_size = 0
        
        # レイヤーごとにグループ化してからブロック化（局所性を保つ）
        layer_groups = self._group_parameters_by_layer(model)
        
        for layer_name, params in layer_groups.items():
            for param_name, param in params:
                if not param.requires_grad:
                    continue
                    
                param_size = param.numel()
                
                # 現在のブロックに追加するとサイズを超える場合
                if current_size + param_size > block_size and current_block:
                    blocks.append(current_block)
                    current_block = []
                    current_size = 0
                
                current_block.append((param_name, param))
                current_size += param_size
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _group_parameters_by_layer(self, model: torch.nn.Module) -> Dict[str, List]:
        """パラメータをレイヤーごとにグループ化"""
        layer_groups = {}
        
        for name, param in model.named_parameters():
            # レイヤー名を抽出（例: "transformer.h.0.attn.weight" -> "transformer.h.0"）
            parts = name.split('.')
            if len(parts) >= 3:
                layer_name = '.'.join(parts[:3])
            else:
                layer_name = parts[0]
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = []
            
            layer_groups[layer_name].append((name, param))
        
        return layer_groups
    
    def _compute_block_fisher(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        block_params: List[Tuple[str, torch.nn.Parameter]],
        max_batches: int
    ) -> Dict[str, torch.Tensor]:
        """特定のパラメータブロックのFisher行列を計算"""
        device = next(model.parameters()).device
        block_fisher = {}
        
        # Fisher行列の初期化（CPU上で）
        for param_name, param in block_params:
            block_fisher[param_name] = torch.zeros_like(param, device='cpu')
        
        # パラメータ名のセット（高速検索用）
        block_param_names = set(pn for pn, _ in block_params)
        
        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher", total=max_batches)):
            if batch_idx >= max_batches:
                break
                
            # バッチをデバイスに転送
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 勾配の計算
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # ブロック内のパラメータのみ処理
            for name, param in model.named_parameters():
                if name in block_param_names and param.grad is not None:
                    # Fisher行列の更新（勾配の二乗）
                    fisher_update = param.grad.data.pow(2).cpu()
                    block_fisher[name] += fisher_update
            
            batch_count += 1
            
            # 定期的なメモリクリア
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 平均化
        if batch_count > 0:
            for param_name in block_fisher:
                block_fisher[param_name] /= batch_count
        
        return block_fisher
    
    def _save_fisher_block(
        self, 
        task_path: Path,
        block_idx: int, 
        fisher_data: Dict[str, torch.Tensor]
    ):
        """Fisher行列ブロックをHDF5形式で保存"""
        filepath = task_path / f"fisher_block_{block_idx}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # 圧縮オプションの設定
            for param_name, fisher_matrix in fisher_data.items():
                # パラメータ名をHDF5互換に変換
                safe_name = param_name.replace('.', '_')
                
                # numpy配列に変換して保存
                f.create_dataset(
                    safe_name,
                    data=fisher_matrix.numpy(),
                    compression='gzip',
                    compression_opts=9  # 最大圧縮
                )
                
                # メタデータの保存
                f[safe_name].attrs['original_name'] = param_name
                f[safe_name].attrs['shape'] = fisher_matrix.shape
        
        logger.info(f"Saved Fisher block to: {filepath}")
    
    def _save_block_info(self, task_path: Path):
        """ブロック情報を保存"""
        import json
        
        info_path = task_path / "block_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.block_info, f, indent=2)
    
    def load_fisher_matrices(self, task_names: List[str]) -> List[Dict[str, torch.Tensor]]:
        """複数タスクのFisher行列を効率的にロード"""
        logger.info(f"Loading Fisher matrices for tasks: {task_names}")
        fisher_matrices = []
        
        for task_name in task_names:
            task_fisher = self._load_task_fisher(task_name)
            if task_fisher:
                fisher_matrices.append(task_fisher)
        
        return fisher_matrices
    
    def _load_task_fisher(self, task_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """特定タスクのFisher行列をロード"""
        task_path = self.storage_path / task_name
        
        if not task_path.exists():
            logger.warning(f"Fisher matrix not found for task: {task_name}")
            return None
        
        task_fisher = {}
        
        # すべてのHDF5ファイルを読み込み
        for h5_file in sorted(task_path.glob("fisher_block_*.h5")):
            logger.info(f"Loading: {h5_file}")
            
            with h5py.File(h5_file, 'r') as f:
                for dataset_name in f.keys():
                    # 元のパラメータ名を復元
                    original_name = f[dataset_name].attrs['original_name']
                    
                    # データをロード（必要に応じてGPUに転送）
                    fisher_data = torch.from_numpy(f[dataset_name][:])
                    task_fisher[original_name] = fisher_data
        
        logger.info(f"Loaded {len(task_fisher)} parameters for task: {task_name}")
        return task_fisher
    
    def compute_compressed_fisher(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task_name: str,
        compression_ratio: float = 0.1
    ) -> Path:
        """圧縮されたFisher行列を計算（低ランク近似）"""
        logger.info(f"Computing compressed Fisher matrix (ratio: {compression_ratio})")
        
        # 通常のFisher行列を計算
        fisher_path = self.compute_fisher_blockwise(
            model, dataloader, task_name, max_batches=50
        )
        
        # 低ランク近似を適用
        compressed_path = self._apply_low_rank_approximation(
            fisher_path, compression_ratio
        )
        
        return compressed_path
    
    def _apply_low_rank_approximation(
        self, 
        fisher_path: Path, 
        compression_ratio: float
    ) -> Path:
        """Fisher行列に低ランク近似を適用"""
        compressed_path = fisher_path.parent / f"{fisher_path.name}_compressed"
        compressed_path.mkdir(exist_ok=True)
        
        # 各ブロックファイルを処理
        for h5_file in fisher_path.glob("fisher_block_*.h5"):
            with h5py.File(h5_file, 'r') as f_in:
                out_file = compressed_path / h5_file.name
                
                with h5py.File(out_file, 'w') as f_out:
                    for dataset_name in f_in.keys():
                        fisher_matrix = f_in[dataset_name][:]
                        
                        # 2D行列に変換
                        original_shape = fisher_matrix.shape
                        matrix_2d = fisher_matrix.reshape(-1, 1)
                        
                        # SVD圧縮（簡易版）
                        if matrix_2d.shape[0] > 100:
                            # 対角成分のみ保持（最も単純な圧縮）
                            compressed = np.diag(matrix_2d.flatten())
                            keep_dims = int(compressed.shape[0] * compression_ratio)
                            compressed = compressed[:keep_dims, :keep_dims]
                        else:
                            compressed = matrix_2d
                        
                        # 保存
                        f_out.create_dataset(
                            dataset_name,
                            data=compressed,
                            compression='gzip'
                        )
                        f_out[dataset_name].attrs['original_shape'] = original_shape
                        f_out[dataset_name].attrs['compression_ratio'] = compression_ratio
        
        return compressed_path
    
    def _log_memory_usage(self, context: str = ""):
        """メモリ使用状況をログ出力"""
        # CPU メモリ
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        # GPU メモリ
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
            
            logger.info(f"{context} - Memory usage: CPU: {cpu_memory:.2f}GB, "
                       f"GPU: {gpu_memory:.2f}GB (cached: {gpu_cached:.2f}GB)")
        else:
            logger.info(f"{context} - Memory usage: CPU: {cpu_memory:.2f}GB")
