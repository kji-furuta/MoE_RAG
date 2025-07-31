import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
import gc
import time
import psutil

def get_memory_usage():
    """メモリ使用量を取得"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
    }

def get_gpu_memory_usage():
    """GPUメモリ使用量を取得"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
            'cached': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
        }
    return {'allocated': 0, 'cached': 0}

class EWCHelper:
    """Elastic Weight Consolidation (EWC) のヘルパークラス"""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        # パラメータをCPUに保存してメモリを節約
        # メタテンソル（device='meta'）はスキップ
        self.params = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.device.type != 'meta':
                try:
                    self.params[n] = p.clone().detach().cpu()
                except Exception as e:
                    print(f"Warning: Cannot clone parameter {n}: {e}")
                    continue
        
        self.fisher_matrix = None
        # GPU メモリをクリア
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"EWCHelper initialized with {len(self.params)} parameters")
        
        # モデルのパラメータ総数と実際に保存されたパラメータ数をチェック
        total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        if len(self.params) < total_params:
            print(f"Warning: Only {len(self.params)}/{total_params} parameters are tracked by EWC.")
            print("Some parameters are on meta device and cannot be used for EWC.")

    def compute_fisher_matrix(self, dataloader: DataLoader, max_batches: int = None):
        """フィッシャー情報行列の対角成分を計算する（最適化版）"""
        print("Starting Fisher matrix computation...")
        start_time = time.time()
        
        # 初期メモリ使用量を記録
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory_usage()
        print(f"Initial memory usage: {initial_memory['rss']:.1f}MB RAM, {initial_gpu_memory['allocated']:.1f}MB GPU")
        
        self.model.eval()
        # Fisher matrixもCPUに保存（メタテンソルはスキップ）
        fisher_matrix = {}
        print("Initializing Fisher matrix...")
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.device.type != 'meta' and n in self.params:
                fisher_matrix[n] = torch.zeros_like(p).cpu()
        print(f"Initialized Fisher matrix for {len(fisher_matrix)} parameters")

        # 最大バッチ数を制限（メモリ節約のため）
        if max_batches is None:
            max_batches = min(len(dataloader), 100)  # デフォルトで最大100バッチ
        
        print(f"Computing Fisher matrix using {max_batches} batches...")
        
        batch_count = 0
        for batch in tqdm(dataloader, desc="Computing Fisher Matrix", total=max_batches):
            if batch_count >= max_batches:
                break
                
            try:
                print(f"Processing batch {batch_count + 1}/{max_batches}")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model.zero_grad()
                
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # 勾配計算
                loss.backward()

                # Fisher matrixの更新（メモリ効率化）
                updated_params = 0
                for n, p in self.model.named_parameters():
                    if p.grad is not None and p.requires_grad and p.device.type != 'meta' and n in fisher_matrix:
                        # 勾配の二乗を計算してCPUに移動
                        grad_squared = p.grad.pow(2)
                        fisher_matrix[n] += grad_squared.detach().cpu()
                        # メモリを即座に解放
                        del grad_squared
                        updated_params += 1
                
                print(f"Updated {updated_params} parameters in batch {batch_count + 1}")
                batch_count += 1
                
                # 定期的にメモリをクリア
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_memory = get_memory_usage()
                    current_gpu_memory = get_gpu_memory_usage()
                    print(f"Memory cleared at batch {batch_count}")
                    print(f"Current memory: {current_memory['rss']:.1f}MB RAM, {current_gpu_memory['allocated']:.1f}MB GPU")
                    
            except Exception as e:
                print(f"Error processing batch {batch_count}: {e}")
                continue
        
        # 平均化
        if batch_count > 0:
            self.fisher_matrix = {n: f / batch_count for n, f in fisher_matrix.items()}
            print(f"Fisher matrix computation completed in {time.time() - start_time:.2f} seconds")
            print(f"Processed {batch_count} batches successfully")
        else:
            print("Warning: No batches were processed successfully")
            self.fisher_matrix = fisher_matrix
        
        self.model.train()
        # 最終クリーンアップ
        torch.cuda.empty_cache()
        gc.collect()
        
        # 最終メモリ使用量を記録
        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory_usage()
        print(f"Final memory usage: {final_memory['rss']:.1f}MB RAM, {final_gpu_memory['allocated']:.1f}MB GPU")
        print("Fisher matrix computation finished")

    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """EWC損失を計算する（最適化版）"""
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix has not been computed. Call compute_fisher_matrix first.")

        ewc_loss = 0
        for n, p in model.named_parameters():
            # メタテンソルはスキップ
            if p.requires_grad and n in self.params and p.device.type != 'meta' and n in self.fisher_matrix:
                # Fisher matrixとparameterをGPUに移して計算
                fisher = self.fisher_matrix[n].to(self.device)
                old_param = self.params[n].to(self.device)
                ewc_loss += (fisher * (p - old_param).pow(2)).sum()
                # 計算後すぐにCPUに戻す
                del fisher, old_param
        
        torch.cuda.empty_cache()
        return ewc_loss