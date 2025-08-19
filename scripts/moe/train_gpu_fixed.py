#!/usr/bin/env python3
"""
GPU対応MoEトレーニングスクリプト（修正版）
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, '/workspace')

from src.moe.moe_architecture import CivilEngineeringMoEModel, MoEConfig
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class SimpleDataset(Dataset):
    """シンプルなデータセット"""
    def __init__(self, num_samples=100, seq_len=64, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 安全な範囲のinput_ids生成（語彙サイズの半分まで）
        input_ids = torch.randint(0, self.vocab_size // 2, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len)
        # ダミーのターゲット（next token prediction用）
        labels = torch.roll(input_ids, -1)
        labels[-1] = 0  # 最後のトークンは0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_on_gpu():
    print("=" * 60)
    print("GPU MoE Training - Fixed Version")
    print("=" * 60)
    
    # GPU設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # モデル設定（小規模）
    config = MoEConfig(
        hidden_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        expert_capacity_factor=1.25,
        domain_specific_routing=False,  # シンプルにするため無効化
        aux_loss_coef=0.01
    )
    
    # モデル作成
    model = CivilEngineeringMoEModel(config, base_model=None)
    model = model.to(device)
    
    # 出力層を追加（語彙予測用）
    lm_head = nn.Linear(config.hidden_size, 32000, bias=False).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"LM Head Parameters: {sum(p.numel() for p in lm_head.parameters()):,}")
    
    # データセット
    train_dataset = SimpleDataset(num_samples=50, seq_len=32)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # オプティマイザ
    all_params = list(model.parameters()) + list(lm_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    
    # トレーニング
    model.train()
    lm_head.train()
    
    print("\nStarting training...")
    print("-" * 40)
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # データをGPUに移動
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            try:
                # MoEモデルのforward
                outputs = model(input_ids, attention_mask)
                hidden_states = outputs['hidden_states']
                aux_loss = outputs['aux_loss']
                
                # 言語モデルヘッドで予測
                logits = lm_head(hidden_states)
                
                # Cross entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                # 総損失（メイン損失 + 補助損失）
                total_batch_loss = loss + aux_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_batch_loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx:3d}: "
                          f"Loss = {total_batch_loss.item():.4f} "
                          f"(LM: {loss.item():.4f}, Aux: {aux_loss.item():.4f})")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # GPU メモリ状況
        if device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
    
    print("\n" + "=" * 60)
    print("✅ GPU Training completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # CUDA環境変数設定
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 非同期実行を有効化
    
    # キャッシュクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    success = train_on_gpu()
    
    if not success:
        print("\n❌ Training failed")
        sys.exit(1)
