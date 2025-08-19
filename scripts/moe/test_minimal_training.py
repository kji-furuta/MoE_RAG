#!/usr/bin/env python3
"""
最小限のMoEトレーニングテスト
"""
import torch
import sys
sys.path.insert(0, '/workspace')
from src.moe.moe_architecture import CivilEngineeringMoEModel, MoEConfig

def minimal_training_test():
    print("=" * 60)
    print("Minimal MoE Training Test")
    print("=" * 60)
    
    # 最小構成
    config = MoEConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        domain_specific_routing=False  # ドメインルーティングを無効化
    )
    
    model = CivilEngineeringMoEModel(config, base_model=None)
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Model on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # トレーニングループ（1バッチのみ）
    model.train()
    
    for step in range(3):
        # 安全な入力データ
        batch_size = 1
        seq_len = 32
        vocab_size = 32000
        
        input_ids = torch.randint(0, min(1000, vocab_size), (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # Forward
        try:
            outputs = model(input_ids, attention_mask)
            loss = outputs['aux_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Step {step+1}: Loss = {loss.item():.6f}")
            
        except Exception as e:
            print(f"Error at step {step+1}: {e}")
            return False
    
    print("\n✅ Training test completed successfully!")
    return True

if __name__ == "__main__":
    success = minimal_training_test()
    if not success:
        print("\n❌ Training test failed")
