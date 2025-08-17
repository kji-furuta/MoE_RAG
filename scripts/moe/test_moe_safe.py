#!/usr/bin/env python3
'''
修正後のMoEモデルの安全なテスト
'''
import torch
import sys
sys.path.insert(0, '/workspace')

from src.moe.moe_architecture import CivilEngineeringMoEModel, MoEConfig

def test_safe_moe():
    print("=" * 60)
    print("Safe MoE Test - 修正版")
    print("=" * 60)
    
    # 最小構成でテスト
    config = MoEConfig(
        hidden_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        expert_capacity_factor=1.25,
        domain_specific_routing=True
    )
    
    try:
        # モデル作成
        model = CivilEngineeringMoEModel(config, base_model=None)
        print(f"✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # GPU移動
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
            print(f"✓ Model moved to GPU")
        else:
            device = 'cpu'
            print(f"! Running on CPU")
        
        # 安全な入力データ作成
        batch_size = 2
        seq_len = 32
        vocab_size = 32000  # embedding層と一致
        
        # input_idsを確実に範囲内に
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        print(f"\n✓ Input data created")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        print(f"  vocab_size: {vocab_size}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {outputs['hidden_states'].shape}")
        print(f"  Aux loss: {outputs['aux_loss'].item():.6f}")
        
        # メモリ使用量
        if torch.cuda.is_available():
            print(f"\n📊 GPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_moe()
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed! MoE is ready for training.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed. Please check the errors above.")
        print("=" * 60)
