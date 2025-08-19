#!/usr/bin/env python3
"""
境界値問題修正の検証スクリプト
"""

import torch
import sys
sys.path.insert(0, '/workspace')

def verify_fixes():
    print("=" * 60)
    print("Tokenizer Boundary Fix Verification")
    print("=" * 60)
    
    # 1. Tokenizer検証
    print("\n1. Testing Safe Tokenizer:")
    print("-" * 40)
    
    class SafeTokenizer:
        vocab_size = 32000
        safe_vocab_size = 30000
        
        def __call__(self, text, **kwargs):
            max_length = kwargs.get('max_length', 100)
            return {
                'input_ids': torch.randint(11, self.safe_vocab_size, (1, max_length)),
                'attention_mask': torch.ones(1, max_length)
            }
        
        def __len__(self):
            return self.vocab_size
    
    tokenizer = SafeTokenizer()
    
    # 100回のサンプリングで最大値を確認
    max_values = []
    for _ in range(100):
        sample = tokenizer("test text", max_length=128)
        max_values.append(sample['input_ids'].max().item())
    
    print(f"✓ Tokenizer max value over 100 samples: {max(max_values)}")
    print(f"  Safe limit: {tokenizer.safe_vocab_size - 1}")
    print(f"  Actual vocab size: {tokenizer.vocab_size}")
    print(f"  Safety margin: {(1 - max(max_values)/tokenizer.vocab_size) * 100:.1f}%")
    
    # 2. Model検証
    print("\n2. Testing Model with Boundary Values:")
    print("-" * 40)
    
    from src.moe.moe_architecture import CivilEngineeringMoEModel, MoEConfig
    
    config = MoEConfig(
        hidden_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        domain_specific_routing=False
    )
    
    model = CivilEngineeringMoEModel(config, base_model=None)
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # 境界値テスト
    test_cases = [
        ("Safe range", torch.tensor([[100, 1000, 10000, 29999]]).to(device)),
        ("Near boundary", torch.tensor([[30000, 30500, 31000, 31500]]).to(device)),
        ("At boundary", torch.tensor([[31998, 31999, 32000, 32001]]).to(device))
    ]
    
    for name, input_ids in test_cases:
        try:
            # 入力前の最大値
            max_before = input_ids.max().item()
            
            with torch.no_grad():
                outputs = model(input_ids)
            
            print(f"✓ {name}: max_input={max_before} → Success")
            
        except Exception as e:
            print(f"✗ {name}: max_input={max_before} → Error: {str(e)[:50]}")
    
    # 3. Full Training Test
    print("\n3. Testing Full Training Pipeline:")
    print("-" * 40)
    
    from torch.utils.data import DataLoader, Dataset
    
    class TestDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
            self.tokenizer = SafeTokenizer()
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            sample = self.tokenizer("test", max_length=64)
            return {
                'input_ids': sample['input_ids'].squeeze(),
                'attention_mask': sample['attention_mask'].squeeze(),
                'labels': sample['input_ids'].squeeze()
            }
    
    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=2)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    success_count = 0
    for batch in dataloader:
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = outputs['aux_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Batch failed: {e}")
            break
    
    if success_count == len(dataloader):
        print(f"✅ All {success_count} batches processed successfully!")
    else:
        print(f"⚠️ Only {success_count}/{len(dataloader)} batches succeeded")
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)

if __name__ == "__main__":
    verify_fixes()
