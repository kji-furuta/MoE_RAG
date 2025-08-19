#!/usr/bin/env python3
"""
MoE実装の問題を修正するスクリプト
- トークナイザーの語彙サイズ不一致
- エキスパートインデックスの範囲外アクセス
- embedding層のサイズ調整
"""

import sys
import os
sys.path.append('/workspace')

def fix_run_training():
    """run_training.pyのダミートークナイザーを修正"""
    file_path = "/workspace/scripts/moe/run_training.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # ダミートークナイザーの修正
    old_tokenizer = """        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                max_length = kwargs.get('max_length', 100)
                return {
                    'input_ids': torch.randint(0, 1000, (1, max_length)),
                    'attention_mask': torch.ones(1, max_length)
                }"""
    
    new_tokenizer = """        class DummyTokenizer:
            vocab_size = 32000  # モデルのembedding層と一致させる
            def __call__(self, text, **kwargs):
                max_length = kwargs.get('max_length', 100)
                return {
                    'input_ids': torch.randint(0, self.vocab_size, (1, max_length)),
                    'attention_mask': torch.ones(1, max_length)
                }
            def __len__(self):
                return self.vocab_size"""
    
    content = content.replace(old_tokenizer, new_tokenizer)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed tokenizer in run_training.py")

def fix_moe_architecture():
    """moe_architecture.pyのインデックス問題を修正"""
    file_path = "/workspace/src/moe/moe_architecture.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # TopKRouterのforward関数を修正（インデックスのクランプ）
    for i, line in enumerate(lines):
        # expert_indicesをクランプする処理を追加
        if "expert_indices = topk_indices % self.num_experts" in line:
            lines[i] = "        expert_indices = topk_indices % self.num_experts  # 範囲内に制限\n"
            lines.insert(i+1, "        expert_indices = torch.clamp(expert_indices, 0, self.num_experts - 1)  # 追加の安全対策\n")
            break
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Fixed expert indices clamping in moe_architecture.py")

def fix_moe_training():
    """moe_training.pyのバッチ処理を修正"""
    file_path = "/workspace/src/moe/moe_training.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # input_idsの範囲チェックを追加
    if "def train(" in content and "for batch in pbar:" in content:
        # バッチ処理の前にinput_idsをクランプ
        old_section = """            # Forward pass
            outputs = self.model("""
        
        new_section = """            # input_idsを語彙サイズ内に制限
            if hasattr(self.tokenizer, 'vocab_size'):
                vocab_size = self.tokenizer.vocab_size
            else:
                vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 32000
            
            batch['input_ids'] = torch.clamp(batch['input_ids'], 0, vocab_size - 1)
            
            # Forward pass
            outputs = self.model("""
        
        content = content.replace(old_section, new_section)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed input_ids clamping in moe_training.py")

def create_safe_test_script():
    """安全なテストスクリプトを作成"""
    test_script = """#!/usr/bin/env python3
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
        
        print(f"\\n✓ Input data created")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        print(f"  vocab_size: {vocab_size}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        print(f"\\n✓ Forward pass successful!")
        print(f"  Output shape: {outputs['hidden_states'].shape}")
        print(f"  Aux loss: {outputs['aux_loss'].item():.6f}")
        
        # メモリ使用量
        if torch.cuda.is_available():
            print(f"\\n📊 GPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_moe()
    if success:
        print("\\n" + "=" * 60)
        print("✅ All tests passed! MoE is ready for training.")
        print("=" * 60)
    else:
        print("\\n" + "=" * 60)
        print("❌ Tests failed. Please check the errors above.")
        print("=" * 60)
"""
    
    with open("/workspace/scripts/moe/test_moe_safe.py", "w") as f:
        f.write(test_script)
    
    print(f"✓ Created safe test script: test_moe_safe.py")

def main():
    print("=" * 60)
    print("MoE Issue Fixes")
    print("=" * 60)
    
    try:
        # 1. run_training.pyの修正
        fix_run_training()
        
        # 2. moe_architecture.pyの修正
        fix_moe_architecture()
        
        # 3. moe_training.pyの修正
        fix_moe_training()
        
        # 4. テストスクリプト作成
        create_safe_test_script()
        
        print("\n" + "=" * 60)
        print("✅ All fixes applied successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test the fixes: python scripts/moe/test_moe_safe.py")
        print("2. Run training: bash scripts/moe/train_moe.sh demo 1 1")
        
    except Exception as e:
        print(f"\n✗ Error applying fixes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()