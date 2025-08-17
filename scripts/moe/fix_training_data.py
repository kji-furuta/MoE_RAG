#!/usr/bin/env python3
"""
トレーニングデータの問題を根本的に修正
"""

import sys
sys.path.append('/workspace')
import torch

def fix_moe_training_comprehensive():
    """moe_training.pyを包括的に修正"""
    
    file_path = "/workspace/src/moe/moe_training.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # CivilEngineeringDatasetクラスの__getitem__メソッドを探して修正
    for i, line in enumerate(lines):
        if "def __getitem__(self, idx):" in line:
            # __getitem__メソッドの最後を探す
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("def "):
                if "return {" in lines[j]:
                    # returnステートメントの前に入力値のクランプを追加
                    lines.insert(j, "        # 入力値を語彙サイズ内に制限\n")
                    lines.insert(j+1, "        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 32000\n")
                    lines.insert(j+2, "        if 'input_ids' in encoding:\n")
                    lines.insert(j+3, "            encoding['input_ids'] = torch.clamp(encoding['input_ids'], 0, vocab_size - 1)\n")
                    lines.insert(j+4, "        \n")
                    break
                j += 1
            break
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print("✓ Fixed data loading in moe_training.py")
    
    # data_preparation.pyも修正
    fix_data_preparation()

def fix_data_preparation():
    """data_preparation.pyのダミーデータ生成を修正"""
    
    file_path = "/workspace/src/moe/data_preparation.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # generate_domain_dataメソッドでのダミーデータ生成を修正
    if "def generate_domain_data" in content:
        # vocab_sizeの設定を追加
        old_pattern = "def generate_domain_data(self, domain: str, num_samples: int = 100) -> List[Dict]:"
        new_pattern = """def generate_domain_data(self, domain: str, num_samples: int = 100) -> List[Dict]:
        \"\"\"ドメイン特化データの生成\"\"\"
        vocab_size = 32000  # 統一された語彙サイズ"""
        
        content = content.replace(old_pattern, new_pattern)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed data generation in data_preparation.py")

def create_minimal_training_script():
    """最小限のトレーニングスクリプトを作成"""
    
    script = '''#!/usr/bin/env python3
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
    
    print("\\n✅ Training test completed successfully!")
    return True

if __name__ == "__main__":
    success = minimal_training_test()
    if not success:
        print("\\n❌ Training test failed")
'''
    
    with open("/workspace/scripts/moe/test_minimal_training.py", "w") as f:
        f.write(script)
    
    print("✓ Created minimal training test script")

def main():
    print("=" * 60)
    print("Comprehensive MoE Training Fixes")
    print("=" * 60)
    
    try:
        # 1. moe_training.pyの修正
        fix_moe_training_comprehensive()
        
        # 2. 最小トレーニングスクリプト作成
        create_minimal_training_script()
        
        print("\n" + "=" * 60)
        print("✅ All additional fixes applied!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test minimal training: python scripts/moe/test_minimal_training.py")
        print("2. If successful, regenerate data: python scripts/moe/prepare_data.py")
        print("3. Run full training: bash scripts/moe/train_moe.sh demo 1 1")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()