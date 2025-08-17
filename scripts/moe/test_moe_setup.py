#!/usr/bin/env python3
"""
MoE Setup Test Script
セットアップの確認用スクリプト
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

def test_imports():
    """必要なモジュールのインポートテスト"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import einops
        print("✓ einops")
        
        import scipy
        print("✓ scipy")
        
        import tensorboardX
        print("✓ tensorboardX")
        
        # MoEモジュールのテスト（存在する場合）
        try:
            from src.moe import MoEConfig
            print("✓ MoEモジュール")
        except ImportError:
            print("! MoEモジュールは未配置です")
        
        return True
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False

def test_gpu():
    """GPU利用可能性のテスト"""
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU利用可能: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("✗ GPUが利用できません")
        return False

def test_memory():
    """メモリ容量のテスト"""
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"✓ GPU {i} メモリ: {mem_total:.1f}GB (使用中: {mem_allocated:.1f}GB)")
        return True
    return False

def main():
    print("=" * 50)
    print("MoEセットアップテスト")
    print("=" * 50)
    
    print("\n1. モジュールインポートテスト")
    test_imports()
    
    print("\n2. GPU利用可能性テスト")
    test_gpu()
    
    print("\n3. メモリ容量テスト")
    test_memory()
    
    print("\n" + "=" * 50)
    print("テスト完了")

if __name__ == "__main__":
    main()
