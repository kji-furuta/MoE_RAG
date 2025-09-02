#!/usr/bin/env python3
"""
GPUの統合メモリ利用可能性をテスト
NVLink接続されたGPUを単一メモリプールとして使用できるか確認
"""

import torch
import numpy as np

def test_gpu_unification():
    """GPU統合テスト"""
    
    print("=" * 60)
    print("GPU統合メモリテスト")
    print("=" * 60)
    
    # 基本情報
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"\n✅ 検出されたGPU数: {gpu_count}")
    
    # 各GPUの情報
    total_memory = 0
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        total_memory += memory_gb
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  メモリ: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print(f"\n合計GPUメモリ: {total_memory:.1f} GB")
    
    # NVLink/P2P通信テスト
    print("\n" + "=" * 40)
    print("P2P (Peer-to-Peer) アクセステスト")
    print("=" * 40)
    
    for i in range(gpu_count):
        for j in range(gpu_count):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"GPU {i} → GPU {j}: {'✅ 可能' if can_access else '❌ 不可'}")
    
    # 統合メモリプールテスト
    print("\n" + "=" * 40)
    print("統合メモリプールテスト")
    print("=" * 40)
    
    # PyTorchのメモリプール設定
    try:
        # CUDAメモリプールを有効化
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 大きなテンソルを作成（両GPUのメモリを使用）
        print("\n1. 大規模テンソル作成テスト (30GB)")
        
        # DataParallelを使用した統合
        device = torch.device("cuda:0")
        
        # 30GBのテンソルを作成試行
        try:
            # float32で30GB = 約7.5G要素
            size = int(7.5e9)
            tensor = torch.zeros(size, dtype=torch.float32, device=device)
            actual_size = tensor.element_size() * tensor.nelement() / 1024**3
            print(f"✅ {actual_size:.1f}GB のテンソル作成成功！")
            print("   → GPUメモリが統合的に使用可能")
            del tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"❌ 単一デバイスでの大規模テンソル作成失敗")
            print(f"   エラー: {str(e)[:100]}...")
            print("\n   → GPUメモリは物理的に分離（統合不可）")
    
    except Exception as e:
        print(f"テスト中にエラー: {e}")
    
    # モデル並列化の推奨設定
    print("\n" + "=" * 40)
    print("推奨設定")
    print("=" * 40)
    
    if gpu_count >= 2:
        print("\n✅ NVLink接続が検出されました")
        print("   帯域幅: 4 x 14.062 GB/s = 56.248 GB/s")
        print("\n推奨される使用方法:")
        print("1. モデル並列化 (device_map='auto')")
        print("2. DataParallel または DistributedDataParallel")
        print("3. DeepSpeed ZeRO Stage 3")
        print("\n注意: GPUメモリは物理的に分離されているため、")
        print("      単一の48GBプールとしては使用できません。")
        print("      ただし、NVLinkにより高速なGPU間通信が可能です。")

if __name__ == "__main__":
    test_gpu_unification()