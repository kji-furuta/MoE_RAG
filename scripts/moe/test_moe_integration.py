#!/usr/bin/env python3
"""
MoE Integration Test
MoE実装の統合テスト
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moe_modules():
    """MoEモジュールのインポートテスト"""
    print("\n1. MoEモジュールインポートテスト")
    print("-" * 40)
    
    try:
        from src.moe import (
            MoEConfig,
            CivilEngineeringExpert,
            MoELayer,
            CivilEngineeringMoEModel,
            ExpertType
        )
        print("✓ moe_architecture モジュール")
        
        from src.moe import (
            MoETrainingConfig,
            CivilEngineeringDataset,
            MoETrainer
        )
        print("✓ moe_training モジュール")
        
        from src.moe import (
            DomainData,
            CivilEngineeringDataPreparator
        )
        print("✓ data_preparation モジュール")
        
        return True
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False


def test_model_creation():
    """モデル作成テスト"""
    print("\n2. MoEモデル作成テスト")
    print("-" * 40)
    
    try:
        from src.moe.moe_architecture import MoEConfig, CivilEngineeringMoEModel
        
        # 小規模テストモデルの作成
        config = MoEConfig(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2,
            dropout=0.1
        )
        
        model = CivilEngineeringMoEModel(config, base_model=None)
        
        print(f"✓ モデル作成成功")
        print(f"  エキスパート数: {config.num_experts}")
        print(f"  隠れ層サイズ: {config.hidden_size}")
        print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ モデル作成エラー: {e}")
        return False


def test_forward_pass():
    """順伝播テスト"""
    print("\n3. 順伝播テスト")
    print("-" * 40)
    
    try:
        from src.moe.moe_architecture import MoEConfig, MoELayer
        
        # テスト用の小規模設定
        config = MoEConfig(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2
        )
        
        # MoEレイヤーの作成
        moe_layer = MoELayer(config)
        moe_layer.eval()
        
        # ダミー入力
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # 順伝播
        with torch.no_grad():
            output, info = moe_layer(hidden_states)
        
        print(f"✓ 順伝播成功")
        print(f"  入力形状: {hidden_states.shape}")
        print(f"  出力形状: {output.shape}")
        print(f"  補助損失: {info['aux_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 順伝播エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preparation():
    """データ準備テスト"""
    print("\n4. データ準備テスト")
    print("-" * 40)
    
    try:
        from src.moe.data_preparation import CivilEngineeringDataPreparator
        
        # テスト用のデータ準備
        preparator = CivilEngineeringDataPreparator(
            output_dir="./data/civil_engineering_test"
        )
        
        # 少量のサンプルを生成
        preparator.generate_training_data(num_samples_per_domain=5)
        
        print(f"✓ データ生成成功")
        print(f"  出力ディレクトリ: ./data/civil_engineering_test")
        
        # 生成されたファイルの確認
        output_dir = Path("./data/civil_engineering_test/train")
        if output_dir.exists():
            files = list(output_dir.glob("*.jsonl"))
            print(f"  生成ファイル数: {len(files)}")
            for f in files[:3]:  # 最初の3つを表示
                print(f"    - {f.name}")
        
        return True
    except Exception as e:
        print(f"✗ データ準備エラー: {e}")
        return False


def test_expert_routing():
    """エキスパートルーティングテスト"""
    print("\n5. エキスパートルーティングテスト")
    print("-" * 40)
    
    try:
        from src.moe.moe_architecture import MoEConfig, TopKRouter, ExpertType
        
        config = MoEConfig(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2,
            domain_specific_routing=True
        )
        
        router = TopKRouter(config)
        router.eval()
        
        # テスト入力
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # ルーティング
        with torch.no_grad():
            router_logits, expert_indices, expert_weights = router(hidden_states)
        
        print(f"✓ ルーティング成功")
        print(f"  選択されたエキスパート形状: {expert_indices.shape}")
        print(f"  エキスパート重み形状: {expert_weights.shape}")
        
        # エキスパート分布の確認
        unique_experts = torch.unique(expert_indices)
        print(f"  使用エキスパート: {unique_experts.tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ ルーティングエラー: {e}")
        return False


def test_training_loop():
    """トレーニングループテスト（ミニバッチ）"""
    print("\n6. トレーニングループテスト")
    print("-" * 40)
    
    try:
        from src.moe.moe_architecture import MoEConfig, MoELayer
        import torch.nn as nn
        import torch.optim as optim
        
        # 小規模モデル
        config = MoEConfig(
            hidden_size=128,
            num_experts=4,
            num_experts_per_tok=2
        )
        
        moe_layer = MoELayer(config)
        optimizer = optim.Adam(moe_layer.parameters(), lr=1e-4)
        
        # ミニバッチでのトレーニング
        moe_layer.train()
        for step in range(3):
            # ダミーデータ
            hidden_states = torch.randn(2, 10, config.hidden_size)
            target = torch.randn(2, 10, config.hidden_size)
            
            # Forward
            output, info = moe_layer(hidden_states)
            
            # Loss
            mse_loss = nn.MSELoss()(output, target)
            total_loss = mse_loss + info['aux_loss']
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print(f"  Step {step+1}: Loss={total_loss.item():.4f}, "
                  f"MSE={mse_loss.item():.4f}, Aux={info['aux_loss'].item():.4f}")
        
        print(f"✓ トレーニングループ成功")
        return True
    except Exception as e:
        print(f"✗ トレーニングエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("=" * 60)
    print("MoE実装統合テスト - AI_FT_7プロジェクト")
    print("=" * 60)
    
    # 各テストの実行
    tests = [
        ("MoEモジュールインポート", test_moe_modules),
        ("モデル作成", test_model_creation),
        ("順伝播", test_forward_pass),
        ("データ準備", test_data_preparation),
        ("エキスパートルーティング", test_expert_routing),
        ("トレーニングループ", test_training_loop)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} で予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} : {test_name}")
    
    print("-" * 60)
    print(f"結果: {passed}/{total} テスト成功")
    
    if passed == total:
        print("\n🎉 すべてのテストが成功しました！")
        print("\n次のステップ:")
        print("1. データ準備: python scripts/moe/prepare_data.py")
        print("2. トレーニング: python scripts/moe/run_training.py --demo_mode")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
