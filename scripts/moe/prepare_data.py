#!/usr/bin/env python3
"""
Prepare MoE Training Data
MoEトレーニングデータの準備
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

from src.moe.data_preparation import CivilEngineeringDataPreparator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """データ準備のメイン処理"""
    
    print("=" * 60)
    print("MoE土木・建設分野データ準備")
    print("=" * 60)
    
    # データ準備器の初期化
    preparator = CivilEngineeringDataPreparator(
        output_dir="./data/civil_engineering"
    )
    
    # トレーニングデータの生成
    print("\n1. トレーニングデータを生成中...")
    preparator.generate_training_data(num_samples_per_domain=100)
    
    # 検証データの作成
    print("\n2. 検証データを作成中...")
    preparator.create_validation_data(ratio=0.1)
    
    # テストシナリオの作成
    print("\n3. テストシナリオを作成中...")
    preparator.create_test_scenarios()
    
    # データ分布の分析
    print("\n4. データ分布を分析中...")
    stats = preparator.analyze_data_distribution()
    
    print("\n" + "=" * 60)
    print("✓ データ準備が完了しました！")
    print("=" * 60)
    
    print("\n生成されたファイル:")
    print("  - data/civil_engineering/train/*.jsonl (トレーニングデータ)")
    print("  - data/civil_engineering/val/*.jsonl (検証データ)")
    print("  - data/civil_engineering/test_scenarios.json (テストシナリオ)")
    
    return stats


if __name__ == "__main__":
    main()
