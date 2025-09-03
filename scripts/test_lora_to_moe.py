#!/usr/bin/env python3
"""
LoRA to MoE変換テストスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.moe.lora_to_moe_adapter import integrate_lora_to_moe, LoRAMoEConfig, LoRAToMoEAdapter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_lora_to_moe():
    """単一のLoRAアダプタをMoEに変換するテスト"""
    
    logger.info("=== 単一LoRA → MoE変換テスト ===")
    
    # テスト用の設定
    config = LoRAMoEConfig(
        lora_paths=["/workspace/outputs/lora_deepseek32b"],  # 既存のLoRAパス
        expert_names=["road_design_expert"],
        base_model_name="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        num_experts=1,
        num_experts_per_token=1,
        output_dir="/workspace/outputs/test_moe_single"
    )
    
    adapter = LoRAToMoEAdapter(config)
    result = adapter.convert_to_moe()
    
    if result["success"]:
        logger.info(f"✅ 変換成功: {result['message']}")
        logger.info(f"出力先: {result['output_dir']}")
        logger.info("次のステップ:")
        for step in result.get("next_steps", []):
            logger.info(f"  - {step}")
    else:
        logger.error(f"❌ 変換失敗: {result['error']}")
    
    return result


def test_multiple_lora_to_moe():
    """複数のLoRAアダプタをMoEに統合するテスト"""
    
    logger.info("=== 複数LoRA → MoE統合テスト ===")
    
    # 複数のLoRAパス（実際には同じLoRAを異なる専門家として使用）
    result = integrate_lora_to_moe(
        lora_paths=[
            "/workspace/outputs/lora_deepseek32b",  # 道路設計
            "/workspace/outputs/lora_deepseek32b",  # 橋梁（仮に同じ）
            "/workspace/outputs/lora_deepseek32b",  # トンネル（仮に同じ）
        ],
        expert_names=[
            "road_geometry_expert",
            "bridge_design_expert",
            "tunnel_design_expert"
        ],
        base_model="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        output_dir="/workspace/outputs/test_moe_multiple"
    )
    
    if result["success"]:
        logger.info(f"✅ 統合成功: {result['message']}")
        logger.info(f"エキスパート数: {result['num_experts']}")
        logger.info(f"エキスパート名: {', '.join(result['expert_names'])}")
    else:
        logger.error(f"❌ 統合失敗: {result['error']}")
    
    return result


def test_moe_inference():
    """変換されたMoEモデルの推論テスト"""
    
    logger.info("=== MoE推論テスト ===")
    
    import torch
    
    moe_path = Path("/workspace/outputs/test_moe_multiple/moe_model.pt")
    
    if not moe_path.exists():
        logger.warning("MoEモデルが見つかりません。先に変換を実行してください。")
        return
    
    # モデルをロード
    checkpoint = torch.load(moe_path, map_location="cpu")
    logger.info(f"モデルロード成功")
    logger.info(f"エキスパート: {checkpoint.get('expert_names', [])}")
    
    # テスト入力
    test_queries = [
        "設計速度100km/hの最小曲線半径は？",
        "橋梁の設計荷重について教えてください",
        "トンネルの換気方式の種類は？"
    ]
    
    logger.info("テストクエリ:")
    for query in test_queries:
        logger.info(f"  - {query}")
        # TODO: 実際の推論処理を実装
    
    logger.info("推論テスト完了（実際の推論は未実装）")


def main():
    """メインテスト実行"""
    
    logger.info("=" * 60)
    logger.info("LoRA to MoE 変換テスト開始")
    logger.info("=" * 60)
    
    # 1. 単一LoRA変換テスト
    single_result = test_single_lora_to_moe()
    
    # 2. 複数LoRA統合テスト
    if single_result and single_result["success"]:
        multiple_result = test_multiple_lora_to_moe()
        
        # 3. 推論テスト
        if multiple_result and multiple_result["success"]:
            test_moe_inference()
    
    logger.info("=" * 60)
    logger.info("テスト完了")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()