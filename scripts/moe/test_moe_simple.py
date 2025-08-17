#!/usr/bin/env python3
"""
Simple MoE Implementation Test
MoE実装の簡易テスト
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moe_basic():
    """基本的なMoE動作テスト"""
    
    class SimpleMoELayer(nn.Module):
        """簡易版MoEレイヤー"""
        
        def __init__(self, hidden_size: int = 512, num_experts: int = 8):
            super().__init__()
            self.num_experts = num_experts
            self.hidden_size = hidden_size
            
            # エキスパート（簡易FFN）
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                for _ in range(num_experts)
            ])
            
            # ルーター
            self.router = nn.Linear(hidden_size, num_experts)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, hidden_size = x.shape
            
            # ルーティング
            router_logits = self.router(x.view(-1, hidden_size))
            router_probs = torch.softmax(router_logits, dim=-1)
            
            # Top-2エキスパート選択
            top_k = 2
            top_probs, top_indices = torch.topk(router_probs, top_k, dim=-1)
            
            # エキスパート処理（簡易版）
            output = torch.zeros_like(x.view(-1, hidden_size))
            
            for expert_idx in range(self.num_experts):
                # このエキスパートが選択されたトークン
                mask = (top_indices == expert_idx).any(dim=-1)
                if mask.any():
                    expert_input = x.view(-1, hidden_size)[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # 重み付き出力
                    weights = top_probs[mask][:, (top_indices[mask] == expert_idx).nonzero(as_tuple=True)[1]]
                    output[mask] += expert_output * weights.unsqueeze(-1)
            
            return output.view(batch_size, seq_len, hidden_size)
    
    # テスト実行
    print("簡易MoEレイヤーのテスト...")
    
    # モデル作成
    model = SimpleMoELayer(hidden_size=512, num_experts=8)
    model.eval()
    
    # ダミー入力
    batch_size, seq_len, hidden_size = 2, 10, 512
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    
    # 推論
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ 入力形状: {dummy_input.shape}")
    print(f"✓ 出力形状: {output.shape}")
    print(f"✓ エキスパート数: {model.num_experts}")
    print(f"✓ パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    return True


def test_domain_routing():
    """ドメイン特化ルーティングのテスト"""
    
    # 土木・建設ドメインキーワード
    domains = {
        "structural_design": ["構造", "梁", "柱", "基礎", "耐震"],
        "road_design": ["道路", "舗装", "線形", "勾配", "交差点"],
        "geotechnical": ["地盤", "土質", "支持力", "沈下", "液状化"],
        "hydraulics": ["排水", "流量", "管渠", "ポンプ", "洪水"],
        "materials": ["コンクリート", "鋼材", "アスファルト", "強度"],
        "construction_management": ["工程", "安全", "品質", "施工", "管理"],
        "regulations": ["基準", "法規", "JIS", "道路構造令", "建築基準法"],
        "environmental": ["環境", "騒音", "振動", "廃棄物", "維持"]
    }
    
    # テストクエリ
    test_queries = [
        "道路の設計速度と曲線半径の関係について教えてください",
        "RC造建物の耐震設計における保有水平耐力とは？",
        "N値15の地盤における直接基礎の支持力計算方法",
        "コンクリートの配合設計で水セメント比の決定方法"
    ]
    
    print("\nドメインルーティングテスト...")
    
    for query in test_queries:
        # 簡易的なキーワードマッチング
        matched_domains = []
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in query:
                    matched_domains.append(domain)
                    break
        
        print(f"\nクエリ: {query[:30]}...")
        print(f"マッチしたドメイン: {matched_domains if matched_domains else ['general']}")
    
    return True


def test_memory_usage():
    """メモリ使用量の推定"""
    
    print("\nメモリ使用量推定...")
    
    # パラメータ数からメモリ使用量を推定
    hidden_size = 4096  # CALM3-22Bの想定サイズ
    num_experts = 8
    num_layers = 40  # 仮定
    
    # エキスパートのパラメータ数（FFN: hidden -> 4*hidden -> hidden）
    expert_params = 2 * hidden_size * (4 * hidden_size)
    total_expert_params = expert_params * num_experts * num_layers
    
    # ルーターのパラメータ数
    router_params = hidden_size * num_experts * num_layers
    
    total_params = total_expert_params + router_params
    
    # メモリ使用量（BF16の場合）
    memory_gb = (total_params * 2) / (1024**3)  # 2 bytes per param in BF16
    
    print(f"推定パラメータ数: {total_params:,}")
    print(f"推定メモリ使用量（BF16）: {memory_gb:.1f} GB")
    
    # スパース活性化時のメモリ
    active_experts = 2
    sparse_memory_gb = memory_gb * (active_experts / num_experts)
    print(f"スパース活性化時（2/8エキスパート）: {sparse_memory_gb:.1f} GB")
    
    return True


def main():
    print("=" * 60)
    print("MoE実装テスト - AI_FT_7プロジェクト")
    print("=" * 60)
    
    # 各テストの実行
    tests = [
        ("基本MoE動作テスト", test_moe_basic),
        ("ドメインルーティングテスト", test_domain_routing),
        ("メモリ使用量推定", test_memory_usage)
    ]
    
    for test_name, test_func in tests:
        print(f"\n### {test_name} ###")
        try:
            success = test_func()
            if success:
                print(f"✓ {test_name} 成功")
        except Exception as e:
            print(f"✗ {test_name} 失敗: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    
    print("\n次のステップ：")
    print("1. 完全なMoEモジュールファイルをsrc/moe/に配置")
    print("2. データ準備: python scripts/moe/prepare_data.py")
    print("3. トレーニング開始: bash scripts/moe/train_moe.sh")


if __name__ == "__main__":
    main()
