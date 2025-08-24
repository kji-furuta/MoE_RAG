"""
Optimized Mixture of Experts (MoE) Architecture
最適化されたMoEアーキテクチャ実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """MoE設定"""
    hidden_size: int = 4096
    intermediate_size: int = 11008  # FFN中間層サイズ
    num_experts: int = 8
    num_experts_per_tok: int = 2  # トークンあたりのアクティブExpert数
    expert_capacity_factor: float = 1.25  # Expert容量係数
    router_aux_loss_coef: float = 0.01  # ロードバランシング損失係数
    router_z_loss_coef: float = 0.001  # ルーターz損失係数
    dropout: float = 0.1
    use_bias: bool = False
    
    def validate(self):
        """設定の検証"""
        assert self.num_experts_per_tok <= self.num_experts
        assert self.num_experts_per_tok > 0
        assert self.hidden_size > 0
        assert self.intermediate_size > 0


class FeedForwardExpert(nn.Module):
    """単一のExpert（FFN）"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # SwiGLU activation: hidden_size -> 2 * intermediate_size -> hidden_size
        self.w_gate = nn.Linear(
            config.hidden_size, 
            config.intermediate_size, 
            bias=config.use_bias
        )
        self.w_up = nn.Linear(
            config.hidden_size, 
            config.intermediate_size, 
            bias=config.use_bias
        )
        self.w_down = nn.Linear(
            config.intermediate_size, 
            config.hidden_size, 
            bias=config.use_bias
        )
        
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Expert順伝播（SwiGLU）
        hidden_states: [tokens, hidden_size]
        """
        # SwiGLU: (silu(W_gate(x)) * W_up(x)) -> W_down
        gate_output = self.act_fn(self.w_gate(hidden_states))
        up_output = self.w_up(hidden_states)
        intermediate = gate_output * up_output
        intermediate = self.dropout(intermediate)
        output = self.w_down(intermediate)
        return output


class TopKGate(nn.Module):
    """Top-Kゲート（ルーター）"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # ゲート用の線形層
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        ルーティング計算
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            dispatch_mask: [batch_size * seq_len, num_experts, expert_capacity]
            combine_weights: [batch_size * seq_len, num_experts, expert_capacity]
            expert_indices: [batch_size * seq_len, top_k]
            aux_losses: 補助損失の辞書
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # ゲートロジットの計算
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        gate_logits = self.gate_proj(hidden_states_flat)  # [num_tokens, num_experts]
        
        # Top-K選択
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )  # [num_tokens, top_k]
        
        # Softmax正規化
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [num_tokens, top_k]
        
        # Expert容量の計算
        expert_capacity = int(
            (num_tokens * self.top_k / self.num_experts) * self.config.expert_capacity_factor
        )
        expert_capacity = max(expert_capacity, 1)
        
        # ディスパッチマスクと結合重みの作成
        dispatch_mask = torch.zeros(
            num_tokens, self.num_experts, expert_capacity,
            dtype=torch.bool, device=hidden_states.device
        )
        combine_weights = torch.zeros(
            num_tokens, self.num_experts, expert_capacity,
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # Expert割り当てカウンター
        expert_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )
        
        # 各トークンをExpertに割り当て
        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k].item()
                expert_count = expert_counts[expert_idx].item()
                
                if expert_count < expert_capacity:
                    dispatch_mask[token_idx, expert_idx, expert_count] = True
                    combine_weights[token_idx, expert_idx, expert_count] = top_k_weights[token_idx, k]
                    expert_counts[expert_idx] += 1
        
        # 補助損失の計算
        aux_losses = self._compute_aux_losses(gate_logits, dispatch_mask)
        
        return dispatch_mask, combine_weights, top_k_indices, aux_losses
    
    def _compute_aux_losses(
        self, 
        gate_logits: torch.Tensor,
        dispatch_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """補助損失の計算"""
        aux_losses = {}
        
        if self.training:
            # ロードバランシング損失
            # P(expert) * P(token|expert)のバランスを取る
            gate_probs = F.softmax(gate_logits, dim=-1)  # [num_tokens, num_experts]
            
            # 各Expertが選択される確率の平均
            mean_gate_prob = gate_probs.mean(dim=0)  # [num_experts]
            
            # 各Expertに割り当てられたトークン数
            tokens_per_expert = dispatch_mask.sum(dim=[0, 2]).float()  # [num_experts]
            tokens_per_expert = tokens_per_expert / tokens_per_expert.sum()
            
            # ロードバランシング損失
            aux_losses['load_balancing_loss'] = (
                self.config.router_aux_loss_coef * 
                self.num_experts * 
                torch.sum(mean_gate_prob * tokens_per_expert)
            )
            
            # ルーターz損失（ロジットの大きさを制限）
            router_z_loss = torch.logsumexp(gate_logits, dim=-1).mean()
            aux_losses['router_z_loss'] = self.config.router_z_loss_coef * router_z_loss
        
        return aux_losses


class MixtureOfExperts(nn.Module):
    """最適化されたMoEレイヤー"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # ゲート（ルーター）
        self.gate = TopKGate(config)
        
        # Expert群（FFN）
        self.experts = nn.ModuleList([
            FeedForwardExpert(config)
            for _ in range(config.num_experts)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MoE順伝播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_losses: 補助損失の辞書
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # ルーティング
        dispatch_mask, combine_weights, expert_indices, aux_losses = self.gate(hidden_states)
        
        # Expertの並列処理
        expert_outputs = []
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        for expert_idx, expert in enumerate(self.experts):
            # このExpertに割り当てられたトークンのマスク
            expert_mask = dispatch_mask[:, expert_idx, :]  # [num_tokens, expert_capacity]
            
            if expert_mask.any():
                # Expert入力の収集
                expert_tokens = hidden_states_flat[expert_mask.any(dim=-1)]
                
                if expert_tokens.numel() > 0:
                    # Expert処理
                    expert_output = expert(expert_tokens)
                    expert_outputs.append((expert_idx, expert_output, expert_mask))
        
        # 出力の結合
        final_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx, expert_output, expert_mask in expert_outputs:
            # 各Expertの出力を重み付けして結合
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) > 0:
                # 重みの適用
                weights = combine_weights[token_indices, expert_idx, 0]  # 簡略化
                weighted_output = expert_output * weights.unsqueeze(-1)
                
                # 最終出力への加算
                final_output[token_indices] += weighted_output
        
        # 形状を元に戻す
        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        
        return final_output, aux_losses


class TransformerLayerWithMoE(nn.Module):
    """MoEを組み込んだTransformerレイヤー"""
    
    def __init__(self, config: MoEConfig, attention_module: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        
        # Attention（既存のものを使用するか、ダミー）
        self.attention = attention_module if attention_module else nn.Identity()
        
        # MoE（FFNの代替）
        self.moe = MixtureOfExperts(config)
        
        # Layer Norm
        self.pre_moe_norm = nn.LayerNorm(config.hidden_size)
        self.post_moe_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transformerレイヤーの順伝播
        """
        # Attention（スキップ接続付き）
        residual = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MoE（FFNの代替、スキップ接続付き）
        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        hidden_states, aux_losses = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.post_moe_norm(hidden_states)
        
        return hidden_states, aux_losses


def create_moe_model(
    base_model: Optional[nn.Module] = None,
    config: Optional[MoEConfig] = None,
    replace_ffn_with_moe: bool = True
) -> nn.Module:
    """
    既存モデルのFFN層をMoEに置き換え
    
    Args:
        base_model: ベースとなるTransformerモデル
        config: MoE設定
        replace_ffn_with_moe: FFN層をMoEに置き換えるか
        
    Returns:
        MoE化されたモデル
    """
    if config is None:
        config = MoEConfig()
    
    if base_model is None:
        # テスト用のダミーモデル
        return TransformerLayerWithMoE(config)
    
    if replace_ffn_with_moe:
        # 既存モデルのFFN層をMoEに置き換え
        for name, module in base_model.named_modules():
            # FFN層を見つけてMoEに置き換え
            if 'mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower():
                # 親モジュールを取得して置き換え
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = base_model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    
                    # MoEに置き換え
                    setattr(parent_module, child_name, MixtureOfExperts(config))
                    logger.info(f"Replaced {name} with MoE")
    
    return base_model


# テスト用コード
if __name__ == "__main__":
    # 設定
    config = MoEConfig(
        hidden_size=768,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    # モデル作成
    model = TransformerLayerWithMoE(config)
    
    # ダミー入力
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 順伝播
    output, aux_losses = model(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary losses: {aux_losses}")
    
    # パラメータ数の比較
    total_params = sum(p.numel() for p in model.parameters())
    active_params = config.hidden_size * config.intermediate_size * 3 * config.num_experts_per_tok
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Active parameters per token: {active_params:,}")
    print(f"Efficiency ratio: {active_params / total_params:.2%}")