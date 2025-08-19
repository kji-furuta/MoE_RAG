"""
Mixture of Experts (MoE) Architecture for Civil Engineering & Construction
土木・建設分野特化型MoEモデル実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import constants and configurations
from .constants import (
    ExpertType,
    DEFAULT_MLP_RATIO,
    DEFAULT_VOCAB_SIZE,
    SAFE_VOCAB_LIMIT,
    DOMAIN_KEYWORDS
)
from .base_config import MoEConfig

logger = logging.getLogger(__name__)


class CivilEngineeringExpert(nn.Module):
    """土木・建設分野の個別エキスパート"""
    
    def __init__(
        self,
        config: MoEConfig,
        expert_type: ExpertType,
        intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.expert_type = expert_type
        
        if intermediate_size is None:
            intermediate_size = config.hidden_size * DEFAULT_MLP_RATIO
        
        # FFNレイヤー
        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=config.use_bias)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=config.use_bias)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = nn.SiLU()
        
        # 専門知識埋め込み（ドメイン特化）
        self.domain_embedding = nn.Parameter(
            torch.randn(1, config.hidden_size) * 0.02
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """重み初期化"""
        for module in [self.w1, self.w2, self.w3]:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # ドメイン知識の注入
        hidden_states = hidden_states + self.domain_embedding
        
        # Swish-Gated Linear Unit (SwiGLU)
        x1 = self.act_fn(self.w1(hidden_states))
        x2 = self.w3(hidden_states)
        x = x1 * x2
        x = self.dropout(x)
        output = self.w2(x)
        
        return output


class TopKRouter(nn.Module):
    """Top-Kルーティング機構"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # ルーティング重み
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False
        )
        
        # ドメイン特化ルーティング用の追加レイヤー
        if config.domain_specific_routing:
            self.domain_gate = nn.Linear(
                config.hidden_size,
                config.num_experts,
                bias=True
            )
            
            # キーワード検出用の埋め込み
            self.keyword_embeddings = nn.ModuleDict({
                expert_type.value: nn.Embedding(1000, config.hidden_size)
                for expert_type in ExpertType
            })
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain_keywords: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ルーティング計算
        Returns:
            router_logits: ルーターのロジット
            expert_indices: 選択されたエキスパートのインデックス
            expert_weights: エキスパートの重み
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 基本ルーティングスコア
        router_logits = self.gate(hidden_states_flat)
        
        # ドメイン特化ルーティング
        if self.config.domain_specific_routing and domain_keywords is not None:
            domain_scores = self._compute_domain_scores(
                hidden_states_flat, domain_keywords
            )
            router_logits = router_logits + domain_scores
        
        # ジッターノイズ（訓練時のみ）
        if self.training and self.config.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.config.router_jitter_noise
            router_logits = router_logits + noise
        
        # Top-K選択
        topk_logits, topk_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        
        # Softmax正規化
        expert_weights = F.softmax(topk_logits, dim=-1)
        
        return router_logits, topk_indices, expert_weights
    
    def _compute_domain_scores(
        self,
        hidden_states: torch.Tensor,
        domain_keywords: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """ドメイン特化スコアの計算"""
        domain_scores = torch.zeros(
            hidden_states.size(0),
            self.num_experts,
            device=hidden_states.device
        )
        
        for i, expert_type in enumerate(ExpertType):
            if expert_type.value in domain_keywords:
                keyword_ids = domain_keywords[expert_type.value]
                keyword_emb = self.keyword_embeddings[expert_type.value](keyword_ids)
                
                # コサイン類似度でスコア計算
                similarity = F.cosine_similarity(
                    hidden_states.unsqueeze(1),
                    keyword_emb.unsqueeze(0),
                    dim=-1
                )
                domain_scores[:, i] = similarity.mean(dim=-1)
        
        return domain_scores * self.config.expert_specialization.get(expert_type.value, 1.0)


class MoELayer(nn.Module):
    """MoEレイヤー"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # ルーター
        self.router = TopKRouter(config)
        
        # エキスパート群
        self.experts = nn.ModuleList([
            CivilEngineeringExpert(config, expert_type)
            for expert_type in ExpertType
        ])
        
        assert len(self.experts) == config.num_experts
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain_keywords: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MoE順伝播
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # ルーティング
        router_logits, expert_indices, expert_weights = self.router(
            hidden_states, attention_mask, domain_keywords
        )
        
        # エキスパート処理の準備
        final_output = torch.zeros_like(hidden_states)
        
        # 各エキスパートの処理
        for expert_idx in range(self.config.num_experts):
            # このエキスパートが選択されたトークンのマスク
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # エキスパート入力の取得
                expert_input = hidden_states.view(-1, hidden_dim)[expert_mask]
                
                # エキスパート処理
                expert_output = self.experts[expert_idx](
                    expert_input.unsqueeze(0)
                ).squeeze(0)
                
                # 重み付き出力の集約
                expert_weight = expert_weights.view(-1, self.config.num_experts_per_tok)
                expert_weight_for_this = expert_weight[expert_mask]
                weight_for_this_expert = expert_weight_for_this[
                    expert_indices[expert_mask] == expert_idx
                ].unsqueeze(-1)
                
                # 最終出力への追加
                final_output.view(-1, hidden_dim)[expert_mask] += (
                    expert_output * weight_for_this_expert
                )
        
        # 補助損失の計算（ロードバランシング）
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)
        
        return final_output, {
            "aux_loss": aux_loss,
            "router_logits": router_logits,
            "expert_indices": expert_indices,
            "expert_weights": expert_weights
        }
    
    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """補助損失（ロードバランシング）の計算"""
        if not self.training:
            return torch.tensor(0.0, device=router_logits.device)
        
        # エキスパートごとの選択頻度
        expert_counts = torch.bincount(
            expert_indices.flatten(),
            minlength=self.config.num_experts
        ).float()
        
        # 理想的な均等分布
        ideal_count = expert_indices.numel() / self.config.num_experts
        
        # L2損失
        aux_loss = torch.sum((expert_counts - ideal_count) ** 2) / ideal_count
        
        return aux_loss * self.config.aux_loss_coef


class CivilEngineeringMoEModel(nn.Module):
    """土木・建設分野特化MoEモデル"""
    
    def __init__(self, config: MoEConfig, base_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        
        # ベースモデルのトランスフォーマーレイヤー
        if base_model is not None:
            self.base_layers = base_model
        else:
            # ベースモデルがない場合は埋め込み層とダミー変換
            self.embedding = nn.Embedding(32000, config.hidden_size)
            self.base_layers = None  # 後でforward内で処理
        
        # MoEレイヤー（FFNの置き換え）
        self.moe_layers = nn.ModuleList([
            MoELayer(config)
            for _ in range(12)  # 12層のMoE
        ])
        
        # ドメインキーワード検出器
        self.keyword_detector = DomainKeywordDetector(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """順伝播"""
        # 入力検証とサニタイズ
        input_ids = self._sanitize_input_ids(input_ids)
        
        # ドメインキーワードの検出
        domain_keywords = self.keyword_detector(input_ids)
        
        # ベースモデルの処理
        if self.base_layers is not None:
            hidden_states = self.base_layers(input_ids, attention_mask, **kwargs)
        else:
            # embedding層を使用（勾配が正しく伝播する）
            hidden_states = self.embedding(input_ids)
        
        total_aux_loss = 0.0
        all_router_info = []
        
        # MoEレイヤーの処理
        for moe_layer in self.moe_layers:
            hidden_states, moe_info = moe_layer(
                hidden_states,
                attention_mask,
                domain_keywords
            )
            total_aux_loss += moe_info["aux_loss"]
            all_router_info.append(moe_info)
        
        return {
            "hidden_states": hidden_states,
            "aux_loss": total_aux_loss,
            "router_info": all_router_info
        }
    
    def _sanitize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """入力IDのサニタイズ処理"""
        vocab_size = getattr(self.embedding, 'num_embeddings', DEFAULT_VOCAB_SIZE)
        safe_limit = min(vocab_size - 1, SAFE_VOCAB_LIMIT)
        
        if input_ids.max() >= vocab_size:
            logger.warning(
                f"Input IDs exceed vocab size: max={input_ids.max().item()}, "
                f"vocab_size={vocab_size}. Clamping to safe range."
            )
        
        return torch.clamp(input_ids, min=0, max=safe_limit)


class DomainKeywordDetector(nn.Module):
    """土木・建設ドメインキーワード検出器"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # ドメイン辞書を定数から取得
        self.domain_keywords = DOMAIN_KEYWORDS
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """キーワード検出（簡易実装）"""
        # 実際の実装では、トークナイザーと連携して
        # 入力テキストからドメインキーワードを検出
        detected_keywords = {}
        
        # ダミー実装（実際には入力トークンを解析）
        for expert_type in ExpertType:
            # ランダムにキーワードを検出したと仮定
            if torch.rand(1).item() > 0.5:
                detected_keywords[expert_type.value] = torch.tensor(
                    [0],  # キーワードID
                    device=input_ids.device
                )
        
        return detected_keywords


# ユーティリティ関数
def create_civil_engineering_moe(
    base_model_name: str = "cyberagent/calm3-22b-chat",
    num_experts: int = 8,
    device: Optional[str] = None,
    use_quantization: bool = False,
    use_lora: bool = False,
    model: Optional[nn.Module] = None
) -> nn.Module:
    """
    土木・建設分野MoEモデルの作成
    
    Args:
        base_model_name: ベースモデル名
        num_experts: エキスパート数
        device: 実行デバイス（None の場合は自動選択）
        use_quantization: 量子化を使用するか
        use_lora: LoRAを使用するか
        model: 事前に準備されたモデル（量子化・LoRA済み）
    
    Returns:
        nn.Module: 初期化済みモデル（MoEまたは量子化モデル）
    """
    # 量子化モデルの場合は直接返す（MoEラッパーなし）
    if use_quantization and use_lora and model is not None:
        logger.info("Using quantized model with LoRA directly (no MoE wrapper)")
        return model
    
    config = MoEConfig(
        num_experts=num_experts,
        hidden_size=4096,  # CALM3-22Bの隠れ層サイズ
        num_experts_per_tok=2,
        domain_specific_routing=True
    )
    
    # 設定の検証
    config.validate()
    
    # ベースモデルのロード
    if model is not None:
        # 既に準備されたモデルを使用
        base_model = model
        logger.info("Using pre-configured model")
    else:
        # 通常のモデルロード
        base_model = None  # ここでbase_modelをロード
    
    model = CivilEngineeringMoEModel(config, base_model)
    
    # デバイスの選択（量子化モデルでない場合のみ）
    if not (use_quantization and use_lora):
        if device is None:
            from .utils import get_device
            device = get_device()
        else:
            device = torch.device(device)
        
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    
    logger.info(f"Created Civil Engineering MoE model with {num_experts} experts")
    
    return model


if __name__ == "__main__":
    # テスト実行
    model = create_civil_engineering_moe()
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
