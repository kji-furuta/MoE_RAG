"""
MoE Base Configuration Classes
MoEシステムの基本設定クラス
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from .constants import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_EXPERTS,
    DEFAULT_EXPERTS_PER_TOKEN,
    DEFAULT_AUX_LOSS_COEF,
    DEFAULT_ROUTER_JITTER_NOISE,
    DEFAULT_DROPOUT,
    DEFAULT_CAPACITY_FACTOR,
    DEFAULT_EXPERT_SPECIALIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_ROUTER_LR_MULTIPLIER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_PATH
)


@dataclass
class MoEConfig:
    """MoE設定クラス"""
    # Model Architecture
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_experts: int = DEFAULT_NUM_EXPERTS
    num_experts_per_tok: int = DEFAULT_EXPERTS_PER_TOKEN
    expert_capacity_factor: float = DEFAULT_CAPACITY_FACTOR
    
    # Loss and Regularization
    aux_loss_coef: float = DEFAULT_AUX_LOSS_COEF
    router_jitter_noise: float = DEFAULT_ROUTER_JITTER_NOISE
    dropout: float = DEFAULT_DROPOUT
    
    # Model Configuration
    use_bias: bool = False
    normalize_expert_weights: bool = True
    domain_specific_routing: bool = True
    
    # Expert Specialization
    expert_specialization: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_EXPERT_SPECIALIZATION.copy()
    )
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.expert_specialization is None:
            self.expert_specialization = DEFAULT_EXPERT_SPECIALIZATION.copy()
    
    def validate(self) -> None:
        """設定値の検証"""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_experts > 0, "num_experts must be positive"
        assert 0 < self.num_experts_per_tok <= self.num_experts, \
            "num_experts_per_tok must be between 1 and num_experts"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert self.expert_capacity_factor > 0, "expert_capacity_factor must be positive"


@dataclass
class MoETrainingConfig:
    """MoEトレーニング設定"""
    # Model Configuration
    base_model_name: str = "cyberagent/calm3-22b-chat"
    num_experts: int = DEFAULT_NUM_EXPERTS
    num_experts_per_tok: int = DEFAULT_EXPERTS_PER_TOKEN
    
    # Training Hyperparameters
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    
    # MoE Specific
    aux_loss_weight: float = DEFAULT_AUX_LOSS_COEF
    router_lr_multiplier: float = DEFAULT_ROUTER_LR_MULTIPLIER
    expert_dropout: float = DEFAULT_DROPOUT
    capacity_factor: float = DEFAULT_CAPACITY_FACTOR
    
    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    
    # Paths
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    dataset_path: str = DEFAULT_DATASET_PATH
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    use_wandb: bool = False
    project_name: str = "civil-engineering-moe"
    
    def validate(self) -> None:
        """設定値の検証"""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
        assert self.mixed_precision in ["fp16", "bf16", "fp32"], \
            f"Invalid mixed_precision: {self.mixed_precision}"