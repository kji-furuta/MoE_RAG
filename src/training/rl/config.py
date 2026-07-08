"""
RLAnything Configuration

RLAnythingフレームワークの全コンポーネント設定を管理する。
既存のDPOTrainingConfig/TrainingConfigパターンに準拠。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RLAnythingConfig:
    """RLAnything閉ループ強化学習設定

    3つのコンポーネント（Policy, Reward, Environment）の設定と、
    閉ループオーケストレーションの制御パラメータを統合管理する。
    """

    # ── モデル基本設定 ──────────────────────────────────
    model_name: str = "cyberagent/calm3-22b-chat"
    output_dir: str = "./outputs/rlanything"

    # ── LoRA / QLoRA 設定 ──────────────────────────────
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # ── 量子化設定 ─────────────────────────────────────
    use_quantization: bool = True
    quantization_bits: int = 4  # 4-bit or 8-bit
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ── Policy (GRPO) 設定 ─────────────────────────────
    grpo_num_generations: int = 8  # グループサンプリング数 G
    grpo_temperature: float = 1.0  # サンプリング温度
    grpo_max_new_tokens: int = 512
    grpo_beta: float = 0.04  # KLペナルティ係数
    grpo_epsilon_low: float = 0.2  # クリッピング下限
    grpo_epsilon_high: float = 0.2  # クリッピング上限
    policy_learning_rate: float = 1e-6
    policy_per_device_batch_size: int = 1
    policy_gradient_accumulation_steps: int = 16
    policy_max_steps: int = 500

    # ── Reward Model 設定 ──────────────────────────────
    reward_model_name: Optional[str] = None  # Noneの場合、policyモデルを流用
    reward_learning_rate: float = 5e-6
    reward_per_device_batch_size: int = 2
    reward_gradient_accumulation_steps: int = 8
    reward_max_steps: int = 200
    reward_self_consistency_k: int = 5  # 自己一貫性チェック回数
    reward_outcome_weight: float = 0.6  # λ: outcome報酬の重み
    reward_process_weight: float = 0.4  # (1-λ): process報酬の重み

    # ── Environment Adapter 設定 ───────────────────────
    env_success_rate_low: float = 0.2  # α_low: 難易度下げ閾値
    env_success_rate_high: float = 0.8  # α_high: 難易度上げ閾値
    env_adaptation_window: int = 50  # 成功率計算ウィンドウ
    env_max_difficulty: int = 5  # 最大難易度レベル
    env_min_difficulty: int = 1  # 最小難易度レベル
    env_enable_critical_feedback: bool = True  # LLMフィードバック有効化

    # ── Closed-Loop Orchestration 設定 ─────────────────
    num_iterations: int = 10  # 閉ループ反復回数
    trajectories_per_iteration: int = 64  # イテレーション毎の軌跡数
    reward_update_interval: int = 2  # 報酬モデル更新間隔
    environment_adapt_interval: int = 3  # 環境適応間隔
    convergence_threshold: float = 0.01  # 収束判定閾値
    early_stopping_patience: int = 3  # 早期停止の忍耐度

    # ── メモリ最適化 ───────────────────────────────────
    gradient_checkpointing: bool = True
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    max_memory_per_gpu: str = "23GiB"
    max_cpu_memory: str = "40GiB"

    # ── ログ・保存 ────────────────────────────────────
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "none"

    # ── データ設定 ────────────────────────────────────
    dataset_path: Optional[str] = None
    max_prompt_length: int = 512
    max_completion_length: int = 512

    def __post_init__(self):
        """デフォルトのtarget_modulesを設定"""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        # output_dirをPathに正規化
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RLAnythingConfig":
        """YAMLファイルから設定をロード"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # ネストされたセクションをフラットにマージ
        flat: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        # dataclass fieldに存在するものだけフィルタ
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式でエクスポート"""
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> List[str]:
        """設定値のバリデーション。警告メッセージのリストを返す。"""
        warnings: List[str] = []

        if self.grpo_num_generations < 2:
            warnings.append(
                f"grpo_num_generations={self.grpo_num_generations} は最低2以上推奨"
            )
        if not (0.0 < self.reward_outcome_weight < 1.0):
            warnings.append(
                f"reward_outcome_weight={self.reward_outcome_weight} は0-1の範囲を推奨"
            )
        if self.env_success_rate_low >= self.env_success_rate_high:
            warnings.append(
                f"env_success_rate_low({self.env_success_rate_low}) >= "
                f"env_success_rate_high({self.env_success_rate_high})"
            )
        if self.num_iterations < 1:
            warnings.append(f"num_iterations={self.num_iterations} は1以上必要")

        return warnings
