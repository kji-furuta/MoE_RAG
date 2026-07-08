"""
Reward Integrator - Outcome + Process 統合報酬

RLAnythingの統合報酬計算コンポーネント。
Outcome報酬とProcess報酬を重み付けで統合する。

統合報酬式:
    R_i = λ * R_outcome_i + (1 - λ) * mean(R_process_i_steps)

ここで:
    λ = reward_outcome_weight (デフォルト 0.6)
    R_outcome_i = i番目の応答のOutcome報酬
    R_process_i_steps = i番目の応答の各ステップのProcess報酬
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import RLAnythingConfig
from .trajectory_buffer import Trajectory

logger = logging.getLogger(__name__)


class RewardIntegrator:
    """Outcome + Process 報酬統合器"""

    def __init__(self, config: RLAnythingConfig):
        self.outcome_weight = config.reward_outcome_weight
        self.process_weight = config.reward_process_weight
        logger.info(
            f"RewardIntegrator初期化: outcome_weight={self.outcome_weight}, "
            f"process_weight={self.process_weight}"
        )

    def integrate(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Outcome報酬とProcess報酬を統合して最終報酬を計算

        Args:
            trajectories: outcome_rewardsとprocess_rewardsが設定済みの軌跡リスト

        Returns:
            rewardsが統合された軌跡リスト
        """
        for trajectory in trajectories:
            integrated_rewards: List[float] = []

            for i in range(len(trajectory.completions)):
                # Outcome報酬
                outcome = 0.0
                if i < len(trajectory.outcome_rewards):
                    outcome = trajectory.outcome_rewards[i]

                # Process報酬（ステップ平均）
                process = 0.0
                if i < len(trajectory.process_rewards) and trajectory.process_rewards[i]:
                    step_scores = trajectory.process_rewards[i]
                    process = sum(step_scores) / len(step_scores)

                # 統合: R = λ * outcome + (1-λ) * process
                integrated = (
                    self.outcome_weight * outcome
                    + self.process_weight * process
                )
                integrated_rewards.append(integrated)

            trajectory.rewards = integrated_rewards

            if integrated_rewards:
                logger.debug(
                    f"統合報酬 - prompt[:30]='{trajectory.prompt[:30]}...': "
                    f"mean={sum(integrated_rewards)/len(integrated_rewards):.4f}, "
                    f"max={max(integrated_rewards):.4f}, "
                    f"min={min(integrated_rewards):.4f}"
                )

        return trajectories

    def normalize_rewards(
        self, trajectories: List[Trajectory], group_normalize: bool = True
    ) -> List[Trajectory]:
        """報酬を正規化

        GRPO Advantage計算のために、グループ内で報酬を正規化:
            A_i = (r_i - mean(r)) / (std(r) + eps)

        Args:
            trajectories: 報酬が計算済みの軌跡リスト
            group_normalize: True=軌跡内で正規化, False=全体で正規化
        """
        if group_normalize:
            # 各軌跡（グループ）内で正規化
            for trajectory in trajectories:
                if len(trajectory.rewards) < 2:
                    continue
                rewards = trajectory.rewards
                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                std_r = var_r ** 0.5
                eps = 1e-8
                trajectory.rewards = [
                    (r - mean_r) / (std_r + eps) for r in rewards
                ]
        else:
            # 全軌跡の報酬を集めて一括正規化
            all_rewards = [r for t in trajectories for r in t.rewards]
            if len(all_rewards) < 2:
                return trajectories
            mean_r = sum(all_rewards) / len(all_rewards)
            var_r = sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)
            std_r = var_r ** 0.5
            eps = 1e-8
            for trajectory in trajectories:
                trajectory.rewards = [
                    (r - mean_r) / (std_r + eps) for r in trajectory.rewards
                ]

        return trajectories
