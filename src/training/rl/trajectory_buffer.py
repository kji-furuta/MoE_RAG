"""
Trajectory Buffer

GRPO用の軌跡（trajectory）データを管理するバッファ。
各軌跡はプロンプト、生成された応答群、報酬スコアを含む。
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """単一の軌跡データ

    Attributes:
        prompt: 入力プロンプト
        completions: G個の生成応答リスト
        rewards: 各応答に対する統合報酬スコア
        outcome_rewards: Outcome報酬（正解判定）
        process_rewards: Process報酬（ステップごとの評価）
        difficulty_level: 生成時のタスク難易度
        metadata: 追加メタデータ
    """
    prompt: str
    completions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    outcome_rewards: List[float] = field(default_factory=list)
    process_rewards: List[List[float]] = field(default_factory=list)
    difficulty_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_completions(self) -> int:
        return len(self.completions)

    @property
    def best_completion_idx(self) -> int:
        """最高報酬の応答インデックスを返す"""
        if not self.rewards:
            return 0
        return max(range(len(self.rewards)), key=lambda i: self.rewards[i])

    @property
    def best_completion(self) -> str:
        """最高報酬の応答を返す"""
        if not self.completions:
            return ""
        return self.completions[self.best_completion_idx]

    @property
    def mean_reward(self) -> float:
        """平均報酬を返す"""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)

    @property
    def is_successful(self) -> bool:
        """最高報酬が正の場合、成功と判定"""
        if not self.rewards:
            return False
        return max(self.rewards) > 0.0


class TrajectoryBuffer:
    """軌跡バッファ

    GRPOトレーニング用に軌跡データを蓄積・サンプリングする。
    固定容量のリングバッファとして動作する。
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: deque[Trajectory] = deque(maxlen=capacity)
        self._iteration_buffers: Dict[int, List[Trajectory]] = {}
        logger.info(f"TrajectoryBuffer初期化: capacity={capacity}")

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, trajectory: Trajectory, iteration: Optional[int] = None) -> None:
        """軌跡を追加"""
        self._buffer.append(trajectory)
        if iteration is not None:
            self._iteration_buffers.setdefault(iteration, []).append(trajectory)

    def add_batch(
        self, trajectories: List[Trajectory], iteration: Optional[int] = None
    ) -> None:
        """軌跡をバッチ追加"""
        for t in trajectories:
            self.add(t, iteration=iteration)

    def sample(self, n: int) -> List[Trajectory]:
        """ランダムにn個の軌跡をサンプリング"""
        n = min(n, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def get_iteration_trajectories(self, iteration: int) -> List[Trajectory]:
        """特定イテレーションの軌跡を取得"""
        return self._iteration_buffers.get(iteration, [])

    def compute_success_rate(self, window: Optional[int] = None) -> float:
        """直近の成功率を計算

        Args:
            window: 計算に使用する直近の軌跡数。Noneの場合は全件。
        """
        if not self._buffer:
            return 0.0

        if window is not None:
            trajectories = list(self._buffer)[-window:]
        else:
            trajectories = list(self._buffer)

        successes = sum(1 for t in trajectories if t.is_successful)
        return successes / len(trajectories)

    def compute_mean_reward(self, window: Optional[int] = None) -> float:
        """直近の平均報酬を計算"""
        if not self._buffer:
            return 0.0

        if window is not None:
            trajectories = list(self._buffer)[-window:]
        else:
            trajectories = list(self._buffer)

        total_reward = sum(t.mean_reward for t in trajectories)
        return total_reward / len(trajectories)

    def get_training_pairs(self) -> List[Dict[str, Any]]:
        """GRPOトレーニング用のデータペアを生成

        各軌跡のpromptとcompletionsおよびrewardsを
        trl.GRPOTrainerが期待する形式に変換する。
        """
        pairs: List[Dict[str, Any]] = []
        for t in self._buffer:
            if t.completions and t.rewards:
                pairs.append({
                    "prompt": t.prompt,
                    "completions": t.completions,
                    "rewards": t.rewards,
                })
        return pairs

    def get_reward_training_data(self) -> List[Dict[str, Any]]:
        """報酬モデル学習用のデータを生成

        自己一貫性に基づくペア比較データ:
        best_completionをchosen、それ以外をrejectedとする。
        """
        data: List[Dict[str, Any]] = []
        for t in self._buffer:
            if len(t.completions) < 2 or not t.rewards:
                continue
            best_idx = t.best_completion_idx
            for i, comp in enumerate(t.completions):
                if i != best_idx:
                    data.append({
                        "prompt": t.prompt,
                        "chosen": t.completions[best_idx],
                        "rejected": comp,
                        "chosen_reward": t.rewards[best_idx],
                        "rejected_reward": t.rewards[i],
                    })
        return data

    def clear(self) -> None:
        """バッファをクリア"""
        self._buffer.clear()
        self._iteration_buffers.clear()
        logger.info("TrajectoryBuffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """バッファの統計情報を取得"""
        if not self._buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "num_iterations": 0,
            }
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "mean_reward": self.compute_mean_reward(),
            "success_rate": self.compute_success_rate(),
            "num_iterations": len(self._iteration_buffers),
            "rewards_range": {
                "min": min(t.mean_reward for t in self._buffer),
                "max": max(t.mean_reward for t in self._buffer),
            },
        }
