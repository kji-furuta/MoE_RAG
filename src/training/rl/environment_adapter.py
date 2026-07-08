"""
Environment Adapter - Dynamic Task Difficulty Adjustment

RLAnythingの環境適応コンポーネント。
ポリシーの成功率に基づいてタスク難易度を動的に調整する。

難易度調整ロジック:
    if success_rate < α_low (0.2):  difficulty -= 1  (簡単にする)
    if success_rate > α_high (0.8): difficulty += 1  (難しくする)
    else: 現状維持

Critical Feedback:
    失敗したタスクに対してLLMがフィードバックを生成し、
    次の軌跡のプロンプトに組み込む。
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import RLAnythingConfig
from .trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)


class EnvironmentAdapter:
    """動的タスク難易度調整器

    ポリシーの学習進度に応じてタスク環境を適応させる:
    1. 成功率の監視
    2. 難易度の自動調整
    3. Critical Feedback生成（オプション）
    4. タスクプロンプトの難易度別フィルタリング
    """

    def __init__(self, config: RLAnythingConfig):
        self.config = config
        self.current_difficulty: int = config.env_min_difficulty
        self._difficulty_history: List[Tuple[int, float]] = []  # (iteration, difficulty)
        self._success_rate_history: List[Tuple[int, float]] = []

        # タスクプールを難易度別に管理
        self._task_pools: Dict[int, List[str]] = {}

        # フィードバック生成関数（外部注入可能）
        self._feedback_generator: Optional[Callable] = None

        logger.info(
            f"EnvironmentAdapter初期化: difficulty={self.current_difficulty}, "
            f"range=[{config.env_min_difficulty}, {config.env_max_difficulty}], "
            f"thresholds=({config.env_success_rate_low}, {config.env_success_rate_high})"
        )

    # ── タスクプール管理 ──────────────────────────────

    def register_tasks(
        self, tasks: List[str], difficulty: Optional[int] = None
    ) -> None:
        """タスクプロンプトを登録

        Args:
            tasks: プロンプト文字列のリスト
            difficulty: 難易度レベル。Noneの場合は現在の難易度に登録。
        """
        level = difficulty if difficulty is not None else self.current_difficulty
        level = max(self.config.env_min_difficulty,
                     min(level, self.config.env_max_difficulty))

        if level not in self._task_pools:
            self._task_pools[level] = []
        self._task_pools[level].extend(tasks)

        logger.info(f"タスク登録: difficulty={level}, count={len(tasks)}")

    def register_all_tasks(self, tasks_by_difficulty: Dict[int, List[str]]) -> None:
        """難易度別のタスクを一括登録"""
        for difficulty, tasks in tasks_by_difficulty.items():
            self.register_tasks(tasks, difficulty=difficulty)

    def sample_tasks(
        self, n: int, difficulty: Optional[int] = None
    ) -> List[str]:
        """現在の難易度に適したタスクをサンプリング

        Args:
            n: サンプル数
            difficulty: 指定難易度。Noneの場合は現在の難易度。

        Returns:
            サンプリングされたタスクプロンプト
        """
        level = difficulty if difficulty is not None else self.current_difficulty

        # 指定難易度のプールから取得（存在しない場合は近い難易度から）
        pool = self._get_available_pool(level)
        if not pool:
            logger.warning(
                f"難易度 {level} のタスクプールが空です。全タスクからサンプリングします。"
            )
            pool = [t for tasks in self._task_pools.values() for t in tasks]

        if not pool:
            logger.error("タスクプールが完全に空です。")
            return []

        n = min(n, len(pool))
        sampled = random.sample(pool, n)
        logger.debug(f"タスクサンプリング: difficulty={level}, count={n}")
        return sampled

    def _get_available_pool(self, target_level: int) -> List[str]:
        """指定難易度付近のタスクプールを取得"""
        if target_level in self._task_pools and self._task_pools[target_level]:
            return self._task_pools[target_level]

        # 近い難易度から探索
        for offset in range(1, self.config.env_max_difficulty + 1):
            for candidate in [target_level - offset, target_level + offset]:
                if candidate in self._task_pools and self._task_pools[candidate]:
                    return self._task_pools[candidate]

        return []

    # ── 難易度適応 ────────────────────────────────────

    def adapt(
        self,
        trajectory_buffer: TrajectoryBuffer,
        iteration: int,
    ) -> Dict[str, Any]:
        """成功率に基づいて難易度を動的に調整

        Args:
            trajectory_buffer: 軌跡バッファ
            iteration: 現在のイテレーション番号

        Returns:
            適応結果の辞書
        """
        success_rate = trajectory_buffer.compute_success_rate(
            window=self.config.env_adaptation_window
        )
        old_difficulty = self.current_difficulty

        if success_rate < self.config.env_success_rate_low:
            # 難しすぎる → 難易度を下げる
            self.current_difficulty = max(
                self.config.env_min_difficulty,
                self.current_difficulty - 1,
            )
            action = "decreased"
        elif success_rate > self.config.env_success_rate_high:
            # 簡単すぎる → 難易度を上げる
            self.current_difficulty = min(
                self.config.env_max_difficulty,
                self.current_difficulty + 1,
            )
            action = "increased"
        else:
            action = "maintained"

        self._difficulty_history.append((iteration, self.current_difficulty))
        self._success_rate_history.append((iteration, success_rate))

        result = {
            "action": action,
            "old_difficulty": old_difficulty,
            "new_difficulty": self.current_difficulty,
            "success_rate": success_rate,
            "iteration": iteration,
        }

        logger.info(
            f"環境適応 [iter={iteration}]: success_rate={success_rate:.3f}, "
            f"difficulty: {old_difficulty} → {self.current_difficulty} ({action})"
        )
        return result

    # ── Critical Feedback ─────────────────────────────

    def set_feedback_generator(self, generator: Callable) -> None:
        """フィードバック生成関数を設定

        Args:
            generator: (prompt, failed_completion) -> str 形式の関数
        """
        self._feedback_generator = generator
        logger.info("Critical Feedback生成関数を設定")

    def generate_feedback(
        self, prompt: str, failed_completion: str
    ) -> Optional[str]:
        """失敗した応答に対するCritical Feedbackを生成

        Args:
            prompt: 元のプロンプト
            failed_completion: 失敗した応答

        Returns:
            フィードバック文字列、または生成不可の場合None
        """
        if not self.config.env_enable_critical_feedback:
            return None

        if self._feedback_generator is not None:
            try:
                return self._feedback_generator(prompt, failed_completion)
            except Exception as e:
                logger.warning(f"フィードバック生成失敗: {e}")
                return None

        # デフォルトのフィードバックテンプレート
        return self._default_feedback(prompt, failed_completion)

    def _default_feedback(self, prompt: str, failed_completion: str) -> str:
        """デフォルトのフィードバックテンプレート"""
        return (
            f"前回の回答は不十分でした。以下の点に注意して再度回答してください：\n"
            f"1. 質問の意図を正確に理解する\n"
            f"2. 段階的に推論を進める\n"
            f"3. 回答の根拠を明確にする\n"
            f"元の質問: {prompt[:200]}"
        )

    def augment_prompts_with_feedback(
        self,
        prompts: List[str],
        failed_trajectories: List[Any],
    ) -> List[str]:
        """失敗した軌跡のフィードバックでプロンプトを拡張

        Args:
            prompts: 元のプロンプトリスト
            failed_trajectories: 失敗した軌跡のリスト

        Returns:
            フィードバックが付加されたプロンプトリスト
        """
        augmented: List[str] = []
        failed_map = {
            t.prompt: t for t in failed_trajectories if not t.is_successful
        }

        for prompt in prompts:
            if prompt in failed_map:
                failed_t = failed_map[prompt]
                feedback = self.generate_feedback(
                    prompt, failed_t.completions[0] if failed_t.completions else ""
                )
                if feedback:
                    augmented.append(f"{feedback}\n\n{prompt}")
                    continue
            augmented.append(prompt)

        return augmented

    # ── 統計 ──────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """環境アダプタの統計情報を取得"""
        return {
            "current_difficulty": self.current_difficulty,
            "difficulty_range": {
                "min": self.config.env_min_difficulty,
                "max": self.config.env_max_difficulty,
            },
            "thresholds": {
                "low": self.config.env_success_rate_low,
                "high": self.config.env_success_rate_high,
            },
            "history_length": len(self._difficulty_history),
            "task_pools": {
                level: len(tasks) for level, tasks in self._task_pools.items()
            },
            "difficulty_history": self._difficulty_history[-10:],
            "success_rate_history": self._success_rate_history[-10:],
        }
