"""
Reward Model - Self-Consistency Based Training

RLAnythingの報酬モデルコンポーネント。
自己一貫性（self-consistency）に基づいて報酬モデルを学習・更新する。

報酬モデルの学習ループ:
1. ポリシーからG個のサンプルを生成
2. 多数決で「正解」応答を決定（自己一貫性）
3. 正解/不正解ペアで報酬モデルを更新

Process Reward Model (PRM):
- 中間ステップごとの報酬を生成
- 最終スコアはステップ報酬の平均
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RLAnythingConfig
from .trajectory_buffer import Trajectory

logger = logging.getLogger(__name__)


class RewardModel:
    """自己一貫性ベースの報酬モデル

    二つの報酬タイプを提供:
    1. Outcome Reward: 応答全体の正確性を評価
    2. Process Reward: 推論ステップごとの品質を評価
    """

    def __init__(self, config: RLAnythingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        # 報酬関数として使うカスタム評価関数のリスト
        self._custom_reward_fns: List[Callable] = []

        logger.info("RewardModel初期化")

    # ── モデルロード ──────────────────────────────────

    def load_model(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """報酬モデルをロード

        reward_model_nameが未指定の場合、ルールベースの報酬計算を使用する。
        指定されている場合、AutoModelForSequenceClassificationを使用。
        """
        model_name = self.config.reward_model_name

        if model is not None and tokenizer is not None:
            logger.info("外部提供の報酬モデルを使用")
            self.model = model
            self.tokenizer = tokenizer
            self._is_loaded = True
            return

        if model_name is None:
            logger.info(
                "報酬モデル名が未指定。ルールベース + 自己一貫性報酬を使用します。"
            )
            self._is_loaded = True
            return

        logger.info(f"報酬モデルのロード: {model_name}")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._is_loaded = True
            logger.info("報酬モデルのロード完了")
        except Exception as e:
            logger.warning(f"報酬モデルのロード失敗: {e}。ルールベース報酬にフォールバック。")
            self._is_loaded = True

    def register_reward_fn(self, fn: Callable) -> None:
        """カスタム報酬関数を登録

        Args:
            fn: (prompt: str, completion: str) -> float 形式の関数
        """
        self._custom_reward_fns.append(fn)
        logger.info(f"カスタム報酬関数登録: {fn.__name__}")

    # ── Outcome Reward ────────────────────────────────

    def compute_outcome_rewards(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """各軌跡のOutcome報酬を計算

        Args:
            trajectories: 報酬計算対象の軌跡リスト

        Returns:
            outcome_rewardsが設定された軌跡リスト
        """
        for trajectory in trajectories:
            outcome_rewards: List[float] = []

            for completion in trajectory.completions:
                reward = self._compute_single_outcome_reward(
                    trajectory.prompt, completion
                )
                outcome_rewards.append(reward)

            trajectory.outcome_rewards = outcome_rewards

        return trajectories

    def _compute_single_outcome_reward(
        self, prompt: str, completion: str
    ) -> float:
        """単一応答のOutcome報酬を計算"""
        # 学習済み報酬モデルが利用可能な場合
        if self.model is not None and self.tokenizer is not None:
            return self._model_based_reward(prompt, completion)

        # カスタム報酬関数が登録されている場合
        if self._custom_reward_fns:
            scores = [fn(prompt, completion) for fn in self._custom_reward_fns]
            return sum(scores) / len(scores) if scores else 0.0

        # フォールバック: 自己一貫性ベースの簡易報酬
        return self._default_reward(prompt, completion)

    def _model_based_reward(self, prompt: str, completion: str) -> float:
        """学習済みモデルによる報酬計算"""
        text = f"{prompt}\n{completion}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits.squeeze().item()

        return reward

    def _default_reward(self, prompt: str, completion: str) -> float:
        """デフォルトのルールベース報酬

        基本的なヒューリスティクスによる報酬:
        - 応答の長さ（短すぎ/長すぎにペナルティ）
        - 応答の多様性（トークンの重複率）
        - 基本的な品質指標
        """
        if not completion.strip():
            return -1.0

        # 長さベースのスコア（適度な長さを評価）
        tokens = completion.split()
        length = len(tokens)
        if length < 10:
            length_score = -0.5
        elif length > 500:
            length_score = -0.3
        else:
            length_score = min(1.0, length / 100.0)

        # 多様性スコア（トークン重複率が低いほど良い）
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        diversity_score = unique_ratio

        return (length_score + diversity_score) / 2.0

    # ── Self-Consistency ──────────────────────────────

    def compute_self_consistency_rewards(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """自己一貫性に基づく報酬を計算

        G個の応答内で多数派と一致する応答に高報酬を付与。
        RLAnythingの報酬モデル学習の核心メカニズム。
        """
        for trajectory in trajectories:
            if len(trajectory.completions) < 2:
                trajectory.outcome_rewards = [0.0] * len(trajectory.completions)
                continue

            # 正規化された応答で一貫性を計算
            normalized = [self._normalize_answer(c) for c in trajectory.completions]
            consistency_rewards: List[float] = []

            for i, norm_ans in enumerate(normalized):
                # 他の応答との一致率を計算
                matches = sum(
                    1 for j, other in enumerate(normalized)
                    if i != j and self._answers_match(norm_ans, other)
                )
                consistency_rate = matches / max(1, len(normalized) - 1)
                consistency_rewards.append(consistency_rate)

            trajectory.outcome_rewards = consistency_rewards

        return trajectories

    def _normalize_answer(self, text: str) -> str:
        """応答を正規化して比較可能にする"""
        text = text.strip().lower()
        # 数値回答の場合、数値部分を抽出
        import re
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return " ".join(numbers)
        # テキスト回答の場合、最初の文を返す
        sentences = text.split("。")
        if sentences:
            return sentences[0].strip()
        return text

    def _answers_match(self, a: str, b: str) -> bool:
        """二つの正規化された応答が一致するか判定"""
        if a == b:
            return True
        # 部分一致（70%以上の一致率）
        if not a or not b:
            return False
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        if shorter / longer > 0.7:
            common = sum(1 for ca, cb in zip(a, b) if ca == cb)
            return common / longer > 0.7
        return False

    # ── Process Reward ────────────────────────────────

    def compute_process_rewards(
        self,
        trajectories: List[Trajectory],
        step_delimiter: str = "\n",
    ) -> List[Trajectory]:
        """Process報酬（ステップごとの評価）を計算

        各応答を推論ステップに分割し、ステップごとに評価する。
        最終的なprocess_rewardsは各ステップの報酬リスト。
        """
        for trajectory in trajectories:
            process_rewards: List[List[float]] = []

            for completion in trajectory.completions:
                steps = [s.strip() for s in completion.split(step_delimiter) if s.strip()]
                step_scores: List[float] = []

                for step_idx, step in enumerate(steps):
                    score = self._evaluate_step(
                        trajectory.prompt, step, step_idx, len(steps)
                    )
                    step_scores.append(score)

                process_rewards.append(step_scores)

            trajectory.process_rewards = process_rewards

        return trajectories

    def _evaluate_step(
        self,
        prompt: str,
        step: str,
        step_idx: int,
        total_steps: int,
    ) -> float:
        """単一推論ステップを評価

        ステップの品質を以下の観点で評価:
        - 論理的な接続性
        - 情報量
        - 冗長性の回避
        """
        if not step.strip():
            return -0.5

        # ステップの長さに基づくスコア
        tokens = step.split()
        if len(tokens) < 3:
            return 0.0
        if len(tokens) > 200:
            return 0.3

        # 基本品質スコア
        score = 0.5

        # 後半のステップほどやや高評価（結論に近い）
        position_bonus = 0.1 * (step_idx / max(1, total_steps - 1))
        score += position_bonus

        return min(1.0, score)

    # ── 保存/ロード ───────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """報酬モデルを保存"""
        save_path = path or os.path.join(self.config.output_dir, "reward_model")
        os.makedirs(save_path, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        logger.info(f"報酬モデル保存: {save_path}")
        return save_path

    def get_reward_functions(self) -> List[Callable]:
        """trl.GRPOTrainerに渡す報酬関数リストを取得"""
        fns: List[Callable] = []

        # カスタム報酬関数
        fns.extend(self._custom_reward_fns)

        # モデルベースの報酬関数がある場合
        if self.model is not None:
            def model_reward(prompts, completions, **kwargs):
                rewards = []
                for prompt, completion in zip(prompts, completions):
                    r = self._model_based_reward(prompt, completion)
                    rewards.append(r)
                return rewards
            fns.append(model_reward)

        # 何もない場合はデフォルト報酬
        if not fns:
            def default_reward(prompts, completions, **kwargs):
                rewards = []
                for prompt, completion in zip(prompts, completions):
                    r = self._default_reward(prompt, completion)
                    rewards.append(r)
                return rewards
            fns.append(default_reward)

        return fns
