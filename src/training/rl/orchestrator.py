"""
RLAnything Orchestrator - Closed-Loop Dynamic RL Controller

3つのコンポーネント（Policy, Reward, Environment）を閉ループで
共最適化するオーケストレータ。

閉ループサイクル:
    1. 環境からタスクをサンプリング
    2. ポリシーでG個の応答を生成
    3. 報酬モデルでOutcome + Process報酬を計算
    4. 統合報酬を計算し軌跡バッファに保存
    5. ポリシーをGRPOで更新
    6. 報酬モデルを自己一貫性で更新（定期的）
    7. 環境の難易度を適応（定期的）
    8. 収束判定 → 繰り返しまたは終了
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset

from .config import RLAnythingConfig
from .environment_adapter import EnvironmentAdapter
from .policy_trainer import PolicyTrainer
from .reward_integrator import RewardIntegrator
from .reward_model import RewardModel
from .trajectory_buffer import Trajectory, TrajectoryBuffer

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


class RLAnythingOrchestrator:
    """RLAnything 閉ループオーケストレータ

    メインの実行エントリポイント。全コンポーネントの初期化・制御・
    イテレーション管理・収束判定・結果保存を統括する。
    """

    def __init__(self, config: RLAnythingConfig):
        self.config = config

        # コンポーネント初期化
        self.policy = PolicyTrainer(config)
        self.reward_model = RewardModel(config)
        self.reward_integrator = RewardIntegrator(config)
        self.environment = EnvironmentAdapter(config)
        self.trajectory_buffer = TrajectoryBuffer(
            capacity=config.trajectories_per_iteration * config.num_iterations * 2
        )

        # 実行状態
        self.current_iteration: int = 0
        self.is_running: bool = False
        self._metrics_history: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._best_mean_reward: float = float("-inf")
        self._patience_counter: int = 0

        # コールバック
        self._on_iteration_complete: Optional[Callable] = None
        self._status_callback: Optional[Callable] = None

        logger.info(f"RLAnythingOrchestrator初期化: {config.num_iterations}イテレーション")

    # ── セットアップ ──────────────────────────────────

    def setup(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        reward_model: Optional[Any] = None,
        reward_tokenizer: Optional[Any] = None,
    ) -> None:
        """全コンポーネントを初期化

        Args:
            model: 事前ロード済みのポリシーモデル
            tokenizer: 事前ロード済みのトークナイザ
            reward_model: 事前ロード済みの報酬モデル
            reward_tokenizer: 事前ロード済みの報酬モデル用トークナイザ
        """
        logger.info("=== RLAnything セットアップ開始 ===")
        self._update_status("セットアップ開始: モデルをロード中...")

        # 1. ポリシーモデルのロード
        self.policy.load_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            status_callback=self._status_callback,
        )

        # 2. 報酬モデルのロード
        self.reward_model.load_model(model=reward_model, tokenizer=reward_tokenizer)

        # 3. 設定バリデーション
        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"設定警告: {w}")

        logger.info("=== RLAnything セットアップ完了 ===")

    def set_callbacks(
        self,
        on_iteration_complete: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
    ) -> None:
        """コールバック関数を設定"""
        self._on_iteration_complete = on_iteration_complete
        self._status_callback = status_callback

    # ── メイン実行ループ ──────────────────────────────

    def run(self) -> Dict[str, Any]:
        """閉ループ強化学習を実行

        Returns:
            最終結果の辞書
        """
        self.is_running = True
        self._start_time = time.time()

        logger.info("=" * 60)
        logger.info("RLAnything 閉ループ学習開始")
        logger.info(f"  モデル: {self.config.model_name}")
        logger.info(f"  イテレーション数: {self.config.num_iterations}")
        logger.info(f"  GRPO生成数: {self.config.grpo_num_generations}")
        logger.info(f"  報酬重み: outcome={self.config.reward_outcome_weight}, "
                    f"process={self.config.reward_process_weight}")
        logger.info("=" * 60)

        try:
            for iteration in range(self.config.num_iterations):
                self.current_iteration = iteration
                self._update_status(f"イテレーション {iteration + 1}/{self.config.num_iterations}")

                # 1イテレーションを実行
                metrics = self._run_iteration(iteration)
                self._metrics_history.append(metrics)

                # コールバック
                if self._on_iteration_complete:
                    self._on_iteration_complete(iteration, metrics)

                # 収束判定
                if self._check_convergence(metrics):
                    logger.info(f"収束判定により学習を早期終了 (iter={iteration})")
                    break

        except Exception as e:
            logger.error(f"RLAnything実行エラー: {e}", exc_info=True)
            self.is_running = False
            raise
        finally:
            self.is_running = False

        # 最終結果
        result = self._compile_results()
        self._save_results(result)

        logger.info("=" * 60)
        logger.info("RLAnything 閉ループ学習完了")
        logger.info(f"  実行時間: {result['elapsed_time']:.1f}秒")
        logger.info(f"  最終平均報酬: {result['final_mean_reward']:.4f}")
        logger.info(f"  最終成功率: {result['final_success_rate']:.4f}")
        logger.info("=" * 60)

        return result

    def _run_iteration(self, iteration: int) -> Dict[str, Any]:
        """1イテレーション（閉ループ1サイクル）を実行"""
        iter_start = time.time()
        metrics: Dict[str, Any] = {"iteration": iteration}

        logger.info(f"--- イテレーション {iteration} 開始 ---")

        # Step 1: 環境からタスクをサンプリング
        prompts = self.environment.sample_tasks(
            self.config.trajectories_per_iteration
        )
        if not prompts:
            logger.warning("タスクプールが空。既存プロンプトを再利用します。")
            prompts = self._get_fallback_prompts()

        metrics["num_prompts"] = len(prompts)

        # Step 2: Critical Feedbackでプロンプトを拡張（前回の失敗がある場合）
        if iteration > 0 and self.config.env_enable_critical_feedback:
            prev_trajectories = self.trajectory_buffer.get_iteration_trajectories(
                iteration - 1
            )
            failed = [t for t in prev_trajectories if not t.is_successful]
            if failed:
                prompts = self.environment.augment_prompts_with_feedback(
                    prompts, failed
                )

        # Step 3: ポリシーでグループサンプリング
        trajectories = self.policy.generate_group(
            prompts=prompts,
            num_generations=self.config.grpo_num_generations,
        )
        metrics["num_trajectories"] = len(trajectories)

        # Step 4: Outcome報酬を計算
        if self.config.reward_self_consistency_k > 0:
            trajectories = self.reward_model.compute_self_consistency_rewards(
                trajectories
            )
        else:
            trajectories = self.reward_model.compute_outcome_rewards(trajectories)

        # Step 5: Process報酬を計算
        trajectories = self.reward_model.compute_process_rewards(trajectories)

        # Step 6: 統合報酬を計算
        trajectories = self.reward_integrator.integrate(trajectories)

        # 難易度情報をメタデータに付加
        for t in trajectories:
            t.difficulty_level = self.environment.current_difficulty

        # Step 7: 軌跡バッファに保存
        self.trajectory_buffer.add_batch(trajectories, iteration=iteration)

        # Step 8: メトリクス計算
        mean_reward = self.trajectory_buffer.compute_mean_reward(
            window=self.config.env_adaptation_window
        )
        success_rate = self.trajectory_buffer.compute_success_rate(
            window=self.config.env_adaptation_window
        )
        metrics["mean_reward"] = mean_reward
        metrics["success_rate"] = success_rate
        metrics["difficulty"] = self.environment.current_difficulty

        # Step 9: ポリシー更新（GRPOトレーニング）
        policy_metrics = self._update_policy(trajectories)
        metrics["policy"] = policy_metrics

        # Step 10: 報酬モデル更新（定期的）
        if (iteration + 1) % self.config.reward_update_interval == 0:
            reward_metrics = self._update_reward_model()
            metrics["reward_model"] = reward_metrics

        # Step 11: 環境適応（定期的）
        if (iteration + 1) % self.config.environment_adapt_interval == 0:
            env_metrics = self.environment.adapt(
                self.trajectory_buffer, iteration
            )
            metrics["environment"] = env_metrics

        metrics["elapsed_seconds"] = time.time() - iter_start
        logger.info(
            f"--- イテレーション {iteration} 完了: "
            f"mean_reward={mean_reward:.4f}, success_rate={success_rate:.4f}, "
            f"difficulty={self.environment.current_difficulty}, "
            f"time={metrics['elapsed_seconds']:.1f}s ---"
        )

        return metrics

    # ── コンポーネント更新 ────────────────────────────

    def _update_policy(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """ポリシーをGRPOで更新"""
        try:
            # 軌跡データからtrl.GRPOTrainer用のデータセットを構築
            training_data = []
            for t in trajectories:
                if t.completions and t.rewards:
                    training_data.append({"prompt": t.prompt})

            if not training_data:
                return {"status": "skipped", "reason": "no_training_data"}

            dataset = Dataset.from_list(training_data)
            reward_fns = self.reward_model.get_reward_functions()

            # GRPOTrainerをセットアップして実行
            self.policy.setup_grpo_trainer(dataset, reward_fns)
            result = self.policy.train_step()

            return {"status": "updated", **result}

        except Exception as e:
            logger.warning(f"ポリシー更新失敗: {e}")
            return {"status": "failed", "error": str(e)}

    def _update_reward_model(self) -> Dict[str, Any]:
        """報酬モデルを自己一貫性データで更新"""
        training_data = self.trajectory_buffer.get_reward_training_data()
        if not training_data:
            return {"status": "skipped", "reason": "no_preference_data"}

        logger.info(f"報酬モデル更新: {len(training_data)}ペア")
        # 報酬モデルの更新は将来的にフルトレーニングループを実装
        # 現時点ではデータ蓄積のみ
        return {
            "status": "data_collected",
            "num_pairs": len(training_data),
        }

    # ── 収束判定 ──────────────────────────────────────

    def _check_convergence(self, metrics: Dict[str, Any]) -> bool:
        """収束を判定"""
        mean_reward = metrics.get("mean_reward", 0.0)

        # 改善判定
        if mean_reward > self._best_mean_reward + self.config.convergence_threshold:
            self._best_mean_reward = mean_reward
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        # 早期停止
        if self._patience_counter >= self.config.early_stopping_patience:
            logger.info(
                f"早期停止: {self.config.early_stopping_patience}イテレーション "
                f"改善なし (best={self._best_mean_reward:.4f})"
            )
            return True

        return False

    # ── ユーティリティ ────────────────────────────────

    def _get_fallback_prompts(self) -> List[str]:
        """フォールバック用のプロンプトを生成"""
        # バッファから既存プロンプトを再利用
        existing = list(set(t.prompt for t in self.trajectory_buffer.sample(
            min(self.config.trajectories_per_iteration, len(self.trajectory_buffer))
        )))
        if existing:
            return existing

        # 完全なフォールバック
        return ["以下の質問に回答してください。"]

    def _update_status(self, message: str) -> None:
        """ステータスを更新（コールバック経由）"""
        if self._status_callback:
            self._status_callback(message)
        logger.info(f"Status: {message}")

    def _compile_results(self) -> Dict[str, Any]:
        """最終結果をコンパイル"""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        buffer_stats = self.trajectory_buffer.get_stats()
        env_stats = self.environment.get_stats()

        return {
            "status": "completed",
            "model_name": self.config.model_name,
            "num_iterations_completed": self.current_iteration + 1,
            "num_iterations_planned": self.config.num_iterations,
            "elapsed_time": elapsed,
            "final_mean_reward": buffer_stats["mean_reward"],
            "final_success_rate": buffer_stats["success_rate"],
            "best_mean_reward": self._best_mean_reward,
            "buffer_stats": buffer_stats,
            "environment_stats": env_stats,
            "metrics_history": self._metrics_history,
            "config": self.config.to_dict(),
            "timestamp": datetime.now(JST).isoformat(),
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """結果をJSONファイルに保存"""
        results_dir = os.path.join(self.config.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"rlanything_results_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"結果保存: {filepath}")

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """チェックポイントを保存"""
        ckpt_path = path or os.path.join(
            self.config.output_dir,
            f"checkpoint_iter{self.current_iteration}",
        )
        os.makedirs(ckpt_path, exist_ok=True)

        # ポリシーアダプタ保存
        self.policy.save_adapter(os.path.join(ckpt_path, "policy"))

        # 報酬モデル保存
        self.reward_model.save(os.path.join(ckpt_path, "reward"))

        # 状態保存
        state = {
            "iteration": self.current_iteration,
            "best_mean_reward": self._best_mean_reward,
            "patience_counter": self._patience_counter,
            "metrics_history": self._metrics_history,
            "environment_stats": self.environment.get_stats(),
            "buffer_stats": self.trajectory_buffer.get_stats(),
        }
        with open(os.path.join(ckpt_path, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"チェックポイント保存: {ckpt_path}")
        return ckpt_path

    # ── ステータス取得 ────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """現在の実行ステータスを取得"""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return {
            "is_running": self.is_running,
            "current_iteration": self.current_iteration,
            "total_iterations": self.config.num_iterations,
            "elapsed_seconds": elapsed,
            "buffer_size": len(self.trajectory_buffer),
            "current_difficulty": self.environment.current_difficulty,
            "best_mean_reward": self._best_mean_reward,
            "patience_counter": self._patience_counter,
            "latest_metrics": self._metrics_history[-1] if self._metrics_history else None,
        }
