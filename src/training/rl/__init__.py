"""
RLAnything - Closed-Loop Dynamic Reinforcement Learning Framework

RLAnythingフレームワーク: 環境(Q)、ポリシー(π)、報酬モデル(r)の
3コンポーネントを閉ループで共最適化する強化学習システム。

Based on: "RLAnything: an Integrated Framework for Closed-Loop Dynamic
Reinforcement Learning" (arXiv:2602.02488)

Components:
    - PolicyTrainer: GRPOベースのポリシー最適化 (trl.GRPOTrainer wrapper)
    - RewardModel: 自己一貫性ベースの報酬モデル学習
    - EnvironmentAdapter: 動的タスク難易度調整
    - TrajectoryBuffer: 軌跡データのサンプリング・保存
    - RewardIntegrator: Outcome + Process報酬の統合
    - RLAnythingOrchestrator: 閉ループ制御オーケストレータ
"""

from .config import RLAnythingConfig
from .policy_trainer import PolicyTrainer
from .reward_model import RewardModel
from .reward_integrator import RewardIntegrator
from .environment_adapter import EnvironmentAdapter
from .trajectory_buffer import Trajectory, TrajectoryBuffer
from .orchestrator import RLAnythingOrchestrator

__all__ = [
    "RLAnythingConfig",
    "PolicyTrainer",
    "RewardModel",
    "RewardIntegrator",
    "EnvironmentAdapter",
    "Trajectory",
    "TrajectoryBuffer",
    "RLAnythingOrchestrator",
]
