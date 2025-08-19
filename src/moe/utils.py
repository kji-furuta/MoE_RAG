"""
MoE System Utilities
MoEシステムのユーティリティ関数
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import yaml
from .exceptions import MoEConfigError, MoEModelError
from .constants import DEFAULT_VOCAB_SIZE, SAFE_VOCAB_LIMIT

logger = logging.getLogger(__name__)


def validate_input_ids(
    input_ids: torch.Tensor,
    vocab_size: int = DEFAULT_VOCAB_SIZE
) -> torch.Tensor:
    """
    入力IDの検証とサニタイズ
    
    Args:
        input_ids: 入力トークンID
        vocab_size: 語彙サイズ
    
    Returns:
        サニタイズされた入力ID
    
    Raises:
        MoEModelError: 入力が無効な場合
    """
    if input_ids is None:
        raise MoEModelError("input_ids cannot be None")
    
    if not isinstance(input_ids, torch.Tensor):
        raise MoEModelError(f"input_ids must be torch.Tensor, got {type(input_ids)}")
    
    if input_ids.numel() == 0:
        raise MoEModelError("input_ids cannot be empty")
    
    # 安全な範囲にクランプ
    safe_limit = min(vocab_size - 1, SAFE_VOCAB_LIMIT)
    
    if input_ids.max() >= vocab_size:
        logger.warning(
            f"Input IDs exceed vocab size: max={input_ids.max().item()}, "
            f"vocab_size={vocab_size}. Clamping to safe range."
        )
    
    return torch.clamp(input_ids, min=0, max=safe_limit)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    設定ファイルの読み込み
    
    Args:
        config_path: 設定ファイルのパス
    
    Returns:
        設定辞書
    
    Raises:
        MoEConfigError: 設定ファイルが読み込めない場合
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise MoEConfigError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise MoEConfigError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Loaded config from {config_path}")
        return config
    
    except Exception as e:
        raise MoEConfigError(f"Failed to load config from {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    設定ファイルの保存
    
    Args:
        config: 設定辞書
        config_path: 保存先パス
    
    Raises:
        MoEConfigError: 設定ファイルが保存できない場合
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise MoEConfigError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Saved config to {config_path}")
    
    except Exception as e:
        raise MoEConfigError(f"Failed to save config to {config_path}: {e}")


def compute_load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    aux_loss_coef: float = 0.01
) -> torch.Tensor:
    """
    ロードバランス損失の計算
    
    Args:
        router_logits: ルーターのロジット
        expert_indices: 選択されたエキスパートのインデックス
        num_experts: エキスパート数
        aux_loss_coef: 補助損失係数
    
    Returns:
        ロードバランス損失
    """
    if router_logits is None or expert_indices is None:
        return torch.tensor(0.0)
    
    # エキスパートごとの選択頻度
    expert_counts = torch.bincount(
        expert_indices.flatten(),
        minlength=num_experts
    ).float()
    
    # 理想的な均等分布
    ideal_count = expert_indices.numel() / num_experts
    
    # L2損失
    balance_loss = torch.sum((expert_counts - ideal_count) ** 2) / ideal_count
    
    return balance_loss * aux_loss_coef


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    利用可能なデバイスを取得
    
    Args:
        prefer_cuda: CUDAを優先するか
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """
    モデルのパラメータ数をカウント
    
    Args:
        model: PyTorchモデル
    
    Returns:
        総パラメータ数
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    学習可能なパラメータ数をカウント
    
    Args:
        model: PyTorchモデル
    
    Returns:
        学習可能パラメータ数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(num: int) -> str:
    """
    数値を人間が読みやすい形式にフォーマット
    
    Args:
        num: 数値
    
    Returns:
        フォーマット済み文字列
    """
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"