import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _is_bf16_supported() -> bool:
    """Return True when the active CUDA stack supports bfloat16."""
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except (RuntimeError, AssertionError, TypeError):  # defensive
            return False

    try:
        device_count = torch.cuda.device_count()
    except Exception:  # defensive
        device_count = 0

    for idx in range(device_count):
        try:
            major, _ = torch.cuda.get_device_capability(idx)
        except Exception:  # defensive
            continue
        if major >= 8:  # Ampere (SM80) or newer
            return True
    return False


class TrainingConfig:
    """トレーニング設定クラス"""
    def __init__(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 1,  # 32Bモデル用に削減
        gradient_accumulation_steps: int = 32,  # メモリ効率のため増加
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        eval_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 10,
        output_dir: str = "./outputs",
        fp16: bool = True,
        gradient_checkpointing: bool = True,
        ddp: bool = False,
        local_rank: int = -1,
        world_size: int = 1,
        data_file_path: Optional[str] = None,
        eval_data_file_path: Optional[str] = None,
        replay_data_path: Optional[str] = None,
        replay_mix_ratio: float = 0.1,
        ewc_lambda: float = 0.0,
        max_seq_length: Optional[int] = 256,
        mixed_precision: Optional[str] = None,
        enforce_memory_limits: bool = True,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.ddp = ddp
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_file_path = data_file_path
        self.eval_data_file_path = eval_data_file_path
        self.replay_data_path = replay_data_path
        self.replay_mix_ratio = replay_mix_ratio
        self.ewc_lambda = ewc_lambda
        self.max_seq_length = max_seq_length
        self.mixed_precision = mixed_precision
        self.enforce_memory_limits = enforce_memory_limits

    def effective_batch_size(self) -> int:
        """Return the effective batch size after gradient accumulation."""
        return max(1, int(self.batch_size)) * max(1, int(self.gradient_accumulation_steps))

    def apply_memory_optimizations(
        self,
        target_batch_size: int,
        target_seq_length: Optional[int] = None,
        preserve_effective_batch: bool = True,
        min_gradient_accumulation: Optional[int] = None,
    ) -> Dict[str, Tuple[Optional[int], int]]:
        """Clamp training parameters to reduce memory consumption.

        Returns a mapping of adjusted hyper-parameters for logging purposes.
        Each value is a tuple of (old_value, new_value).
        """

        if target_batch_size < 1:
            raise ValueError("target_batch_size must be at least 1")

        adjustments: Dict[str, Tuple[Optional[int], int]] = {}

        original_batch_size = max(1, int(self.batch_size))
        original_grad_accum = max(1, int(self.gradient_accumulation_steps))
        effective_batch = original_batch_size * original_grad_accum

        if original_batch_size != target_batch_size:
            adjustments["batch_size"] = (original_batch_size, target_batch_size)
            self.batch_size = target_batch_size
        else:
            self.batch_size = target_batch_size

        desired_grad_accum = original_grad_accum
        if preserve_effective_batch:
            desired_grad_accum = max(1, math.ceil(effective_batch / self.batch_size))

        if min_gradient_accumulation is not None:
            desired_grad_accum = max(desired_grad_accum, int(max(1, min_gradient_accumulation)))

        if desired_grad_accum != self.gradient_accumulation_steps:
            adjustments["gradient_accumulation_steps"] = (
                self.gradient_accumulation_steps,
                desired_grad_accum,
            )
            self.gradient_accumulation_steps = desired_grad_accum

        if target_seq_length is not None:
            current_seq_length = getattr(self, "max_seq_length", None)
            if current_seq_length is None or current_seq_length > target_seq_length:
                adjustments["max_seq_length"] = (current_seq_length, target_seq_length)
                self.max_seq_length = target_seq_length

        return adjustments

    def resolved_mixed_precision(self) -> str:
        """Return the preferred mixed precision mode for Accelerate."""
        if self.mixed_precision:
            return self.mixed_precision

        if not getattr(self, "fp16", False):
            return "no"

        if _is_bf16_supported():
            return "bf16"

        return "fp16"


class TextDataset(Dataset):
    """テキストデータセットクラス"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

    def set_max_length(self, max_length: int) -> None:
        """Update the truncation length for future samples."""
        self.max_length = max_length


class StreamingTextDataset(IterableDataset):
    """大規模なテキストファイルを一行ずつ読み込むデータセット"""
    def __init__(self, file_path: str, tokenizer, max_length: int = 256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                sample = None
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = {"text": line}

                sample = self._prepare_sample(record)
                if sample is None:
                    continue
                yield sample

    def set_max_length(self, max_length: int) -> None:
        """Update the truncation length for streaming samples."""
        self.max_length = max_length

    def _prepare_sample(self, record: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(record, dict):
            if "input_ids" in record:
                return self._materialize_from_token_ids(record)

            text = self._extract_text(record)
            if text:
                return self._tokenize_text(text)

        elif isinstance(record, str) and record:
            return self._tokenize_text(record)

        return None

    def _extract_text(self, payload: Dict[str, Any]) -> Optional[str]:
        text = payload.get("text")
        if text:
            return text

        prompt = payload.get("prompt") or payload.get("instruction")
        completion = payload.get("completion") or payload.get("response")
        if prompt or completion:
            return f"{prompt or ''}{completion or ''}".strip()

        messages = payload.get("messages")
        if isinstance(messages, list):
            ordered_segments = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content") or message.get("text")
                if isinstance(content, list):
                    # OpenAI形式: contentは辞書のリスト
                    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    content = "".join(parts)
                if content:
                    speaker = message.get("role")
                    if speaker:
                        ordered_segments.append(f"{speaker}: {content}")
                    else:
                        ordered_segments.append(content)
            if ordered_segments:
                return "\n".join(ordered_segments)

        return None

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        batch = {key: value.squeeze(0) for key, value in tokenized.items()}

        if "attention_mask" not in batch:
            attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
            attention_mask[: batch["input_ids"].ne(self.tokenizer.pad_token_id or 0).sum()] = 1
            batch["attention_mask"] = attention_mask

        batch["labels"] = self._build_labels(batch["input_ids"], batch["attention_mask"])
        return batch

    def _materialize_from_token_ids(self, payload: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        input_ids = payload.get("input_ids")
        if not isinstance(input_ids, (list, tuple)):
            return None

        pad_token = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
        )

        ids_tensor = self._pad_or_truncate_tensor(input_ids, pad_token)

        attention_mask_data = payload.get("attention_mask")
        if isinstance(attention_mask_data, (list, tuple)):
            attention_mask = self._pad_or_truncate_tensor(attention_mask_data, 0, pad_value=0)
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
            if len(input_ids) < self.max_length:
                attention_mask[len(input_ids):] = 0

        labels_data = payload.get("labels")
        if isinstance(labels_data, (list, tuple)):
            labels_tensor = self._pad_or_truncate_tensor(labels_data, -100, pad_value=-100)
        else:
            labels_tensor = self._build_labels(ids_tensor, attention_mask)

        return {
            "input_ids": ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }

    def _pad_or_truncate_tensor(
        self,
        values: Any,
        default_pad: int,
        pad_value: Optional[int] = None,
    ) -> torch.Tensor:
        pad_id = default_pad if pad_value is None else pad_value
        tensor = torch.tensor(list(values), dtype=torch.long)
        if tensor.numel() >= self.max_length:
            return tensor[: self.max_length]

        pad_len = self.max_length - tensor.numel()
        padding = torch.full((pad_len,), pad_id, dtype=torch.long)
        return torch.cat([tensor, padding], dim=0)

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        labels = input_ids.clone()
        if attention_mask is None:
            return labels
        ignore_index = -100
        labels = labels.masked_fill(attention_mask == 0, ignore_index)
        return labels


class CombinedStreamingDataset(IterableDataset):
    """複数のStreamingTextDatasetを結合し、指定された比率でデータを混合するデータセット"""
    def __init__(
        self,
        main_dataset: StreamingTextDataset,
        replay_dataset: StreamingTextDataset,
        mix_ratio: float = 0.1
    ):
        self.main_dataset = main_dataset
        self.replay_dataset = replay_dataset
        self.mix_ratio = mix_ratio

    def __iter__(self):
        main_iter = iter(self.main_dataset)
        replay_iter = iter(self.replay_dataset)

        while True:
            # メインデータから取得
            try:
                yield next(main_iter)
            except StopIteration:
                main_iter = iter(self.main_dataset) # メインデータが尽きたらリセット
                yield next(main_iter) # リセット後、再度取得

            # リプレイデータから取得 (mix_ratioに基づいて)
            if torch.rand(1).item() < self.mix_ratio:
                try:
                    yield next(replay_iter)
                except StopIteration:
                    replay_iter = iter(self.replay_dataset) # リプレイデータが尽きたらリセット
                    yield next(replay_iter) # リセット後、再度取得

    def set_max_length(self, max_length: int) -> None:
        """Propagate sequence length updates to underlying datasets."""
        if hasattr(self.main_dataset, "set_max_length"):
            self.main_dataset.set_max_length(max_length)
        elif hasattr(self.main_dataset, "max_length"):
            self.main_dataset.max_length = max_length

        if hasattr(self.replay_dataset, "set_max_length"):
            self.replay_dataset.set_max_length(max_length)
        elif hasattr(self.replay_dataset, "max_length"):
            self.replay_dataset.max_length = max_length


def prepare_model_for_training(
    model: nn.Module,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True
) -> nn.Module:
    """モデルをトレーニング用に準備"""
    
    # Gradient Checkpointing
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Flash Attention
    if use_flash_attention and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "flash_attention_2"
        logger.info("Enabled Flash Attention 2")
    
    # Disable cache for training
    model.config.use_cache = False
    
    return model


def setup_distributed_training(local_rank: int) -> Tuple[torch.device, int]:
    """分散学習のセットアップ"""
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        logger.info(f"Initialized distributed training: rank {local_rank}/{world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
    
    return device, world_size


def get_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, Any]:
    """オプティマイザとスケジューラを取得"""
    
    # パラメータグループの設定（weight decayの適用を制御）
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def compute_metrics(eval_preds) -> Dict[str, float]:
    """評価メトリクスを計算"""
    predictions, labels = eval_preds
    
    # Perplexityの計算
    loss = nn.CrossEntropyLoss()(
        torch.tensor(predictions).view(-1, predictions.shape[-1]),
        torch.tensor(labels).view(-1)
    )
    perplexity = torch.exp(loss).item()
    
    return {"perplexity": perplexity}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    config: TrainingConfig,
    metrics: Optional[Dict[str, float]] = None
):
    """チェックポイントを保存"""
    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # モデルとトークナイザを保存
    if hasattr(model, "module"):
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    
    # トレーニング状態を保存
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics
    }, os.path.join(checkpoint_dir, "training_state.pt"))
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str
) -> Tuple[int, int]:
    """チェックポイントをロード"""
    training_state = torch.load(
        os.path.join(checkpoint_path, "training_state.pt"),
        map_location="cpu"
    )
    
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    
    epoch = training_state["epoch"]
    step = training_state["step"]
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, step {step})")
    
    return epoch, step


class GradientAccumulator:
    """グラディエント累積のヘルパークラス"""
    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """オプティマイザをステップすべきかどうか"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def get_scale(self) -> float:
        """勾配のスケール係数を取得"""
        return 1.0 / self.accumulation_steps
