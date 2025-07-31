import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Dict, Any, Optional, List, Tuple
import logging
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json

logger = logging.getLogger(__name__)

class TrainingConfig:
    """トレーニング設定クラス"""
    def __init__(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
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
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
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


class TextDataset(Dataset):
    """テキストデータセットクラス"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
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


class StreamingTextDataset(IterableDataset):
    """大規模なテキストファイルを一行ずつ読み込むデータセット"""
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        tokenized_inputs = self.tokenizer(
                            text,
                            max_length=self.max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        # バッチ次元を削除して辞書の各値をTensorにする
                        yield {k: v.squeeze(0) for k, v in tokenized_inputs.items()}
                except json.JSONDecodeError:
                    # 不正なJSON行はスキップ
                    continue


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