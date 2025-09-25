"""
EWC対応フルファインチューニングトレーナー
継続学習のためのEWC損失を統合したトレーナー
"""
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from .full_finetuning import FullFinetuningTrainer
from .ewc_utils import EWCHelper
from .training_utils import TrainingConfig, TextDataset

logger = logging.getLogger(__name__)


class EWCFullFinetuningTrainer(FullFinetuningTrainer):
    """EWC対応のフルファインチューニングトレーナー"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer,
        train_dataset: TextDataset,
        eval_dataset: Optional[TextDataset] = None,
        ewc_helpers: Optional[List[EWCHelper]] = None,
        ewc_lambda: float = 5000.0
    ):
        # 基本属性を設定
        self.model = model
        self.config = config
        # Ensure critical configuration attributes exist with sane defaults
        self._original_gradient_accumulation = self._resolve_config_value("gradient_accumulation_steps", 32)
        self._original_batch_size = self._resolve_config_value("batch_size", 1)
        self._original_effective_batch_size = max(1, int(self._original_batch_size)) * max(1, int(self._original_gradient_accumulation))
        self._resolve_config_value("learning_rate", 2e-5)
        self._resolve_config_value("max_grad_norm", 1.0)
        self._resolve_config_value("logging_steps", 10)
        self._resolve_config_value("eval_steps", 100)
        self._resolve_config_value("save_steps", 500)
        self._resolve_config_value("warmup_steps", 100)
        self._resolve_config_value("output_dir", "./outputs")
        self._resolve_config_value("weight_decay", 0.01)
        self._resolve_config_value("adam_epsilon", 1e-8)
        self._resolve_config_value("max_seq_length", 256)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.ewc_helpers = ewc_helpers or []
        self.ewc_lambda = ewc_lambda

        # Apply memory-focused clamping before preparing the accelerator
        self.memory_optimization_adjustments = {}
        self.max_seq_length = getattr(self.config, "max_seq_length", 256) or 256
        self._apply_memory_optimizations()
        self.mixed_precision_mode = self._resolve_mixed_precision_mode()

        # Acceleratorの初期化（重要！）
        from accelerate import Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision_mode,
            log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
        )

        logger.info(f"Using mixed precision mode: {self.mixed_precision_mode}")
        if self.memory_optimization_adjustments:
            for key, (old_value, new_value) in self.memory_optimization_adjustments.items():
                logger.info(f"Memory optimization adjusted {key}: {old_value} -> {new_value}")
        logger.info(
            "Effective batch size after accumulation: %s (was %s)",
            self.effective_batch_size,
            self._original_effective_batch_size,
        )

        # デバイスの設定
        self.device = self.accelerator.device if self.accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルをデバイスに移動（Acceleratorがprepareで処理するので不要）
        # self.model = self.model.to(self.device)
        
        # EWC使用時のログ
        if self.ewc_helpers:
            logger.info(f"EWC enabled with {len(self.ewc_helpers)} previous tasks")
            logger.info(f"EWC lambda: {self.ewc_lambda}")
        
        # num_train_epochs が未設定の場合は num_epochs を利用
        if not hasattr(self.config, "num_train_epochs"):
            self.config.num_train_epochs = getattr(self.config, "num_epochs", 1)

        # データローダーの準備
        self._prepare_dataloaders()

        # データセットのメタ情報を計算
        self._train_is_iterable = isinstance(self.train_dataset, IterableDataset)
        self._train_dataset_size = self._estimate_dataset_size(self.train_dataset)
        self._train_batches_per_epoch = self._compute_batches_per_epoch(
            dataset_size=self._train_dataset_size,
            is_iterable=self._train_is_iterable,
            fallback_steps=getattr(self.config, "steps_per_epoch", None),
            dataloader=self.train_dataloader
        )

        if self.eval_dataset:
            self._eval_is_iterable = isinstance(self.eval_dataset, IterableDataset)
            self._eval_dataset_size = self._estimate_dataset_size(self.eval_dataset)
            self._eval_batches_per_epoch = self._compute_batches_per_epoch(
                dataset_size=self._eval_dataset_size,
                is_iterable=self._eval_is_iterable,
                fallback_steps=getattr(self.config, "eval_steps_per_epoch", None),
                dataloader=self.eval_dataloader
            )
        else:
            self._eval_is_iterable = False
            self._eval_dataset_size = None
            self._eval_batches_per_epoch = None

        if self._train_is_iterable and self._train_batches_per_epoch is None:
            raise ValueError(
                "Iterable training dataset length is unknown. Provide 'steps_per_epoch' in the training config "
                "or use a dataset with a finite length."
            )

    def _resolve_config_value(self, attr_name: str, default_value):
        """Ensure the training config exposes the requested attribute with a default."""
        if hasattr(self.config, attr_name):
            current_value = getattr(self.config, attr_name)
            if current_value is None:
                setattr(self.config, attr_name, default_value)
                return default_value
            return current_value

        setattr(self.config, attr_name, default_value)
        return default_value

    def _apply_memory_optimizations(self):
        """Clamp batch/sequence settings to keep memory usage predictable."""
        enforce_limits = getattr(self.config, "enforce_memory_limits", True)
        adjustments: Dict[str, Tuple[Optional[int], int]] = {}

        if enforce_limits and hasattr(self.config, "apply_memory_optimizations"):
            min_grad = max(1, int(getattr(self, "_original_gradient_accumulation", 1)))
            adjustments = self.config.apply_memory_optimizations(
                target_batch_size=1,
                target_seq_length=256,
                preserve_effective_batch=True,
                min_gradient_accumulation=min_grad,
            )
        else:
            current_seq_length = getattr(self.config, "max_seq_length", None)
            if current_seq_length is None or current_seq_length > 256:
                adjustments["max_seq_length"] = (current_seq_length, 256)
                self.config.max_seq_length = 256

        self.memory_optimization_adjustments = adjustments
        self.max_seq_length = getattr(self.config, "max_seq_length", 256) or 256

        # Update cached effective batch size for logging
        if hasattr(self.config, "effective_batch_size"):
            self.effective_batch_size = self.config.effective_batch_size()
        else:
            self.effective_batch_size = max(1, int(self.config.batch_size)) * max(
                1, int(self.config.gradient_accumulation_steps)
            )

        # Propagate truncation limits to datasets/tokenizer
        for dataset in filter(None, [self.train_dataset, self.eval_dataset]):
            self._propagate_max_length(dataset)

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "model_max_length"):
                try:
                    if (
                        self.tokenizer.model_max_length is None
                        or self.tokenizer.model_max_length > self.max_seq_length
                    ):
                        self.tokenizer.model_max_length = self.max_seq_length
                except TypeError:
                    self.tokenizer.model_max_length = self.max_seq_length
            if hasattr(self.tokenizer, "init_kwargs"):
                try:
                    self.tokenizer.init_kwargs["model_max_length"] = self.max_seq_length
                except (AttributeError, TypeError):
                    pass

    def _propagate_max_length(self, dataset):
        """Update dataset objects so they respect the capped sequence length."""
        if dataset is None:
            return
        if hasattr(dataset, "set_max_length"):
            dataset.set_max_length(self.max_seq_length)
        elif hasattr(dataset, "max_length"):
            dataset.max_length = self.max_seq_length

    def _resolve_mixed_precision_mode(self) -> str:
        """Select an appropriate mixed precision mode for Accelerate."""
        candidate = None
        if hasattr(self.config, "resolved_mixed_precision"):
            resolver = getattr(self.config, "resolved_mixed_precision")
            try:
                candidate = resolver()
            except TypeError:
                candidate = resolver
        elif getattr(self.config, "mixed_precision", None):
            candidate = self.config.mixed_precision

        if candidate in {"no", "fp16", "bf16"}:
            return candidate

        if getattr(self.config, "fp16", False):
            return "fp16"

        return "no"
    
    def _prepare_dataloaders(self):
        """データローダーの準備"""
        dataloader_kwargs = dict(
            batch_size=self.config.batch_size,
            num_workers=0,
            pin_memory=True,
        )

        if isinstance(self.train_dataset, IterableDataset):
            # IterableDataset は DataLoader に shuffle を渡せない
            self.train_dataloader = DataLoader(
                self.train_dataset,
                **dataloader_kwargs,
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                shuffle=True,
                **dataloader_kwargs,
            )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

    def _estimate_dataset_size(self, dataset) -> Optional[int]:
        """データセットのおおよそのサイズを推定"""
        if dataset is None:
            return None

        if isinstance(dataset, IterableDataset):
            return self._estimate_iterable_dataset_size(dataset)

        try:
            return len(dataset)
        except TypeError:
            return None

    def _estimate_iterable_dataset_size(self, dataset: IterableDataset) -> Optional[int]:
        """IterableDataset の場合にサイズを推定"""
        file_path = getattr(dataset, "file_path", None)
        if file_path and os.path.exists(file_path):
            valid_records = 0
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("text"):
                            valid_records += 1
            except OSError as exc:
                logger.warning(f"Failed to estimate dataset size from {file_path}: {exc}")
                return None
            return valid_records

        # 予備的なサイズ情報がある場合はそれを使用
        for attr_name in ("length", "approx_len", "size"):
            attr_value = getattr(dataset, attr_name, None)
            if isinstance(attr_value, int) and attr_value >= 0:
                return attr_value
        return None

    def _compute_batches_per_epoch(
        self,
        dataset_size: Optional[int],
        is_iterable: bool,
        fallback_steps: Optional[int],
        dataloader: Optional[DataLoader] = None
    ) -> Optional[int]:
        """1エポックあたりのバッチ数を計算"""
        if is_iterable:
            if fallback_steps is not None:
                return int(fallback_steps)
            if dataset_size is not None:
                return max(1, math.ceil(dataset_size / max(1, self.config.batch_size)))
            return None

        # Map-style Dataset の場合は DataLoader の長さを使用
        try:
            loader = dataloader if dataloader is not None else self.train_dataloader
            return len(loader)
        except TypeError:
            return dataset_size
    
    def _truncate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slice tensors so the last dimension does not exceed max_seq_length."""
        if self.max_seq_length is None or not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.dim() == 0 or tensor.size(-1) <= self.max_seq_length:
            return tensor
        slices = (slice(None),) * (tensor.dim() - 1) + (slice(0, self.max_seq_length),)
        return tensor[slices].contiguous()

    def _move_batch_to_device(self, batch, device: torch.device):
        """再帰的にバッチデータを指定デバイスへ移動"""
        if isinstance(batch, torch.Tensor):
            batch = self._truncate_tensor(batch)
            if batch.device != device:
                batch = batch.to(device, non_blocking=True)
            return batch
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            converted = [self._move_batch_to_device(v, device) for v in batch]
            return tuple(converted) if isinstance(batch, tuple) else converted
        return batch

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """EWC損失を含む損失計算"""
        # 8bit量子化やLoRAのCPUオフロード構成に合わせて入力デバイスを決定
        unwrapped_model = model
        if getattr(self, "accelerator", None) is not None:
            try:
                unwrapped_model = self.accelerator.unwrap_model(model)
            except AttributeError:
                pass

        target_device = self.device
        try:
            embeddings = unwrapped_model.get_input_embeddings()
        except AttributeError:
            embeddings = None

        if embeddings is not None:
            embedding_weight = getattr(embeddings, "weight", None)
            if isinstance(embedding_weight, torch.Tensor):
                target_device = embedding_weight.device
        elif hasattr(unwrapped_model, "device"):
            target_device = unwrapped_model.device

        # デバイスに転送
        batch = self._move_batch_to_device(batch, target_device)

        # 通常の損失計算
        outputs = model(**batch)

        # outputsが辞書の場合とCausalLMOutputの場合を処理
        if isinstance(outputs, dict):
            task_loss = outputs.get('loss')
        else:
            task_loss = outputs.loss

        # task_lossがNoneの場合のエラーハンドリング
        if task_loss is None:
            raise ValueError("Model did not return a loss. Ensure labels are provided in the batch.")

        # EWC損失の計算
        ewc_loss = torch.tensor(0.0, device=self.device)
        if self.ewc_helpers:
            for helper in self.ewc_helpers:
                try:
                    ewc_loss += helper.compute_ewc_loss(model)
                except Exception as e:
                    logger.warning(f"Error computing EWC loss: {e}")
            
            ewc_loss = self.ewc_lambda * ewc_loss
            
            # メトリクスの記録
            if hasattr(self, 'current_step'):
                # task_lossがTensorであることを確認
                task_loss_value = task_loss.item() if hasattr(task_loss, 'item') else float(task_loss)
                ewc_loss_value = ewc_loss.item() if hasattr(ewc_loss, 'item') else float(ewc_loss)
                self.log_metrics({
                    "loss/task": task_loss_value,
                    "loss/ewc": ewc_loss_value,
                    "loss/total": task_loss_value + ewc_loss_value
                })
        
        return task_loss + ewc_loss
    
    def train(self):
        """EWC対応の学習実行"""
        logger.info("Starting EWC-enabled training...")
        if self._train_dataset_size is not None:
            logger.info(f"Number of training examples: {self._train_dataset_size}")
        elif self._train_is_iterable:
            logger.info("Number of training examples: unknown (streaming IterableDataset)")
        else:
            logger.info("Number of training examples: unavailable")

        if self._train_batches_per_epoch is not None:
            logger.info(f"Batches per epoch: {self._train_batches_per_epoch}")
        elif self._train_is_iterable:
            logger.info("Batches per epoch: unlimited (will run until iterator is exhausted)")

        total_epochs = int(self.config.num_train_epochs)
        logger.info(f"Number of epochs: {total_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Max sequence length: {self.max_seq_length}")

        # モデルを学習モードに
        self.model.train()
        
        # オプティマイザとスケジューラーの準備
        self._setup_optimizer_and_scheduler()
        
        # 学習ループ
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(total_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")
            
            epoch_loss = 0.0
            steps_in_epoch = 0

            expected_batches = self._train_batches_per_epoch
            progress_bar = tqdm(
                total=expected_batches,
                desc=f"Training Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True
            )

            data_iterator = iter(self.train_dataloader)

            while True:
                if expected_batches is not None and steps_in_epoch >= expected_batches:
                    break
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    break
                
                steps_in_epoch += 1
                # 勾配の累積
                with self.accelerator.accumulate(self.model):
                    # 損失計算
                    loss = self.compute_loss(self.model, batch)
                    
                    # バックプロパゲーション
                    self.accelerator.backward(loss)
                    
                    # 勾配クリッピング
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # オプティマイザステップ
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # メトリクスの更新
                    epoch_loss += loss.item()
                    
                    # 進捗バーの更新
                    if self.config.logging_steps and (
                        steps_in_epoch % self.config.logging_steps == 0 or steps_in_epoch == 1
                    ):
                        current_loss = epoch_loss / max(1, steps_in_epoch)
                        try:
                            current_lr = self.scheduler.get_last_lr()[0]
                        except (AttributeError, IndexError):
                            current_lr = self.optimizer.param_groups[0]["lr"]
                        progress_bar.set_postfix({
                            'loss': f"{current_loss:.4f}",
                            'lr': f"{current_lr:.2e}"
                        })
                    
                    global_step += 1
                    progress_bar.update(1)
                    
                    # メモリ管理
                    if torch.cuda.is_available() and steps_in_epoch % 50 == 0:
                        torch.cuda.empty_cache()

            progress_bar.close()
            
            # エポック終了時の処理
            if steps_in_epoch == 0:
                logger.warning("No training steps were executed in this epoch.")
                continue

            avg_epoch_loss = epoch_loss / steps_in_epoch
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")
            
            # 評価の実行
            if self.eval_dataset and (epoch + 1) % self.config.eval_steps == 0:
                eval_loss = self.evaluate()
                logger.info(f"Evaluation loss: {eval_loss:.4f}")
                
                # ベストモデルの保存
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(suffix="best")
            
            # チェックポイントの保存
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(epoch, global_step)
        
        # 最終モデルの保存
        self.save_model(suffix="final")
        logger.info("Training completed!")
    
    def evaluate(self) -> float:
        """評価の実行"""
        if not self.eval_dataset:
            return 0.0
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0
        total_steps = 0

        expected_batches = self._eval_batches_per_epoch
        progress_bar = tqdm(
            total=expected_batches,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=True
        )

        data_iterator = iter(self.eval_dataloader)

        with torch.no_grad():
            while True:
                if expected_batches is not None and total_steps >= expected_batches:
                    break
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    break

                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                total_steps += 1
                progress_bar.update(1)

        progress_bar.close()
        
        if total_steps == 0:
            logger.warning("Evaluation dataset produced no batches.")
            avg_loss = 0.0
        else:
            avg_loss = total_loss / total_steps
        self.model.train()
        
        return avg_loss
    
    def save_model(self, suffix: str = ""):
        """モデルの保存"""
        if suffix:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{suffix}")
        else:
            save_path = self.config.output_dir
        
        os.makedirs(save_path, exist_ok=True)
        
        # モデルとトークナイザーの保存
        logger.info(f"Saving model to {save_path}")
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_path,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model)
        )
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        # トレーニング情報の保存
        training_info = {
            "ewc_enabled": len(self.ewc_helpers) > 0,
            "ewc_lambda": self.ewc_lambda,
            "num_previous_tasks": len(self.ewc_helpers),
            "training_config": self.config.__dict__
        }
        
        import json
        with open(os.path.join(save_path, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=2)
    
    def save_checkpoint(self, epoch: int, global_step: int):
        """チェックポイントの保存"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-epoch-{epoch}"
        )
        
        self.save_model(suffix=f"epoch-{epoch}")
        
        # オプティマイザとスケジューラーの状態も保存
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        torch.save(
            checkpoint,
            os.path.join(checkpoint_path, "trainer_state.pt")
        )
        
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """メトリクスのログ記録"""
        if self.accelerator.is_main_process:
            # Weights & Biasesへのログ（環境変数が設定されている場合）
            if os.environ.get("WANDB_API_KEY"):
                try:
                    import wandb
                    wandb.log(metrics)
                except ImportError:
                    pass
            
            # コンソールへの出力（デバッグ用）
            if hasattr(self, 'current_step') and self.current_step % 100 == 0:
                logger.debug(f"Step {self.current_step}: {metrics}")
    
    def _setup_optimizer_and_scheduler(self):
        """オプティマイザとスケジューラーのセットアップ"""
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # パラメータグループの設定
        weight_decay = self._resolve_config_value("weight_decay", 0.01)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # オプティマイザ
        learning_rate = self._resolve_config_value("learning_rate", 2e-5)
        adam_epsilon = self._resolve_config_value("adam_epsilon", 1e-8)
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
        
        # スケジューラー
        batches_per_epoch = self._train_batches_per_epoch
        if batches_per_epoch is None:
            try:
                batches_per_epoch = len(self.train_dataloader)
            except TypeError as exc:
                raise ValueError(
                    "Unable to determine batches per epoch. Provide 'steps_per_epoch' in the training config "
                    "for iterable datasets."
                ) from exc

        updates_per_epoch = math.ceil(batches_per_epoch / max(1, self.config.gradient_accumulation_steps))
        total_epochs = int(getattr(self.config, "num_train_epochs", 1))
        num_training_steps = max(1, updates_per_epoch * total_epochs)

        warmup_ratio = getattr(self.config, "warmup_ratio", None)
        if warmup_ratio is not None:
            num_warmup_steps = int(num_training_steps * warmup_ratio)
        else:
            warmup_steps = getattr(self.config, "warmup_steps", 0)
            num_warmup_steps = min(max(0, warmup_steps), num_training_steps - 1 if num_training_steps > 1 else 0)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Acceleratorでラップ
        self.model, self.optimizer, self.train_dataloader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.scheduler
            )
