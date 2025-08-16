import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm
import os
from accelerate import Accelerator

from .training_utils import (
    TrainingConfig,
    TextDataset,
    StreamingTextDataset,
    CombinedStreamingDataset,
    prepare_model_for_training,
    setup_distributed_training,
    get_optimizer_and_scheduler,
    save_checkpoint,
    load_checkpoint,
    GradientAccumulator
)
from .ewc_utils import EWCHelper
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class FullFinetuningTrainer:
    """フルファインチューニングのトレーナー"""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        use_accelerate: bool = True
    ):
        self.base_model = model
        self.config = config
        self.use_accelerate = use_accelerate
        
        # Acceleratorの初期化
        if use_accelerate:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision="fp16" if config.fp16 else "no",
                log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
            )
            self.device = self.accelerator.device
        else:
            # 通常のDDP/DPの場合
            self.device, self.world_size = setup_distributed_training(config.local_rank)
            self.accelerator = None
        
        # モデルの準備
        self._prepare_model()

        self.ewc_helper = None
        if self.config.ewc_lambda > 0:
            self.ewc_helper = EWCHelper(self.model, self.device)
        
    def _prepare_model(self):
        """モデルの準備"""
        # モデルとトークナイザーのロード
        if self.base_model.model is None:
            self.base_model.load_model()
            self.base_model.load_tokenizer()
        
        self.model = self.base_model.model
        self.tokenizer = self.base_model.tokenizer
        
        # モデルの準備（トレーニング用）
        self.model = prepare_model_for_training(
            self.model,
            gradient_checkpointing=self.config.gradient_checkpointing,
            use_flash_attention=True
        )
        
        # 分散学習の設定
        if not self.use_accelerate:
            if self.config.ddp and self.config.local_rank != -1:
                self.model = self.model.to(self.device)
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=False
                )
                logger.info("Using DistributedDataParallel")
            elif torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
                self.model = self.model.to(self.device)
                logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            else:
                self.model = self.model.to(self.device)
        
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None
    ):
        """トレーニングの実行"""
        
        # データセットの準備
        if self.config.data_file_path:
            train_dataset = StreamingTextDataset(self.config.data_file_path, self.tokenizer)
        else:
            raise ValueError("Training data file path is required")

        if self.config.replay_data_path:
            replay_dataset = StreamingTextDataset(self.config.replay_data_path, self.tokenizer)
            train_dataset = CombinedStreamingDataset(
                main_dataset=train_dataset,
                replay_dataset=replay_dataset,
                mix_ratio=self.config.replay_mix_ratio
            )
        
        eval_dataset = None
        if self.config.eval_data_file_path:
            eval_dataset = StreamingTextDataset(self.config.eval_data_file_path, self.tokenizer)
        
        # DataLoaderの準備
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,
                pin_memory=True
            )
        
        # オプティマイザーとスケジューラーの準備
        # StreamingDatasetのため、num_training_stepsは概算
        # 実際のエポック数はデータサイズに依存
        num_training_steps = (len(train_dataloader) if hasattr(train_dataloader, '__len__') else 1000) * self.config.num_epochs // self.config.gradient_accumulation_steps
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model,
            self.config,
            num_training_steps
        )
        
        # Acceleratorによる準備
        if self.use_accelerate:
            self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, scheduler
            )
            if eval_dataloader:
                eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # EWCのフィッシャー行列計算
        if self.ewc_helper:
            logger.info("Computing Fisher matrix for EWC...")
            fisher_dataloader = DataLoader(
                train_dataset, # Using train_dataset for Fisher computation
                batch_size=self.config.batch_size,
                num_workers=4,
                pin_memory=True
            )
            if self.use_accelerate:
                fisher_dataloader = self.accelerator.prepare(fisher_dataloader)
            # 最適化されたFisher matrix計算（最大50バッチに制限）
            self.ewc_helper.compute_fisher_matrix(fisher_dataloader, max_batches=50)
            logger.info("Fisher matrix computation complete.")
        
        # チェックポイントからの再開
        start_epoch = 0
        global_step = 0
        if resume_from_checkpoint:
            start_epoch, global_step = load_checkpoint(
                self.model, optimizer, scheduler, resume_from_checkpoint
            )
        
        # トレーニングループの開始
        logger.info("Starting training...")
        self.model.train()
        
        accumulator = GradientAccumulator(self.config.gradient_accumulation_steps)
        best_eval_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not (self.config.local_rank in [-1, 0])
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward pass
                if self.use_accelerate:
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        if self.ewc_helper and self.config.ewc_lambda > 0:
                            ewc_loss = self.ewc_helper.compute_ewc_loss(self.model)
                            loss = loss + self.config.ewc_lambda * ewc_loss
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    # 通常の勾配計算/更新
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    if self.ewc_helper and self.config.ewc_lambda > 0:
                        ewc_loss = self.ewc_helper.compute_ewc_loss(self.model)
                        loss = loss + self.config.ewc_lambda * ewc_loss
                    
                    scaled_loss = loss * accumulator.get_scale()
                    
                    if self.config.fp16:
                        from torch.cuda.amp import autocast, GradScaler
                        scaler = GradScaler()
                        with autocast():
                            scaled_loss.backward()
                    else:
                        scaled_loss.backward()
                    
                    if accumulator.should_step():
                        if self.config.fp16:
                            scaler.unscale_(optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        
                        if self.config.fp16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                
                # ログの記録
                epoch_loss += loss.item()
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
                    if self.use_accelerate and self.accelerator.is_main_process:
                        self.accelerator.log({"train_loss": avg_loss}, step=global_step)
                
                # 評価
                if global_step % self.config.eval_steps == 0 and eval_dataloader:
                    eval_loss = self._evaluate(eval_dataloader)
                    logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self._save_best_model()
                    
                    self.model.train()
                
                # チェックポイントの保存
                if global_step % self.config.save_steps == 0:
                    if self.use_accelerate:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            save_checkpoint(
                                self.model, optimizer, scheduler,
                                epoch, global_step, self.config,
                                {"loss": epoch_loss / (step + 1)}
                            )
                    elif self.config.local_rank in [-1, 0]:
                        save_checkpoint(
                            self.model, optimizer, scheduler,
                            epoch, global_step, self.config,
                            {"loss": epoch_loss / (step + 1)}
                        )
        
        logger.info("Training completed!")
        return self.model
    
    def _evaluate(self, eval_dataloader) -> float:
        """評価の実行"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=self.config.local_rank not in [-1, 0]):
                if not self.use_accelerate:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                if self.use_accelerate:
                    loss = self.accelerator.gather(loss).mean()
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss
    
    def _save_best_model(self):
        """ベストモデルの保存"""
        best_model_path = os.path.join(self.config.output_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        
        if self.use_accelerate:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
        else:
            if hasattr(self.model, "module"):
                self.model.module.save_pretrained(best_model_path)
            else:
                self.model.save_pretrained(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
        
        logger.info(f"Best model saved to {best_model_path}")