"""
EWC対応フルファインチューニングトレーナー
継続学習のためのEWC損失を統合したトレーナー
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
import logging
from tqdm import tqdm
import os

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
        # 親クラスの初期化
        super().__init__(model, config)
        
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.ewc_helpers = ewc_helpers or []
        self.ewc_lambda = ewc_lambda
        
        # EWC使用時のログ
        if self.ewc_helpers:
            logger.info(f"EWC enabled with {len(self.ewc_helpers)} previous tasks")
            logger.info(f"EWC lambda: {self.ewc_lambda}")
        
        # データローダーの準備
        self._prepare_dataloaders()
    
    def _prepare_dataloaders(self):
        """データローダーの準備"""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
    
    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """EWC損失を含む損失計算"""
        # デバイスに転送
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 通常の損失計算
        outputs = model(**batch)
        task_loss = outputs.loss
        
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
                self.log_metrics({
                    "loss/task": task_loss.item(),
                    "loss/ewc": ewc_loss.item(),
                    "loss/total": (task_loss + ewc_loss).item()
                })
        
        return task_loss + ewc_loss
    
    def train(self):
        """EWC対応の学習実行"""
        logger.info("Starting EWC-enabled training...")
        logger.info(f"Number of training examples: {len(self.train_dataset)}")
        logger.info(f"Number of epochs: {self.config.num_train_epochs}")
        logger.info(f"Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        # モデルを学習モードに
        self.model.train()
        
        # オプティマイザとスケジューラーの準備
        self._setup_optimizer_and_scheduler()
        
        # 学習ループ
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(int(self.config.num_train_epochs)):
            logger.info(f"\nEpoch {epoch + 1}/{int(self.config.num_train_epochs)}")
            
            epoch_loss = 0
            task_loss_sum = 0
            ewc_loss_sum = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
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
                    if step % self.config.logging_steps == 0:
                        current_loss = epoch_loss / (step + 1)
                        current_lr = self.scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            'loss': f"{current_loss:.4f}",
                            'lr': f"{current_lr:.2e}"
                        })
                    
                    global_step += 1
                    
                    # メモリ管理
                    if step % 50 == 0:
                        torch.cuda.empty_cache()
            
            # エポック終了時の処理
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
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
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                total_steps += 1
        
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
        from transformers import AdamW, get_linear_schedule_with_warmup
        
        # パラメータグループの設定
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # オプティマイザ
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon
        )
        
        # スケジューラー
        num_training_steps = (
            len(self.train_dataloader) // self.config.gradient_accumulation_steps
            * self.config.num_train_epochs
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
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
