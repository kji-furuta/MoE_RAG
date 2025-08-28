import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Union
import logging
from tqdm import tqdm
import os
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import BitsAndBytesConfig
from accelerate import Accelerator

from .training_utils import (
    TrainingConfig,
    TextDataset,
    get_optimizer_and_scheduler,
    save_checkpoint,
    GradientAccumulator
)
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LoRAConfig:
    """LoRA設定クラス"""
    def __init__(
        self,
        r: int = 16,                      # LoRAの次元
        lora_alpha: int = 32,             # LoRAのアルファ値
        target_modules: Optional[List[str]] = None,  # ターゲットモジュール
        lora_dropout: float = 0.05,       # ドロップアウト率
        bias: str = "none",               # バイアスの種類
        task_type: str = "CAUSAL_LM",     # タスクタイプ
        use_qlora: bool = False,          # QLoRAを使用するか
        qlora_4bit: bool = True,          # 4bit量子化を使用するか (Falseなら8bit)
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.use_qlora = use_qlora
        self.qlora_4bit = qlora_4bit


class LoRAFinetuningTrainer:
    """LoRA/QLoRAファインチューニングのトレーナー"""
    
    def __init__(
        self,
        model: BaseModel,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        train_dataset: Optional[TextDataset] = None,
        eval_dataset: Optional[TextDataset] = None
    ):
        self.base_model = model
        self.lora_config = lora_config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Acceleratorの初期化
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision="fp16" if training_config.fp16 else "no",
            log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
        )
        
        # モデルの準備
        self._prepare_model()
        
    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """BitsAndBytes設定の取得"""
        if self.lora_config.qlora_4bit:
            return UnifiedQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            return UnifiedQuantizationConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
    
    def _prepare_model(self):
        """モデルの準備"""
        # QLoRAの場合、4bit/8bit量子化モデルをロード
        if self.lora_config.use_qlora:
            if self.base_model.model is None:
                # 量子化設定を一時的に変更
                original_8bit = self.base_model.load_in_8bit
                original_4bit = self.base_model.load_in_4bit
                
                self.base_model.load_in_8bit = not self.lora_config.qlora_4bit
                self.base_model.load_in_4bit = self.lora_config.qlora_4bit
                
                self.base_model.load_model()
                self.base_model.load_tokenizer()
                
                # 設定を元に戻す
                self.base_model.load_in_8bit = original_8bit
                self.base_model.load_in_4bit = original_4bit
        else:
            # 通常のLoRAの場合
            if self.base_model.model is None:
                self.base_model.load_model()
                self.base_model.load_tokenizer()
        
        self.model = self.base_model.model
        self.tokenizer = self.base_model.tokenizer
        
        # QLoRAの場合のモデル準備
        if self.lora_config.use_qlora:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_config.gradient_checkpointing
            )
        
        # LoRA設定の作成
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self._find_target_modules(),
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=getattr(TaskType, self.lora_config.task_type)
        )
        
        # PEFTモデルの準備
        self.model = get_peft_model(self.model, peft_config)
        
        # 学習可能なパラメータ数の表示:
        self.model.print_trainable_parameters()
        
    def _find_target_modules(self) -> List[str]:
        """ターゲットモジュールの特定"""
        if self.lora_config.target_modules:
            return self.lora_config.target_modules
        
        # 一般的なモデルタイプに対するデフォルト設定
        model_type = self.model.config.model_type.lower()
        
        if "llama" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_type:
            return ["c_attn", "c_proj", "c_fc"]
        elif "bloom" in model_type:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "opt" in model_type:
            return ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        else:
            # その他
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def train(
        self,
        train_texts: Optional[List[str]] = None,
        eval_texts: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """トレーニングの実行"""
        
        # データセットの準備
        if train_texts and self.train_dataset is None:
            self.train_dataset = TextDataset(train_texts, self.tokenizer)
        
        if eval_texts and self.eval_dataset is None:
            self.eval_dataset = TextDataset(eval_texts, self.tokenizer)
        
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")
        
        # DataLoaderの準備
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # オプティマイザーとスケジューラーの準備
        num_training_steps = len(train_dataloader) * self.training_config.num_epochs // self.training_config.gradient_accumulation_steps
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model,
            self.training_config,
            num_training_steps
        )
        
        # Acceleratorによる準備
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # トレーニングループの開始
        logger.info("Starting LoRA training...")
        self.model.train()
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.training_config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.training_config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.max_grad_norm
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # ログの記録
                if global_step % self.training_config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
                    if self.accelerator.is_main_process:
                        self.accelerator.log({
                            "train_loss": avg_loss,
                            "learning_rate": scheduler.get_last_lr()[0]
                        }, step=global_step)
                
                # 評価
                if global_step % self.training_config.eval_steps == 0 and eval_dataloader:
                    eval_loss = self._evaluate(eval_dataloader)
                    logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self._save_best_model()
                    
                    self.model.train()
                
                # チェックポイントの保存
                if global_step % self.training_config.save_steps == 0:
                    self._save_checkpoint(epoch, global_step)
                
                global_step += 1
        
        logger.info("LoRA training completed!")
        
        # 最終モデルの保存
        self._save_final_model()
        
        return self.model
    
    def _evaluate(self, eval_dataloader) -> float:
        """評価の実行"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(
                eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process
            ):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = self.accelerator.gather(loss).mean()
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, step: int):
        """チェックポイントの保存"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(
                self.training_config.output_dir,
                f"checkpoint-{step}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # LoRAモデルの保存
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            
            # トレーニング状態の保存
            torch.save({
                "epoch": epoch,
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, os.path.join(checkpoint_dir, "training_state.pt"))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_best_model(self):
        """ベストモデルの保存"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            best_model_path = os.path.join(
                self.training_config.output_dir,
                "best_lora_model"
            )
            os.makedirs(best_model_path, exist_ok=True)
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            
            logger.info(f"Best LoRA model saved to {best_model_path}")
    
    def _save_final_model(self):
        """最終モデルの保存とマージ（QLoRA以外）"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # LoRAアダプターの保存
            lora_path = os.path.join(
                self.training_config.output_dir,
                "final_lora_model"
            )
            os.makedirs(lora_path, exist_ok=True)
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(lora_path)
            self.tokenizer.save_pretrained(lora_path)
            
            # マージ済みモデルの保存
            if not self.lora_config.use_qlora:  # QLoRAの場合はマージ不要
                merged_path = os.path.join(
                    self.training_config.output_dir,
                    "merged_model"
                )
                os.makedirs(merged_path, exist_ok=True)
                
                merged_model = unwrapped_model.merge_and_unload()
                merged_model.save_pretrained(merged_path)
                self.tokenizer.save_pretrained(merged_path)
                
                logger.info(f"Merged model saved to {merged_path}")
            
            logger.info(f"Final LoRA model saved to {lora_path}")
    
    @staticmethod
    def load_lora_model(
        base_model_name: str,
        lora_adapter_path: str,
        device: Optional[torch.device] = None
    ) -> tuple:
        """既存のLoRAモデルをロード"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # ベースモデルのロード
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device is None else None
        )
        
        # LoRAアダプターのロード
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        # トークナイザーのロード
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        
        if device is not None:
            model = model.to(device)
        
        return model, tokenizer