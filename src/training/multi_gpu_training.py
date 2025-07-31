import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Optional, Dict, Any, List
import logging
import os
from accelerate import Accelerator, DistributedDataParallelKwargs

from .training_utils import TrainingConfig, TextDataset
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MultiGPUTrainingConfig(TrainingConfig):
    """マルチGPU用の拡張トレーニング設定"""
    
    def __init__(
        self,
        strategy: str = "ddp",  # "ddp", "model_parallel", "pipeline"
        max_memory_per_gpu: Optional[Dict[int, str]] = None,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.strategy = strategy
        self.max_memory_per_gpu = max_memory_per_gpu or {0: "22GB", 1: "22GB"}
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size


class AdvancedMultiGPUTrainer:
    """高度なマルチGPUトレーニング"""
    
    def __init__(
        self,
        model: BaseModel,
        config: MultiGPUTrainingConfig,
        train_dataset: Optional[TextDataset] = None,
        eval_dataset: Optional[TextDataset] = None
    ):
        self.base_model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Acceleratorの設定
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16" if config.fp16 else "no",
            log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
            kwargs_handlers=[ddp_kwargs]
        )
        
        self._setup_strategy()
    
    def _setup_strategy(self):
        """トレーニング戦略のセットアップ"""
        if self.config.strategy == "ddp":
            self._setup_ddp()
        elif self.config.strategy == "model_parallel":
            self._setup_model_parallel()
        elif self.config.strategy == "pipeline":
            self._setup_pipeline_parallel()
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _setup_ddp(self):
        """DistributedDataParallel のセットアップ"""
        logger.info("Setting up DistributedDataParallel (DDP)")
        
        # モデルのロード
        if self.base_model.model is None:
            self.base_model.load_model()
            self.base_model.load_tokenizer()
        
        self.model = self.base_model.model
        self.tokenizer = self.base_model.tokenizer
        
        # DDP用の設定
        self.model.config.use_cache = False
        
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
    
    def _setup_model_parallel(self):
        """Model Parallelism のセットアップ"""
        logger.info("Setting up Model Parallelism")
        
        # カスタムデバイスマップを作成
        device_map = self._create_device_map()
        
        # モデルをロード（デバイスマップ付き）
        if self.base_model.model is None:
            # 一時的にdevice_mapを設定
            original_device = self.base_model.device
            self.base_model.device = None  # device_mapを使用するためNoneに設定
            
            self.base_model.load_model()
            self.base_model.load_tokenizer()
            
            # モデルを手動で配置
            self.model = self.base_model.model
            self._apply_device_map(device_map)
            
            self.base_model.device = original_device
        else:
            self.model = self.base_model.model
            self._apply_device_map(device_map)
        
        self.tokenizer = self.base_model.tokenizer
    
    def _create_device_map(self) -> Dict[str, int]:
        """効率的なデバイスマップを作成"""
        device_count = torch.cuda.device_count()
        if device_count < 2:
            return "auto"
        
        # レイヤー数を取得
        num_layers = len([n for n, _ in self.base_model.model.named_modules() 
                         if 'layer' in n or 'block' in n])
        
        layers_per_gpu = num_layers // device_count
        
        device_map = {}
        current_device = 0
        layer_count = 0
        
        for name, module in self.base_model.model.named_modules():
            if 'layer' in name or 'block' in name:
                device_map[name] = current_device
                layer_count += 1
                
                if layer_count >= layers_per_gpu and current_device < device_count - 1:
                    current_device += 1
                    layer_count = 0
            else:
                # エンベディングや出力層は最初のGPUに配置
                device_map[name] = 0
        
        return device_map
    
    def _apply_device_map(self, device_map: Dict[str, int]):
        """デバイスマップを適用"""
        for name, device_id in device_map.items():
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                module.to(f"cuda:{device_id}")
    
    def _setup_pipeline_parallel(self):
        """Pipeline Parallelism のセットアップ"""
        logger.info("Setting up Pipeline Parallelism")
        # Pipeline parallelismの実装（高度な機能）
        raise NotImplementedError("Pipeline parallelism not yet implemented")
    
    def train(
        self,
        train_texts: Optional[List[str]] = None,
        eval_texts: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """マルチGPUトレーニングを実行"""
        
        # データセットの準備
        if train_texts and self.train_dataset is None:
            self.train_dataset = TextDataset(train_texts, self.tokenizer)
        
        if eval_texts and self.eval_dataset is None:
            self.eval_dataset = TextDataset(eval_texts, self.tokenizer)
        
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")
        
        # DataLoaderの準備（DDPの場合）
        if self.config.strategy == "ddp":
            return self._train_ddp()
        elif self.config.strategy == "model_parallel":
            return self._train_model_parallel()
        else:
            raise NotImplementedError(f"Training not implemented for {self.config.strategy}")
    
    def _train_ddp(self):
        """DDP用のトレーニングループ"""
        from torch.utils.data import DataLoader
        
        # DataLoaderの準備
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # オプティマイザとスケジューラ
        from .training_utils import get_optimizer_and_scheduler
        num_training_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model, self.config, num_training_steps
        )
        
        # Acceleratorで準備
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # トレーニングループ
        logger.info(f"Starting multi-GPU training with {self.accelerator.num_processes} processes")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            if self.accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # ロギング
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    if self.accelerator.is_main_process:
                        logger.info(f"Step {global_step}: loss = {avg_loss:.4f}")
                        self.accelerator.log({"train_loss": avg_loss}, step=global_step)
                
                # 評価
                if global_step % self.config.eval_steps == 0 and eval_dataloader:
                    self._evaluate_ddp(eval_dataloader, global_step)
                    self.model.train()
                
                # チェックポイント保存
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint_ddp(epoch, global_step, optimizer, scheduler)
        
        logger.info("Multi-GPU training completed!")
        return self.model
    
    def _train_model_parallel(self):
        """Model Parallel用のトレーニングループ"""
        # Model Parallelの場合は通常のトレーニングループを使用
        # ただし、データは適切なデバイスに配置する必要がある
        logger.info("Training with model parallelism")
        
        # 簡略化されたトレーニングループ
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                # 入力を最初のGPUに配置
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        return self.model
    
    def _evaluate_ddp(self, eval_dataloader, step):
        """DDP用の評価"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = self.accelerator.gather(loss).mean()
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        if self.accelerator.is_main_process:
            logger.info(f"Step {step}: eval_loss = {avg_loss:.4f}")
            self.accelerator.log({"eval_loss": avg_loss}, step=step)
    
    def _save_checkpoint_ddp(self, epoch, step, optimizer, scheduler):
        """DDP用のチェックポイント保存"""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # モデルを保存
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # トレーニング状態を保存
            torch.save({
                "epoch": epoch,
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, os.path.join(checkpoint_dir, "training_state.pt"))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")


def setup_distributed_training(rank: int, world_size: int):
    """分散学習のセットアップ"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # NCCL バックエンドを使用（GPU間通信に最適）
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """分散学習のクリーンアップ"""
    dist.destroy_process_group()


class LargeModelLoader:
    """大規模モデル用のローダー"""
    
    @staticmethod
    def load_large_model(
        model_name: str,
        max_memory: Optional[Dict[int, str]] = None,
        offload_folder: Optional[str] = None
    ):
        """大規模モデルをマルチGPUに効率的にロード"""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig, AutoModelForCausalLM
        
        if max_memory is None:
            max_memory = {0: "22GB", 1: "22GB"}
        
        config = AutoConfig.from_pretrained(model_name)
        
        # 空のモデルを初期化
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        
        # チェックポイントをロードして配置
        model = load_checkpoint_and_dispatch(
            model,
            model_name,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            dtype=torch.float16
        )
        
        return model