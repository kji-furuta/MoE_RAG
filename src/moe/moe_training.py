"""
MoE Training Script for Civil Engineering Domain
土木・建設分野特化MoEモデルのトレーニング
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import asdict
import os

# Import from refactored modules
from .base_config import MoETrainingConfig
from .constants import DEFAULT_VOCAB_SIZE, SAFE_VOCAB_LIMIT
from .utils import validate_input_ids, load_config, save_config, get_device
from .exceptions import MoETrainingError, MoEDataError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def safe_collate_fn(batch):
    """安全なバッチ処理（input_idsを語彙サイズ内に制限）"""
    try:
        # バッチデータを結合
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
        
        # input_idsを語彙サイズ内に制限
        input_ids = validate_input_ids(input_ids, vocab_size=DEFAULT_VOCAB_SIZE)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    except Exception as e:
        raise MoEDataError(f"Error in collate_fn: {e}")

class CivilEngineeringDataset(Dataset):
    """土木・建設分野データセット"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        expert_type_mapping: Optional[Dict] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expert_type_mapping = expert_type_mapping or {}
        
        # データの読み込み
        self.data = self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """データ読み込み"""
        data = []
        
        # 各専門分野のデータを読み込み
        expert_domains = [
            "structural_design",
            "road_design",
            "geotechnical",
            "hydraulics",
            "materials",
            "construction_management",
            "regulations",
            "environmental"
        ]
        
        for domain in expert_domains:
            domain_file = Path(data_path) / f"{domain}.jsonl"
            if domain_file.exists():
                with open(domain_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        item['expert_domain'] = domain
                        data.append(item)
            else:
                logger.warning(f"Domain file not found: {domain_file}")
        
        # データがない場合はダミーデータを生成
        if not data:
            logger.warning("No data files found, generating dummy data")
            data = self._generate_dummy_data(expert_domains)
        
        return data
    
    def _generate_dummy_data(self, domains: List[str]) -> List[Dict]:
        """ダミーデータ生成"""
        dummy_data = []
        for domain in domains:
            for i in range(10):  # 各ドメイン10サンプル
                dummy_data.append({
                    "question": f"{domain}に関する質問{i+1}",
                    "answer": f"{domain}に関する回答{i+1}",
                    "expert_domain": domain
                })
        return dummy_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # プロンプトの構築
        prompt = self._create_prompt(item)
        
        # トークナイズ
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 入力値を語彙サイズ内に制限
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 32000
        if 'input_ids' in encoded:
            encoded['input_ids'] = torch.clamp(encoded['input_ids'], 0, vocab_size - 1)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'expert_domain': item.get('expert_domain', 'general'),
            'labels': encoded['input_ids'].squeeze()
        }
    
    def _create_prompt(self, item: Dict) -> str:
        """プロンプト生成"""
        domain = item.get('expert_domain', 'general')
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # ドメイン特化プロンプト
        domain_prompts = {
            'structural_design': "構造設計の専門家として、",
            'road_design': "道路設計の専門家として、",
            'geotechnical': "地盤工学の専門家として、",
            'hydraulics': "水理・排水の専門家として、",
            'materials': "材料工学の専門家として、",
            'construction_management': "施工管理の専門家として、",
            'regulations': "建設法規の専門家として、",
            'environmental': "環境・維持管理の専門家として、"
        }
        
        prefix = domain_prompts.get(domain, "土木・建設の専門家として、")
        
        return f"{prefix}以下の質問に答えてください。\n\n質問: {question}\n\n回答: {answer}"


class MoETrainer:
    """MoEモデルトレーナー"""
    
    def __init__(
        self,
        model: nn.Module,
        config: MoETrainingConfig,
        tokenizer
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 最適化設定
        self._setup_optimizer()
        
        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # WandB初期化（オプション）
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project=config.project_name, config=asdict(config))
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
                self.config.use_wandb = False
    
    def _setup_optimizer(self):
        """オプティマイザーの設定"""
        # パラメータグループの設定
        router_params = []
        expert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            elif 'expert' in name:
                expert_params.append(param)
            else:
                other_params.append(param)
        
        # 異なる学習率の設定
        param_groups = [
            {'params': router_params, 'lr': self.config.learning_rate * self.config.router_lr_multiplier},
            {'params': expert_params, 'lr': self.config.learning_rate},
            {'params': other_params, 'lr': self.config.learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """トレーニング実行"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windowsとの互換性のため0に設定
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        # スケジューラー設定
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # 出力ディレクトリの作成
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # トレーニングループ
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_aux_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # データをデバイスに移動
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Mixed Precision Training
                with torch.cuda.amp.autocast() if self.config.mixed_precision and self.device.type == "cuda" else torch.no_grad():
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # 損失計算
                    hidden_states = outputs['hidden_states']
                    aux_loss = outputs['aux_loss']
                    
                    # 言語モデリング損失
                    shift_logits = hidden_states[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss()
                    lm_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # 総損失
                    total_loss = lm_loss + self.config.aux_loss_weight * aux_loss
                
                # Backward pass
                if self.config.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.config.gradient_accumulation_steps
                
                if self.scaler and self.device.type == "cuda":
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler and self.device.type == "cuda":
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # ロギング
                epoch_loss += lm_loss.item()
                epoch_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    avg_aux_loss = epoch_aux_loss / (batch_idx + 1)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'aux_loss': f'{avg_aux_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if self.config.use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/aux_loss': avg_aux_loss,
                                'train/learning_rate': scheduler.get_last_lr()[0],
                                'train/global_step': global_step
                            })
                        except:
                            pass
                
                # チェックポイント保存
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step, epoch_loss / (batch_idx + 1))
                
                # 評価
                if val_dataset and global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate(val_dataset)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self._save_checkpoint(global_step, val_loss, is_best=True)
                    self.model.train()
    
    def evaluate(self, dataset: Dataset) -> float:
        """評価実行"""
        self.model.eval()
        eval_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                hidden_states = outputs['hidden_states']
                shift_logits = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                total_samples += shift_labels.numel()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log({'val/loss': avg_loss})
            except:
                pass
        
        return avg_loss
    
    def _save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """チェックポイント保存"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = f"checkpoint-{step}.pt" if not is_best else "best_model.pt"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': asdict(self.config)
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    """メイン実行関数"""
    # 設定
    config = MoETrainingConfig()
    
    # トークナイザーのロード（ダミー実装）
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    except:
        logger.warning("Failed to load tokenizer, using dummy tokenizer")
        # ダミートークナイザー
        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.randint(0, 1000, (1, 100)),
                    'attention_mask': torch.ones(1, 100)
                }
        tokenizer = DummyTokenizer()
    
    # モデルの作成
    from moe_architecture import create_civil_engineering_moe
    model = create_civil_engineering_moe(
        base_model_name=config.base_model_name,
        num_experts=config.num_experts
    )
    
    # データセットの準備
    train_dataset = CivilEngineeringDataset(
        data_path=f"{config.dataset_path}/train",
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    val_dataset = CivilEngineeringDataset(
        data_path=f"{config.dataset_path}/val",
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    # トレーナーの初期化
    trainer = MoETrainer(model, config, tokenizer)
    
    # トレーニング実行
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
