"""
継続学習パイプライン
フルファインチューニング済みモデルからの継続学習を管理
"""
import torch
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import logging
from tqdm import tqdm

from .ewc_utils import EWCHelper
from .training_utils import TrainingConfig, TextDataset
from .efficient_fisher_manager import EfficientFisherManager
from .dynamic_batch_size import DynamicBatchSizeManager, AdaptiveDataLoader
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ContinualLearningPipeline:
    """完全な継続学習パイプライン"""
    
    def __init__(self, base_model_path: Optional[str] = None, use_efficient_fisher: bool = True):
        self.base_model_path = base_model_path
        self.task_history = []
        self.ewc_data_path = Path("outputs/ewc_data")
        self.ewc_data_path.mkdir(parents=True, exist_ok=True)
        self.use_efficient_fisher = use_efficient_fisher
        
        # 効率的なFisher行列マネージャー
        if use_efficient_fisher:
            self.fisher_manager = EfficientFisherManager(
                storage_path=str(self.ewc_data_path / "fisher_matrices")
            )
        
        # タスク履歴の読み込み
        self.history_file = self.ewc_data_path / "task_history.json"
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.task_history = json.load(f)
                logger.info(f"Loaded {len(self.task_history)} previous tasks")
    
    def load_finetuned_model(self, model_path: str):
        """フルファインチューニング済みモデルをロード"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # outputsディレクトリからのモデルロードをサポート
        if model_path.startswith("outputs/"):
            full_path = Path(model_path)
        else:
            # 最新のフルファインチューニングモデルを自動検出
            full_path = self._find_latest_finetuned_model(model_path)
        
        logger.info(f"Loading finetuned model from: {full_path}")
        
        # training_info.jsonの確認
        training_info_path = full_path / "training_info.json"
        if training_info_path.exists():
            with open(training_info_path) as f:
                training_info = json.load(f)
                self.base_model_info = training_info
                logger.info(f"Model info: {training_info.get('model_name', 'Unknown')}")
        
        # モデルとトークナイザーのロード
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(full_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(full_path),
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _find_latest_finetuned_model(self, pattern: str) -> Path:
        """最新のフルファインチューニングモデルを検索"""
        outputs_dir = Path("outputs")
        
        # パターンに一致するディレクトリを検索
        candidates = []
        for path in outputs_dir.iterdir():
            if path.is_dir() and pattern in path.name:
                candidates.append(path)
        
        if not candidates:
            # より緩い検索
            candidates = list(outputs_dir.glob(f"*{pattern}*"))
        
        if not candidates:
            raise ValueError(f"No finetuned models found matching: {pattern}")
        
        # 最新のモデルを選択（ディレクトリの作成時刻で判断）
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest model: {latest}")
        return latest
    
    def run_continual_task(
        self,
        model,
        tokenizer,
        task_name: str,
        train_dataset_path: str,
        epochs: int = 3,
        use_previous_fisher: bool = True,
        fisher_importance: float = 5000.0,
        batch_size: int = 1,
        learning_rate: float = 2e-5
    ):
        """継続学習タスクを実行"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Continual Learning Task: {task_name}")
        logger.info(f"{'='*50}")
        
        # EWCヘルパーの準備
        ewc_helpers = []
        if use_previous_fisher and len(self.task_history) > 0:
            logger.info("Loading previous Fisher matrices...")
            ewc_helpers = self._load_previous_fisher_matrices()
            logger.info(f"Loaded {len(ewc_helpers)} previous Fisher matrices")
        
        # 出力ディレクトリの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"outputs/continual_{task_name}_{timestamp}"
        
        # トレーニング設定
        config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=16,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=True,
            gradient_checkpointing=True,
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
        )
        
        # データセットの準備
        logger.info(f"Loading dataset from: {train_dataset_path}")
        train_dataset = TextDataset(
            data_path=train_dataset_path,
            tokenizer=tokenizer,
            max_length=2048
        )
        
        # EWC対応トレーナーの作成
        from .ewc_full_finetuning import EWCFullFinetuningTrainer
        trainer = EWCFullFinetuningTrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            ewc_helpers=ewc_helpers,
            ewc_lambda=fisher_importance
        )
        
        # トレーニング実行
        logger.info("Starting training...")
        trainer.train()
        
        # Fisher行列の計算と保存
        logger.info("Computing Fisher matrix for current task...")
        self._compute_and_save_fisher(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset_path=train_dataset_path,
            task_name=task_name
        )
        
        # モデルの保存
        logger.info(f"Saving model to: {output_dir}")
        trainer.save_model()
        
        # タスク履歴の更新
        task_info = {
            "task_name": task_name,
            "timestamp": timestamp,
            "model_path": output_dir,
            "fisher_path": str(self.ewc_data_path / f"fisher_{task_name}.pt"),
            "dataset": train_dataset_path,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "ewc_lambda": fisher_importance
        }
        self.task_history.append(task_info)
        
        # 履歴の保存
        self._save_task_history()
        
        logger.info(f"Task '{task_name}' completed successfully")
        return trainer.model
    
    def _compute_and_save_fisher(self, model, tokenizer, dataset_path: str, task_name: str):
        """Fisher行列を計算して保存"""
        if self.use_efficient_fisher:
            # 効率的なFisher行列計算
            logger.info("Using efficient Fisher matrix computation...")
            
            # データローダーの準備
            dataset = TextDataset(
                data_path=dataset_path,
                tokenizer=tokenizer,
                max_length=2048
            )
            
            # 動的バッチサイズマネージャー
            batch_size_manager = DynamicBatchSizeManager(
                initial_batch_size=4,
                min_batch_size=1,
                max_batch_size=8,
                target_memory_usage=0.7
            )
            
            # アダプティブデータローダー
            dataloader = AdaptiveDataLoader(
                dataset,
                batch_size_manager,
                shuffle=True
            )
            
            # Fisher行列の計算（ブロック単位）
            fisher_path = self.fisher_manager.compute_fisher_blockwise(
                model=model,
                dataloader=dataloader,
                task_name=task_name,
                block_size=1000000,  # 1Mパラメータごと
                max_batches=100
            )
            
            logger.info(f"Efficient Fisher matrix saved to: {fisher_path}")
            
        else:
            # 従来のFisher行列計算
            device = next(model.parameters()).device
            ewc_helper = EWCHelper(model, device, use_efficient_storage=True)
            
            # データローダーの準備
            dataset = TextDataset(
                data_path=dataset_path,
                tokenizer=tokenizer,
                max_length=2048
            )
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=4,  # Fisher計算用の小さめのバッチサイズ
                shuffle=True,
                num_workers=0
            )
            
            # Fisher行列の計算
            logger.info("Computing Fisher matrix...")
            ewc_helper.compute_fisher_matrix(dataloader, max_batches=100)
            
            # 保存
            fisher_path = self.ewc_data_path / f"fisher_{task_name}.pt"
            torch.save({
                'fisher_matrix': ewc_helper.fisher_matrix,
                'params': ewc_helper.params,
                'task_name': task_name,
                'timestamp': datetime.now().isoformat()
            }, fisher_path)
            
            logger.info(f"Fisher matrix saved to: {fisher_path}")
    
    def _load_previous_fisher_matrices(self) -> List[EWCHelper]:
        """過去のFisher行列をロード"""
        ewc_helpers = []
        
        if self.use_efficient_fisher:
            # 効率的なFisher行列のロード
            task_names = [task['task_name'] for task in self.task_history]
            fisher_matrices = self.fisher_manager.load_fisher_matrices(task_names)
            
            for fisher_matrix in fisher_matrices:
                # EWCHelper互換のオブジェクトを作成
                helper = type('EWCHelper', (), {
                    'fisher_matrix': fisher_matrix,
                    'params': {},  # 効率的な実装ではparamsは別管理
                    'compute_ewc_loss': lambda self, model: self._compute_ewc_loss_efficient(model, fisher_matrix)
                })()
                
                ewc_helpers.append(helper)
        else:
            # 従来のFisher行列のロード
            for task in self.task_history:
                fisher_path = Path(task['fisher_path'])
                if fisher_path.exists():
                    logger.info(f"Loading Fisher matrix from: {fisher_path}")
                    data = torch.load(fisher_path, map_location='cpu')
                    
                    # EWCHelperの再構築
                    helper = type('EWCHelper', (), {
                        'fisher_matrix': data['fisher_matrix'],
                        'params': data['params']
                    })()
                    
                    ewc_helpers.append(helper)
                else:
                    logger.warning(f"Fisher matrix not found: {fisher_path}")
        
        return ewc_helpers
    
    def _compute_ewc_loss_efficient(self, model, fisher_matrix):
        """効率的なEWC損失計算"""
        ewc_loss = 0
        device = next(model.parameters()).device
        
        for name, param in model.named_parameters():
            if name in fisher_matrix:
                # Fisher行列を必要に応じてGPUに転送
                fisher = fisher_matrix[name].to(device)
                # 現在のパラメータとの差分を計算
                # 注: 効率的な実装では参照パラメータも別途管理が必要
                diff = param  # ここは簡略化
                ewc_loss += (fisher * diff.pow(2)).sum()
        
        return ewc_loss
    
    def _save_task_history(self):
        """タスク履歴を保存"""
        with open(self.history_file, 'w') as f:
            json.dump(self.task_history, f, indent=2)
        logger.info(f"Task history saved to: {self.history_file}")
    
    def evaluate_all_tasks(self, model, tokenizer):
        """全タスクでの性能を評価"""
        logger.info("\n=== Evaluating performance on all tasks ===")
        results = {}
        
        for task in self.task_history:
            logger.info(f"Evaluating on task: {task['task_name']}")
            
            # 評価データセットのパスを推定
            eval_path = task['dataset'].replace('.jsonl', '_eval.jsonl')
            if not Path(eval_path).exists():
                logger.warning(f"Evaluation dataset not found: {eval_path}")
                continue
            
            # 評価の実行
            perplexity = self._evaluate_perplexity(model, tokenizer, eval_path)
            results[task['task_name']] = {
                'perplexity': perplexity,
                'timestamp': datetime.now().isoformat()
            }
        
        # 結果の保存
        results_path = self.ewc_data_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        return results
    
    def _evaluate_perplexity(self, model, tokenizer, dataset_path: str) -> float:
        """パープレキシティを計算"""
        model.eval()
        
        dataset = TextDataset(
            data_path=dataset_path,
            tokenizer=tokenizer,
            max_length=2048
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item() * batch['input_ids'].size(1)
                total_tokens += batch['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
