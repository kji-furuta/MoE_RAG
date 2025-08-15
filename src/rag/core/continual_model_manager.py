"""
継続学習モデルの管理と選択を行うマネージャー
RAGシステムと継続学習モデルの橋渡しを行う
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ContinualTaskInfo:
    """継続学習タスクの情報"""
    task_name: str
    model_path: str
    fisher_path: Optional[str]
    timestamp: str
    dataset: str
    epochs: int
    learning_rate: float
    ewc_lambda: float
    perplexity_scores: Optional[Dict[str, float]] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class ContinualModelManager:
    """継続学習モデルの管理クラス"""
    
    def __init__(self, base_path: Path = Path("outputs")):
        """
        Args:
            base_path: 継続学習モデルが保存されているベースパス
        """
        self.base_path = Path(base_path)
        self.ewc_data_path = self.base_path / "ewc_data"
        self.task_history_file = self.ewc_data_path / "task_history.json"
        self.task_history: List[ContinualTaskInfo] = []
        self.current_task: Optional[str] = None
        self.model_cache: Dict[str, Tuple[Any, Any]] = {}  # {task_name: (model, tokenizer)}
        
        # タスク履歴の読み込み
        self._load_task_history()
    
    def _load_task_history(self):
        """タスク履歴をファイルから読み込む"""
        if self.task_history_file.exists():
            try:
                with open(self.task_history_file, 'r') as f:
                    history_data = json.load(f)
                    self.task_history = [
                        ContinualTaskInfo.from_dict(task) 
                        for task in history_data
                    ]
                logger.info(f"Loaded {len(self.task_history)} continual learning tasks")
            except Exception as e:
                logger.error(f"Failed to load task history: {e}")
                self.task_history = []
        else:
            logger.info("No task history found, starting fresh")
            self.task_history = []
    
    def get_available_tasks(self) -> List[str]:
        """利用可能な継続学習タスクのリストを返す"""
        return [task.task_name for task in self.task_history]
    
    def get_task_info(self, task_name: str) -> Optional[ContinualTaskInfo]:
        """指定したタスクの情報を取得"""
        for task in self.task_history:
            if task.task_name == task_name:
                return task
        return None
    
    def get_latest_task(self) -> Optional[ContinualTaskInfo]:
        """最新の継続学習タスクを取得"""
        if self.task_history:
            return self.task_history[-1]
        return None
    
    def select_task_for_query(self, query: str, context: Optional[str] = None) -> Optional[str]:
        """
        クエリの内容に基づいて最適な継続学習タスクを選択
        
        Args:
            query: ユーザーのクエリ
            context: 追加のコンテキスト情報
            
        Returns:
            選択されたタスク名、または None
        """
        # シンプルなキーワードマッチングによる選択
        # より高度な実装では、埋め込みベクトルによる類似度計算を行う
        
        query_lower = query.lower()
        
        # タスク名とデータセットパスからキーワードを抽出
        best_task = None
        best_score = 0
        
        for task in self.task_history:
            score = 0
            
            # タスク名のマッチング
            task_keywords = task.task_name.lower().split('_')
            for keyword in task_keywords:
                if keyword in query_lower:
                    score += 2  # タスク名の一致は高スコア
            
            # データセット名のマッチング
            dataset_keywords = Path(task.dataset).stem.lower().split('_')
            for keyword in dataset_keywords:
                if keyword in query_lower:
                    score += 1
            
            # パフォーマンススコアを考慮（perplexityが低いほど良い）
            if task.perplexity_scores:
                avg_perplexity = sum(task.perplexity_scores.values()) / len(task.perplexity_scores)
                if avg_perplexity < 10:  # 良好なパフォーマンス
                    score += 1
            
            if score > best_score:
                best_score = score
                best_task = task.task_name
        
        # スコアが閾値以上の場合のみタスクを選択
        if best_score >= 2:
            logger.info(f"Selected continual task '{best_task}' for query (score: {best_score})")
            return best_task
        
        return None
    
    def load_model_for_task(
        self, 
        task_name: str,
        device: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        指定したタスクのモデルとトークナイザーを読み込む
        
        Args:
            task_name: タスク名
            device: デバイス指定
            use_cache: キャッシュを使用するか
            
        Returns:
            (model, tokenizer) のタプル、失敗時は (None, None)
        """
        # キャッシュチェック
        if use_cache and task_name in self.model_cache:
            logger.info(f"Using cached model for task: {task_name}")
            return self.model_cache[task_name]
        
        # タスク情報の取得
        task_info = self.get_task_info(task_name)
        if not task_info:
            logger.error(f"Task not found: {task_name}")
            return None, None
        
        # モデルパスの確認
        model_path = Path(task_info.model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return None, None
        
        try:
            # デバイスの決定
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading continual model from: {model_path}")
            
            # トークナイザーの読み込み
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # モデルの読み込み（メモリ最適化付き）
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            if device == "cuda" and torch.cuda.device_count() > 1:
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = device
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded continual model for task: {task_name}")
            
            # キャッシュに保存
            if use_cache:
                self.model_cache[task_name] = (model, tokenizer)
                # キャッシュサイズ制限（最大3モデル）
                if len(self.model_cache) > 3:
                    oldest_task = list(self.model_cache.keys())[0]
                    del self.model_cache[oldest_task]
                    logger.info(f"Removed {oldest_task} from cache (size limit)")
            
            self.current_task = task_name
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model for task {task_name}: {e}")
            return None, None
    
    def get_fisher_matrix_path(self, task_name: str) -> Optional[Path]:
        """指定したタスクのFisher行列パスを取得"""
        task_info = self.get_task_info(task_name)
        if task_info and task_info.fisher_path:
            fisher_path = Path(task_info.fisher_path)
            if fisher_path.exists():
                return fisher_path
        return None
    
    def should_use_continual_model(
        self, 
        query: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[bool, Optional[str]]:
        """
        クエリに対して継続学習モデルを使用すべきか判断
        
        Args:
            query: ユーザーのクエリ
            confidence_threshold: 使用判断の信頼度閾値
            
        Returns:
            (使用すべきか, 選択されたタスク名)
        """
        # タスク履歴が空の場合は使用しない
        if not self.task_history:
            return False, None
        
        # クエリに基づくタスク選択
        selected_task = self.select_task_for_query(query)
        
        if selected_task:
            # 選択されたタスクの情報を確認
            task_info = self.get_task_info(selected_task)
            if task_info and Path(task_info.model_path).exists():
                return True, selected_task
        
        return False, None
    
    def clear_cache(self):
        """モデルキャッシュをクリア"""
        self.model_cache.clear()
        torch.cuda.empty_cache()
        logger.info("Cleared continual model cache")
    
    def get_status(self) -> Dict[str, Any]:
        """管理状態のステータスを返す"""
        return {
            "total_tasks": len(self.task_history),
            "available_tasks": self.get_available_tasks(),
            "current_task": self.current_task,
            "cached_models": list(self.model_cache.keys()),
            "cache_size": len(self.model_cache),
            "latest_task": self.get_latest_task().task_name if self.get_latest_task() else None
        }