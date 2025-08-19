"""
修正版EWCTrainer
compute_loss引数エラーを解決
"""

from transformers import Trainer
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FixedEWCTrainer(Trainer):
    """
    修正版EWCTrainer
    TransformersのAPIアップデートに対応
    """
    
    def __init__(self, *args, ewc_lambda=5000, fisher_matrix=None, **kwargs):
        # 'tokenizer'警告を回避
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = fisher_matrix
        self.original_params = None
        
        # 元のパラメータを保存
        if self.model is not None:
            self.original_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            logger.info(f"Saved {len(self.original_params)} original parameters for EWC")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        compute_lossメソッドの修正版
        num_items_in_batch引数を適切に処理
        """
        # 通常の損失計算
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # EWC正則化項の追加
        if self.fisher_matrix and self.original_params:
            ewc_loss = 0
            for name, param in model.named_parameters():
                if name in self.fisher_matrix and name in self.original_params:
                    fisher = self.fisher_matrix[name]
                    original = self.original_params[name]
                    ewc_loss += (fisher * (param - original) ** 2).sum()
            
            loss = loss + self.ewc_lambda * ewc_loss
            
            # ログ出力（デバッグ用）
            if self.state.global_step % 100 == 0:
                logger.debug(f"EWC loss contribution: {self.ewc_lambda * ewc_loss:.4f}")
        
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        """
        モデル保存時のディスク容量エラーを回避
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # ディスク容量チェック
        import psutil
        import os
        
        # 必要な空き容量（GB）
        required_space_gb = 70  # 32Bモデル用
        
        # 現在の空き容量を確認
        stat = os.statvfs(output_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if free_space_gb < required_space_gb:
            logger.warning(f"Low disk space: {free_space_gb:.1f} GB free, {required_space_gb} GB required")
            
            # シャーディングを使用して保存
            logger.info("Using sharded save to reduce memory footprint")
            self.model.save_pretrained(
                output_dir,
                max_shard_size="2GB",  # 小さなシャードサイズ
                safe_serialization=True
            )
        else:
            # 通常の保存
            super().save_model(output_dir, _internal_call)


def patch_main_unified():
    """
    main_unified.pyのEWCTrainerを修正版に置き換える関数
    """
    import_code = """
# Import fixed EWC trainer
try:
    from app.fixed_ewc_trainer import FixedEWCTrainer as EWCTrainer
    logger.info("Using fixed EWC trainer")
except ImportError:
    logger.warning("Fixed EWC trainer not found, using original")
    # Original EWCTrainer definition here
"""
    
    print("To fix the EWCTrainer error in main_unified.py:")
    print("1. Add this import at the top of the file:")
    print(import_code)
    print("\n2. Or replace the existing EWCTrainer class with FixedEWCTrainer")
    
    return import_code
