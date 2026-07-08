#!/usr/bin/env python3
"""
RAR Phase 1パイロット学習実験

このスクリプトは100件のRAR形式データでEWC継続学習を実行します。
"""
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
workspace_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_path))

import logging
import torch
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/rar_pilot_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """RAR Phase 1学習実験のメインエントリポイント"""
    logger.info("="*80)
    logger.info("RAR Phase 1 パイロット学習実験開始")
    logger.info("="*80)

    # パラメータ設定
    task_name = "rar_pilot_phase1"
    data_path = "/workspace/data/rar_training/pilot/rar_pilot_100.json"
    model_path = "outputs/lora_20251129_072850"  # 最新のLoRAモデル（相対パス）
    ewc_lambda = 5000  # EWC重要度パラメータ
    epochs = 3
    batch_size = 1  # メモリ効率のため小さく設定
    learning_rate = 2e-5

    logger.info(f"タスク名: {task_name}")
    logger.info(f"データパス: {data_path}")
    logger.info(f"モデルパス: {model_path}")
    logger.info(f"EWC Lambda: {ewc_lambda}")
    logger.info(f"エポック数: {epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"学習率: {learning_rate}")

    # データファイルの存在確認
    if not Path(data_path).exists():
        logger.error(f"データファイルが見つかりません: {data_path}")
        return 1

    # モデルディレクトリの存在確認
    if not Path(model_path).exists():
        logger.error(f"モデルディレクトリが見つかりません: {model_path}")
        return 1

    # GPU使用可能確認
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be slow on CPU.")

    try:
        # ContinualLearningPipelineのインポート
        from src.training.continual_learning_pipeline import ContinualLearningPipeline

        logger.info("継続学習パイプラインを初期化中...")
        pipeline = ContinualLearningPipeline(
            base_model_path=None,
            use_efficient_fisher=True
        )

        # モデルのロード
        logger.info(f"モデルをロード中: {model_path}")
        model, tokenizer = pipeline.load_finetuned_model(model_path)

        logger.info("モデルのロードが完了しました")
        logger.info(f"Model type: {type(model).__name__}")

        # 継続学習タスクの実行
        logger.info("継続学習タスクを開始...")
        trained_model = pipeline.run_continual_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            train_dataset_path=data_path,
            epochs=epochs,
            use_previous_fisher=True,  # 既存のFisher行列を使用
            fisher_importance=ewc_lambda,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # 学習完了
        logger.info("="*80)
        logger.info("RAR Phase 1 学習実験が成功しました")
        logger.info("="*80)

        # 保存されたモデルパスを表示
        latest_model_path = pipeline.get_latest_model_path()
        if latest_model_path:
            logger.info(f"学習済みモデル: {latest_model_path}")

        # タスク履歴を表示
        logger.info(f"タスク履歴数: {len(pipeline.task_history)}")

        return 0

    except Exception as e:
        logger.error(f"学習実験中にエラーが発生しました: {e}", exc_info=True)
        return 1

    finally:
        # メモリのクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU メモリをクリアしました")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
