#!/usr/bin/env python3
"""
RARdata.jsonを使用した学習スクリプト

使用方法:
    python scripts/train_rardata.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""

    logger.info("=" * 80)
    logger.info("RARdata.json 学習実験開始")
    logger.info("=" * 80)

    # 設定
    task_name = "rardata_training"
    data_path = "/workspace/RARdata.json"  # プロジェクトルート直下
    model_path = "outputs/lora_20251129_072850"  # 既存のLoRAモデル

    # 学習パラメータ
    ewc_lambda = 5000  # EWC正則化の強度
    epochs = 3  # エポック数
    batch_size = 1  # バッチサイズ（32Bモデルのため1）
    learning_rate = 2e-5  # 学習率
    gradient_accumulation_steps = 16  # 勾配累積ステップ

    logger.info(f"タスク名: {task_name}")
    logger.info(f"データパス: {data_path}")
    logger.info(f"モデルパス: {model_path}")
    logger.info(f"EWC Lambda: {ewc_lambda}")
    logger.info(f"エポック数: {epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"学習率: {learning_rate}")
    logger.info(f"勾配累積ステップ: {gradient_accumulation_steps}")

    # GPU情報表示
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("GPU not available, training will be slow!")

    try:
        # 継続学習パイプライン初期化
        logger.info("継続学習パイプラインを初期化中...")
        pipeline = ContinualLearningPipeline(
            base_model_path=None,
            use_efficient_fisher=True  # Fisher行列計算を有効化（修正済み）
        )

        # モデルロード
        logger.info(f"モデルをロード中: {model_path}")
        model, tokenizer = pipeline.load_finetuned_model(model_path)

        logger.info("モデルのロードが完了しました")
        logger.info(f"Model type: {type(model).__name__}")

        # 継続学習タスクを開始
        logger.info("継続学習タスクを開始...")
        trained_model = pipeline.run_continual_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            train_dataset_path=data_path,
            epochs=epochs,
            use_previous_fisher=True,  # 以前のFisher行列を使用
            fisher_importance=ewc_lambda,
            batch_size=batch_size,
            learning_rate=learning_rate
            # gradient_accumulation_stepsは内部で自動設定されます (16ステップ)
        )

        # 完了メッセージ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/continual_{task_name}_{timestamp}/checkpoint-final"

        logger.info("=" * 80)
        logger.info("✅ 学習完了！")
        logger.info("=" * 80)
        logger.info(f"保存先: {output_dir}")
        logger.info(f"タスク名: {task_name}")
        logger.info(f"データ件数: 推定100-1000件")
        logger.info(f"学習エポック数: {epochs}")
        logger.info("")
        logger.info("次のステップ:")
        logger.info("1. モデル性能評価")
        logger.info("2. RAGシステムとの統合")
        logger.info("3. 引用精度の測定")

    except Exception as e:
        logger.error(f"学習実験中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
