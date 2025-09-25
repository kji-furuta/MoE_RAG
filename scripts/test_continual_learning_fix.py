#!/usr/bin/env python3
"""
継続学習システムの修正確認テストスクリプト
"""

import sys
import json
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.continual_learning_pipeline import ContinualLearningPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_continual_learning():
    """継続学習システムのテスト"""

    # テスト用の小さいモデルを使用
    base_model = "microsoft/phi-2"  # 小さめのモデルでテスト

    # テストデータの準備
    test_data = [
        {"text": "継続学習のテストデータ1: 道路設計の基本について説明します。"},
        {"text": "継続学習のテストデータ2: 設計速度と曲線半径の関係は重要です。"},
        {"text": "継続学習のテストデータ3: 交通安全施設の設置基準を確認します。"},
    ]

    # テストデータをファイルに保存
    test_data_path = project_root / "data/continual/test_continual_data.jsonl"
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_data_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Test data saved to: {test_data_path}")

    try:
        # パイプラインの初期化
        pipeline = ContinualLearningPipeline(base_model_path=base_model)

        # トークナイザーのロード
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # モデルのロード（量子化なしでテスト）
        logger.info("Loading model without quantization for testing...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("Model loaded successfully")

        # 継続学習タスクの実行
        logger.info("Starting continual learning task...")
        trained_model = pipeline.run_continual_task(
            model=model,
            tokenizer=tokenizer,
            task_name="test_task",
            train_dataset_path=str(test_data_path),
            epochs=1,  # テストなので1エポックのみ
            batch_size=1,
            learning_rate=5e-5,
            use_previous_fisher=False  # 最初のタスクなのでFalse
        )

        logger.info("✅ Continual learning task completed successfully!")

        # 出力ディレクトリの確認
        output_dirs = list(Path("outputs").glob("continual_test_task_*"))
        if output_dirs:
            logger.info(f"Model saved to: {output_dirs[-1]}")

            # adapter_config.jsonの確認
            adapter_config_path = output_dirs[-1] / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                logger.info(f"Adapter config: {json.dumps(adapter_config, indent=2)}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_continual_learning()
    sys.exit(0 if success else 1)