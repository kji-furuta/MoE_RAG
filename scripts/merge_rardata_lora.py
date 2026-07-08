#!/usr/bin/env python3
"""
RARdata学習済みLoRAアダプターをベースモデルにマージ

使用方法:
    python scripts/merge_rardata_lora.py
    python scripts/merge_rardata_lora.py --output outputs/rardata_merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device_map: str = "auto"
):
    """LoRAアダプターをベースモデルにマージ"""

    logger.info("=" * 60)
    logger.info("RARdata LoRAマージツール")
    logger.info("=" * 60)

    # 1. ベースモデルのロード
    logger.info(f"[1/4] ベースモデルをロード: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    logger.info(f"✅ ベースモデルロード完了")

    # 2. LoRAアダプターのロード
    logger.info(f"[2/4] LoRAアダプターをロード: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    logger.info(f"✅ LoRAアダプターロード完了")

    # 3. マージ
    logger.info("[3/4] LoRAアダプターをマージ中...")
    merged_model = model.merge_and_unload()
    logger.info(f"✅ マージ完了")

    # 4. 保存
    logger.info(f"[4/4] マージ済みモデルを保存: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    # トークナイザーも保存
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"✅ 保存完了")

    # 5. 結果サマリー
    logger.info("=" * 60)
    logger.info("✅ マージ成功")
    logger.info("=" * 60)
    logger.info(f"出力先: {output_path}")
    logger.info("")
    logger.info("次のステップ:")
    logger.info("  1. GGUF変換 (llama.cpp)")
    logger.info(f"     python convert.py {output_path}")
    logger.info("")
    logger.info("  2. Ollamaで使用")
    logger.info(f"     ollama create rardata -f Modelfile")
    logger.info("")
    logger.info("  3. 直接推論")
    logger.info("     from transformers import AutoModelForCausalLM")
    logger.info(f"     model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RARdata LoRAマージツール")
    parser.add_argument(
        "--base-model",
        type=str,
        default="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        help="ベースモデルのパス"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/continual_task_103_20251209_002510/checkpoint-final",
        help="LoRAアダプターのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/rardata_merged",
        help="出力先パス"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="デバイスマップ (auto, cpu, cuda:0, など)"
    )

    args = parser.parse_args()

    try:
        merge_lora_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter,
            output_path=args.output,
            device_map=args.device_map
        )
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        raise


if __name__ == "__main__":
    main()
