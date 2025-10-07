#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Trainer
QLoRA + DPO統合実装 for cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """DPO学習設定"""
    # モデル設定
    model_name: str
    output_dir: str = "./dpo_output"

    # LoRA設定
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None

    # DPO設定
    beta: float = 0.1  # DPOのbetaパラメータ
    max_prompt_length: int = 512  # メモリ削減（1024→512）
    max_length: int = 1024  # メモリ削減（2048→1024）

    # 学習設定
    per_device_train_batch_size: int = 1  # VRAM制約のため
    gradient_accumulation_steps: int = 16  # メモリ削減のため増加（8→16）
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    max_steps: int = -1  # -1は無効（num_train_epochsを使用）
    lr_scheduler_type: str = "cosine"

    # メモリ最適化
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True  # Ampere以降のGPU

    # ログ・保存
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # GPU制約（25GB GPUを有効活用するため23GiBに調整）
    max_memory_per_gpu: str = "23GiB"
    max_cpu_memory: str = "40GiB"


class DPOQLoRATrainer:
    """
    QLoRA + DPO統合トレーナー

    メモリ要件:
    - Policy model (4-bit): 16GB
    - Reference model (4-bit, frozen): 16GB
    - Adapters/Activations: 2-4GB
    - 合計: 34-38GB (2x24GB環境で実行可能)
    """

    def __init__(self, config: DPOTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # GPU数の取得
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"利用可能なGPU数: {self.num_gpus}")

        # メモリ制約の設定
        self.max_memory = self._setup_memory_constraints()

    def _setup_memory_constraints(self) -> Dict[Union[int, str], str]:
        """
        メモリ制約の設定
        Codex MCP推奨: 明示的なmax_memoryでAccelerateのヒューリスティック回避

        注意: accelerateライブラリはGPUデバイスを整数インデックスで指定する必要があります
        文字列 "cuda:0" ではなく整数 0 を使用
        """
        max_memory = {}

        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                # 整数インデックスを使用（accelerate要件）
                max_memory[i] = self.config.max_memory_per_gpu

        max_memory["cpu"] = self.config.max_cpu_memory

        logger.info(f"メモリ制約: {max_memory}")
        return max_memory

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """
        4-bit量子化設定の取得
        prompt_kji.md Lines 219-224 に基づく
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Ampere以降のGPU
        )

    def _get_lora_config(self) -> LoraConfig:
        """
        LoRA設定の取得
        prompt_kji.md Lines 245-260 に基づく
        """
        # デフォルトのtarget_modules (Qwen2アーキテクチャ用)
        target_modules = self.config.target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    def load_model_and_tokenizer(self):
        """
        モデルとトークナイザのロード
        4-bit量子化 + device_map="auto"
        """
        logger.info(f"モデルのロード開始: {self.config.model_name}")

        # 量子化設定
        quantization_config = self._get_quantization_config()

        # モデルのロード
        # device_map="auto"でaccelerateが自動配置
        # max_memoryで明示的に制約
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=self.max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # キャッシュ無効化（学習時は不要）
        self.model.config.use_cache = False

        # トークナイザのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # pad_tokenの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("モデルとトークナイザのロード完了")

        # メモリ使用量のログ
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def prepare_model_for_training(self):
        """
        モデルを学習用に準備
        - QLoRA用のkbit training準備
        - LoRAアダプタの適用
        """
        logger.info("モデルを学習用に準備中...")

        # QLoRA用の準備 (gradient checkpointing含む)
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing
        )

        # LoRA設定の取得
        peft_config = self._get_lora_config()

        # PEFTモデルの作成
        self.model = get_peft_model(self.model, peft_config)

        # 学習可能なパラメータ数の表示
        self.model.print_trainable_parameters()

        logger.info("モデルの準備完了")

    def _prepare_external_model_for_training(self):
        """
        外部から提供された量子化済みモデルを学習用に準備
        prepare_model_for_kbit_training()をスキップし、LoRAアダプターのみ適用
        """
        logger.info("外部モデルを学習用に準備中（量子化済みのためLoRAアダプターのみ適用）...")

        # キャッシュ無効化（学習時は不要）
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        # Gradient checkpointingの有効化（メモリ節約）
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing有効化")

        # LoRA設定の取得
        peft_config = self._get_lora_config()

        # PEFTモデルの作成
        self.model = get_peft_model(self.model, peft_config)

        # 学習可能なパラメータ数の表示
        self.model.print_trainable_parameters()

        logger.info("外部モデルの準備完了")

    def load_preference_dataset(self, dataset_path: str, split: str = "train") -> Dataset:
        """
        Preference datasetのロード

        期待される形式:
        - prompt: str
        - chosen: str
        - rejected: str
        """
        logger.info(f"データセットのロード: {dataset_path}")

        # JSONLファイルまたはHugging Face datasetから読み込み
        if Path(dataset_path).suffix == ".jsonl":
            dataset = load_dataset("json", data_files=dataset_path, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)

        # データセットの検証
        required_columns = ["prompt", "chosen", "rejected"]
        for col in required_columns:
            if col not in dataset.column_names:
                raise ValueError(f"データセットに必須カラム'{col}'が存在しません")

        logger.info(f"データセット読み込み完了: {len(dataset)} サンプル")
        return dataset

    def setup_trainer(self, train_dataset: Dataset):
        """
        DPOTrainerのセットアップ (TRL 0.8.0+ compatible)
        prompt_kji.md Lines 267-294 に基づく
        """
        logger.info("DPOTrainerのセットアップ中...")
        logger.warning("⚠️  参照モデルのlog_probsを事前計算中 - メモリピークが発生します")
        logger.warning("⚠️  この工程で一時的にメモリ使用量が増加しますが、完了後は減少します")

        # DPOConfigの設定 (TRL 0.8.0+ではDPOConfig使用)
        from trl import DPOConfig

        training_args = DPOConfig(
            # 基本設定
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,

            # ロギング・保存
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,

            # 最適化設定
            optim=self.config.optim,
            bf16=self.config.bf16,

            # DPO固有の設定
            beta=self.config.beta,
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_length,

            # メモリ効率化設定
            precompute_ref_log_probs=True,  # 参照モデルのlog_probsを事前計算（メモリ節約）
            precompute_ref_batch_size=2,  # 事前計算時のバッチサイズ（4→2に削減でピークメモリ削減）

            # その他
            remove_unused_columns=False,
            report_to="none",  # wandb統合は後で追加可能
        )

        # DPOTrainerのインスタンス化
        # ref_model=None: trainerが自動的に参照モデルのコピーを作成
        # これにより、参照モデルは勾配計算なしで凍結される
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # 自動生成（凍結コピー）
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,  # TRL 0.8.0+ではprocessing_classを使用
        )

        logger.info("DPOTrainerのセットアップ完了")

    def train(self):
        """
        学習の実行
        prompt_kji.md Lines 297 に基づく
        """
        if self.trainer is None:
            raise ValueError("Trainerがセットアップされていません。setup_trainer()を先に実行してください。")

        logger.info("DPO学習を開始します...")

        # 学習実行
        # accelerate launchを使って実行することを前提
        self.trainer.train()

        logger.info("DPO学習が完了しました")

    def save_adapter(self, adapter_path: str):
        """
        LoRAアダプタの保存
        prompt_kji.md Lines 300 に基づく
        """
        logger.info(f"アダプタを保存中: {adapter_path}")

        # アダプタの保存
        self.trainer.save_model(adapter_path)

        # トークナイザも保存
        self.tokenizer.save_pretrained(adapter_path)

        logger.info(f"アダプタの保存完了: {adapter_path}")

    def run_full_pipeline(self, dataset_path: str, adapter_output_path: str,
                         model=None, tokenizer=None):
        """
        完全なDPOパイプラインの実行

        Args:
            dataset_path: Preference datasetのパス (JSONL or HF dataset)
            adapter_output_path: LoRAアダプタの保存先パス
            model: 事前にロード済みのモデル (Noneの場合は新規ロード)
            tokenizer: 事前にロード済みのトークナイザー (Noneの場合は新規ロード)
        """
        try:
            # 1. モデルとトークナイザのロード（未提供の場合のみ）
            if model is not None and tokenizer is not None:
                logger.info("外部から提供されたモデルとトークナイザーを使用")
                self.model = model
                self.tokenizer = tokenizer
                # 外部モデルは既に量子化済みのため、LoRAアダプターのみ適用
                self._prepare_external_model_for_training()
            else:
                logger.info("新規にモデルとトークナイザーをロード")
                self.load_model_and_tokenizer()
                # 2. モデルを学習用に準備
                self.prepare_model_for_training()

            # 3. データセットのロード
            train_dataset = self.load_preference_dataset(dataset_path)

            # 4. Trainerのセットアップ
            self.setup_trainer(train_dataset)

            # 5. 学習の実行
            self.train()

            # 6. アダプタの保存
            self.save_adapter(adapter_output_path)

            logger.info("DPOパイプラインが正常に完了しました")

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error("=" * 60)
                logger.error("GPU メモリ不足エラー (OOM) が発生しました")
                logger.error("=" * 60)
                logger.error(f"エラー詳細: {str(e)}")
                logger.error("")
                logger.error("推奨される対処法:")
                logger.error("  1. シーケンス長を削減:")
                logger.error(f"     現在: max_prompt_length={self.config.max_prompt_length}, max_length={self.config.max_length}")
                logger.error("     推奨: max_prompt_length=256, max_length=512")
                logger.error("")
                logger.error("  2. Gradient Accumulationを増加:")
                logger.error(f"     現在: gradient_accumulation_steps={self.config.gradient_accumulation_steps}")
                logger.error(f"     推奨: gradient_accumulation_steps={self.config.gradient_accumulation_steps * 2}")
                logger.error("")
                logger.error("  3. LoRAランクを削減:")
                logger.error(f"     現在: lora_r={self.config.lora_r}")
                logger.error("     推奨: lora_r=32")
                logger.error("")
                logger.error("  4. メモリ状況を確認:")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1e9
                        total = torch.cuda.get_device_properties(i).total_memory / 1e9
                        logger.error(f"     GPU {i}: {allocated:.2f}GB / {total:.2f}GB 使用中")
                logger.error("=" * 60)
            raise
        except Exception as e:
            logger.error(f"DPOパイプライン実行中にエラーが発生: {str(e)}", exc_info=True)
            raise


def main():
    """
    使用例
    実行方法: accelerate launch src/training/dpo_trainer.py
    """
    # 設定の作成
    config = DPOTrainingConfig(
        model_name="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        output_dir="./outputs/dpo_output",
        lora_r=64,
        lora_alpha=128,
        beta=0.1,
        num_train_epochs=1,
    )

    # トレーナーの作成
    trainer = DPOQLoRATrainer(config)

    # パイプラインの実行
    trainer.run_full_pipeline(
        dataset_path="data/dpo/preference_dataset.jsonl",
        adapter_output_path="./outputs/dpo_adapter"
    )


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
