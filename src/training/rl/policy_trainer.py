"""
Policy Trainer - GRPO (Group Relative Policy Optimization) Wrapper

trl.GRPOTrainerをベースに、RLAnythingの閉ループ内で
ポリシー最適化を担当するコンポーネント。

GRPO Advantage計算:
    A_i = (r_i - mean(r_1..G)) / std(r_1..G)

ポリシー勾配:
    L_GRPO(θ) = -E[min(ratio * A, clip(ratio, 1±ε) * A) - β * KL(π_θ || π_ref)]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import RLAnythingConfig
from .trajectory_buffer import Trajectory

logger = logging.getLogger(__name__)


class PolicyTrainer:
    """GRPOベースのポリシートレーナー

    trl.GRPOTrainerをラップし、以下の機能を提供:
    1. モデル/トークナイザのロード（QLoRA対応）
    2. グループサンプリング（G個の応答生成）
    3. 外部報酬関数によるGRPO最適化
    4. アダプタ保存/ロード
    """

    def __init__(self, config: RLAnythingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.trainer = None
        self._is_loaded = False

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"PolicyTrainer初期化: model={config.model_name}, GPUs={self.num_gpus}")

    # ── モデルロード ──────────────────────────────────

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """量子化設定を取得"""
        if not self.config.use_quantization:
            return None

        if self.config.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.config.quantization_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _get_lora_config(self) -> LoraConfig:
        """LoRA設定を取得"""
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.target_modules,
        )

    def _setup_memory_constraints(self) -> Dict[Union[int, str], str]:
        """メモリ制約を設定（DPOQLoRATrainerパターン準拠）"""
        max_memory: Dict[Union[int, str], str] = {}
        for i in range(self.num_gpus):
            max_memory[i] = self.config.max_memory_per_gpu
        max_memory["cpu"] = self.config.max_cpu_memory
        return max_memory

    def load_model_and_tokenizer(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """モデルとトークナイザをロード

        Args:
            model: 外部から提供する事前ロード済みモデル
            tokenizer: 外部から提供する事前ロード済みトークナイザ
            status_callback: ステータスメッセージを送信するコールバック
        """
        def _report(msg: str) -> None:
            logger.info(msg)
            if status_callback:
                status_callback(msg)

        if model is not None and tokenizer is not None:
            _report("外部提供のモデル/トークナイザを使用")
            self.model = model
            self.tokenizer = tokenizer
        else:
            _report(f"モデルのロード開始: {self.config.model_name}")
            quant_config = self._get_quantization_config()

            # 量子化使用時は max_memory を設定しない
            # (accelerate が量子化前のサイズで device_map を計算し、
            #  meta device に配置してしまう問題を回避)
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if quant_config is not None:
                _report(f"量子化ロード: {self.config.quantization_bits}-bit")
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
                # 量子化時は compute_dtype で精度を制御するため torch_dtype は不要
            else:
                _report("フル精度ロード")
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if self.config.bf16 else torch.float16
                )
                if self.num_gpus > 0:
                    max_memory = self._setup_memory_constraints()
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = max_memory

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
            self.model.config.use_cache = False

            _report("トークナイザをロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # LoRA適用
        if self.config.use_lora:
            _report("LoRAアダプタを適用中...")
            if self.config.use_quantization:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.config.gradient_checkpointing,
                )
            peft_config = self._get_lora_config()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self._is_loaded = True
        self._log_gpu_memory("モデルロード後")
        _report("モデルとトークナイザのロード完了")

    # ── グループサンプリング ─────────────────────────────

    def generate_group(
        self,
        prompts: List[str],
        num_generations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Trajectory]:
        """プロンプト群に対してG個の応答を生成

        GRPOの核心: 各プロンプトに対してG個のサンプルを生成し、
        グループ相対的なadvantage計算の基盤を提供する。

        Args:
            prompts: 入力プロンプトのリスト
            num_generations: 各プロンプトあたりの生成数 G
            temperature: サンプリング温度
            max_new_tokens: 最大生成トークン数

        Returns:
            各プロンプトに対するTrajectoryのリスト
        """
        if not self._is_loaded:
            raise RuntimeError("モデルが未ロードです。load_model_and_tokenizer()を先に実行してください。")

        G = num_generations or self.config.grpo_num_generations
        temp = temperature or self.config.grpo_temperature
        max_tokens = max_new_tokens or self.config.grpo_max_new_tokens

        trajectories: List[Trajectory] = []
        self.model.eval()

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                )
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                completions: List[str] = []
                for _ in range(G):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    # 入力部分を除去して応答部分のみデコード
                    new_tokens = output_ids[0][input_ids.shape[1]:]
                    completion = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )
                    completions.append(completion)

                trajectories.append(Trajectory(
                    prompt=prompt,
                    completions=completions,
                ))

        self.model.train()
        logger.info(
            f"グループサンプリング完了: {len(prompts)}プロンプト × {G}生成 = "
            f"{len(prompts) * G}応答"
        )
        return trajectories

    # ── GRPO最適化 ────────────────────────────────────

    def setup_grpo_trainer(
        self,
        train_dataset: Dataset,
        reward_funcs: List[Callable],
    ) -> None:
        """trl.GRPOTrainerをセットアップ

        Args:
            train_dataset: promptカラムを含むデータセット
            reward_funcs: 報酬関数のリスト
        """
        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            raise ImportError(
                "trl>=0.14.0が必要です。`pip install trl>=0.14.0`を実行してください。"
            )

        logger.info("GRPOTrainerセットアップ中...")

        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.policy_per_device_batch_size,
            gradient_accumulation_steps=self.config.policy_gradient_accumulation_steps,
            learning_rate=self.config.policy_learning_rate,
            max_steps=self.config.policy_max_steps,
            num_generations=self.config.grpo_num_generations,
            temperature=self.config.grpo_temperature,
            max_completion_length=self.config.max_completion_length,
            max_prompt_length=self.config.max_prompt_length,
            beta=self.config.grpo_beta,
            gradient_checkpointing=self.config.gradient_checkpointing,
            bf16=self.config.bf16,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
            remove_unused_columns=False,
        )

        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("GRPOTrainerセットアップ完了")

    def train_step(self) -> Dict[str, float]:
        """GRPOトレーニングを実行

        Returns:
            トレーニングメトリクス
        """
        if self.trainer is None:
            raise RuntimeError(
                "Trainerが未セットアップです。setup_grpo_trainer()を先に実行してください。"
            )

        logger.info("GRPOトレーニング開始...")
        result = self.trainer.train()

        metrics = {
            "policy_loss": result.training_loss if hasattr(result, "training_loss") else 0.0,
        }
        if hasattr(result, "metrics"):
            metrics.update(result.metrics)

        logger.info(f"GRPOトレーニング完了: {metrics}")
        return metrics

    # ── 保存/ロード ───────────────────────────────────

    def save_adapter(self, path: Optional[str] = None) -> str:
        """LoRAアダプタを保存"""
        save_path = path or os.path.join(self.config.output_dir, "policy_adapter")
        os.makedirs(save_path, exist_ok=True)

        if self.trainer is not None:
            self.trainer.save_model(save_path)
        elif self.model is not None:
            self.model.save_pretrained(save_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        logger.info(f"ポリシーアダプタ保存: {save_path}")
        return save_path

    def load_adapter(self, path: str) -> None:
        """保存済みLoRAアダプタをロード"""
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("ベースモデルが未ロードです。")

        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"ポリシーアダプタロード: {path}")

    # ── ユーティリティ ────────────────────────────────

    def _log_gpu_memory(self, label: str = "") -> None:
        """GPU メモリ使用量をログ出力"""
        if not torch.cuda.is_available():
            return
        for i in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(
                f"GPU {i} [{label}] - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB"
            )
