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
from .training_utils import TrainingConfig, TextDataset, StreamingTextDataset
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
        
        # 継続学習ヘルパーの初期化
        from .continual_learning_helper import continual_helper
        self.helper = continual_helper
        
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
        """LoRAファインチューニング済みモデルをロード（LoRA on LoRA対応）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, prepare_model_for_kbit_training

        def _resolve_target_device(loaded_model):
            """モデルの主デバイスを推定"""
            candidates = []

            device_map = getattr(loaded_model, "hf_device_map", None)
            if isinstance(device_map, dict):
                for mapped_device in device_map.values():
                    if isinstance(mapped_device, torch.device):
                        candidates.append(mapped_device)
                    elif isinstance(mapped_device, str):
                        try:
                            device_obj = torch.device(mapped_device)
                        except (RuntimeError, TypeError):
                            continue
                        candidates.append(device_obj)

            try:
                first_param = next(loaded_model.parameters())
                candidates.append(first_param.device)
            except StopIteration:
                pass

            if torch.cuda.is_available():
                candidates.append(torch.device(f"cuda:{torch.cuda.current_device()}"))

            for device_candidate in candidates:
                if isinstance(device_candidate, str):
                    try:
                        device_candidate = torch.device(device_candidate)
                    except (RuntimeError, TypeError):
                        continue
                if isinstance(device_candidate, torch.device) and device_candidate.type == "cuda":
                    return device_candidate

            if candidates:
                device_candidate = candidates[0]
                if isinstance(device_candidate, str):
                    try:
                        return torch.device(device_candidate)
                    except (RuntimeError, TypeError):
                        return torch.device("cpu")
                if isinstance(device_candidate, torch.device):
                    return device_candidate

            return torch.device("cpu")

        def _ensure_input_modules_on_device(loaded_model, target_device):
            """入出力関連モジュールを指定デバイスに移動"""
            if target_device is None:
                return

            if not isinstance(target_device, torch.device):
                try:
                    target_device = torch.device(target_device)
                except (RuntimeError, TypeError):
                    return

            if target_device.type == "meta":
                return

            try:
                embeddings = loaded_model.get_input_embeddings()
            except AttributeError:
                embeddings = None

            if embeddings is not None and hasattr(embeddings, "weight"):
                weight_device = embeddings.weight.device
                if weight_device != target_device:
                    try:
                        embeddings = embeddings.to(target_device)
                        loaded_model.set_input_embeddings(embeddings)
                        if hasattr(loaded_model, "tie_weights"):
                            try:
                                loaded_model.tie_weights()
                            except Exception as tie_err:
                                logger.debug(f"Failed to retie weights after moving embeddings: {tie_err}")
                        logger.info(f"Moved input embeddings to {target_device}")
                    except Exception as embed_err:
                        logger.debug(f"Unable to move embeddings to {target_device}: {embed_err}")

        def _configure_loaded_model(loaded_model, preferred_device=None, enable_gradient_checkpointing=True):
            """継続学習向けの推論設定を整える"""
            target_device = preferred_device or _resolve_target_device(loaded_model)

            if enable_gradient_checkpointing and hasattr(loaded_model, "gradient_checkpointing_enable"):
                already_enabled = getattr(loaded_model, "_gradient_checkpointing_activated", False)
                if not already_enabled:
                    try:
                        loaded_model.gradient_checkpointing_enable()
                        setattr(loaded_model, "_gradient_checkpointing_activated", True)
                        logger.info("Enabled gradient checkpointing for loaded model")
                    except Exception as grad_err:
                        logger.debug(f"Failed to enable gradient checkpointing: {grad_err}")

            if hasattr(loaded_model, "config") and hasattr(loaded_model.config, "use_cache"):
                if getattr(loaded_model.config, "use_cache", True):
                    loaded_model.config.use_cache = False
                    logger.info("Disabled model cache (use_cache=False)")

            _ensure_input_modules_on_device(loaded_model, target_device)
            setattr(loaded_model, "training_device", target_device)

            return target_device

        # outputs ディレクトリからのモデルロードをサポート
        if model_path.startswith("outputs/"):
            full_path = Path(model_path)
        else:
            # 最新のフルファインチューニングモデルを自動検出
            full_path = self._find_latest_finetuned_model(model_path)
        
        logger.info(f"Loading finetuned model from: {full_path}")
        
        # 量子化チェック
        quantization_info = self.helper.detect_quantization(str(full_path))
        if quantization_info["is_quantized"] and not quantization_info["can_finetune"]:
            suggestion = self.helper.suggest_alternative_for_quantized(quantization_info)
            logger.warning(suggestion)
            raise ValueError(f"Cannot fine-tune {quantization_info['quantization_type']} quantized model")
        
        # training_info.json の確認
        training_info_path = full_path / "training_info.json"
        if training_info_path.exists():
            with open(training_info_path) as f:
                training_info = json.load(f)
                self.base_model_info = training_info
                logger.info(f"Model info: {training_info.get('model_name', 'Unknown')}")
        
        # メモリアロケータ設定
        self.helper.setup_memory_allocator(str(full_path))
        
        # モデルとトークナイザーのロード
        try:
            # LoRAアダプターが存在するかチェック
            adapter_config_path = full_path / "adapter_config.json"
            is_lora_model = adapter_config_path.exists()

            if is_lora_model:
                logger.info("LoRAアダプターを検出しました。ベースモデルをロード後、アダプターを適用します。")

                # adapter_config.jsonからベースモデルのパスを取得
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)

                base_model_name_or_path = adapter_config.get("base_model_name_or_path")

                if not base_model_name_or_path:
                    # ベースモデルのパスが指定されていない場合、デフォルトパスを試す
                    possible_paths = [
                        "Qwen/Qwen2.5-7B-Instruct",
                        "outputs/base_model",
                        str(full_path.parent / "base_model")
                    ]
                    for path in possible_paths:
                        if Path(path).exists() or "/" in path:  # HuggingFaceモデル or ローカルパス
                            base_model_name_or_path = path
                            logger.warning(f"ベースモデルパスが未設定のため、デフォルトを使用: {path}")
                            break

                logger.info(f"ベースモデル: {base_model_name_or_path}")

                # ベースモデルパスを記録（保存時に使用）
                self.base_model_path = base_model_name_or_path

                # ベースモデルをロード（メモリ効率化のため8bit量子化を検討）
                model_kwargs = self.helper.prepare_model_kwargs(
                    base_model_name_or_path,
                    force_no_quantization=False,  # メモリ節約のため量子化を許可
                    for_peft=False  # ベースモデルなのでPEFTフラグは不要
                )

                # メモリ制約がある場合は8bit量子化を使用
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    logger.info(f"GPU メモリ: {gpu_memory:.1f}GB")

                    # モデルサイズを推定（32Bモデルは約64GBのVRAMが必要）
                    if "32B" in base_model_name_or_path or "32b" in base_model_name_or_path:
                        if "quantization_config" not in model_kwargs:
                            logger.info("32Bモデルを検出 - bnb 4bit量子化を使用します")
                            # BitsAndBytesConfigを使用
                            from transformers import BitsAndBytesConfig

                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            )
                            bnb_config = self.helper.ensure_quantization_config_interface(bnb_config)
                            model_kwargs["quantization_config"] = bnb_config
                            # 古いパラメータを削除
                            model_kwargs.pop("load_in_8bit", None)
                            model_kwargs.pop("load_in_4bit", None)
                        else:
                            logger.info("32Bモデル向けの量子化設定が既に適用されています")
                    elif gpu_memory < 32:  # 32GB未満の場合は8bit量子化を推奨
                        logger.info("メモリ効率化のため8bit量子化を使用します")
                        from transformers import BitsAndBytesConfig

                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                        bnb_config = self.helper.ensure_quantization_config_interface(bnb_config)
                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs.pop("load_in_8bit", None)
                        model_kwargs.pop("load_in_4bit", None)

                logger.info("ベースモデルをロード中...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    **model_kwargs
                )

                is_8bit_model = getattr(base_model, "is_loaded_in_8bit", False)
                if is_8bit_model:
                    logger.info("Preparing 8bit base model for k-bit training")
                    base_model = prepare_model_for_kbit_training(
                        base_model,
                        use_gradient_checkpointing=True
                    )
                else:
                    _configure_loaded_model(base_model)

                # LoRAアダプターを適用
                logger.info("LoRAアダプターを適用中...")
                model = PeftModel.from_pretrained(
                    base_model,
                    str(full_path),
                    is_trainable=True  # 継続学習のため訓練可能に設定
                )

                target_device = _configure_loaded_model(
                    model,
                    preferred_device=getattr(base_model, "training_device", None),
                    enable_gradient_checkpointing=True
                )
                self.training_device = target_device

                # メモリ節約のため、マージはスキップしてLoRAモデルのまま使用
                logger.info("LoRAモデルのロードが完了しました（メモリ効率化のためマージはスキップ）")

                # GPUメモリをクリア
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                # 通常のフルモデルの場合
                logger.info("フルモデルをロード中...")

                # モデルロード用のkwargsを準備
                model_kwargs = self.helper.prepare_model_kwargs(
                    str(full_path),
                    force_no_quantization=True,  # 継続学習のため量子化を無効化
                    for_peft=False  # 通常モデルなのでPEFTフラグは不要
                )

                model = AutoModelForCausalLM.from_pretrained(
                    str(full_path),
                    **model_kwargs
                )

                target_device = _configure_loaded_model(model)
                self.training_device = target_device
            
            # ロード後の量子化チェック
            loaded_quant_info = self.helper.check_loaded_model_quantization(model)
            if loaded_quant_info["is_quantized"]:
                logger.warning(f"Model loaded with {loaded_quant_info['quantization_type']} quantization")
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(full_path),
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # クリーンアップ
            self.helper.cleanup_offload_dirs()
            self.helper.restore_memory_allocator()
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

    def _configure_for_training(self, model, config: TrainingConfig):
        """トレーニング前のモデル設定を調整し、量子化との非互換性を回避"""

        def _collect_model_candidates(root_model):
            stack = [root_model]
            seen = set()

            while stack:
                current = stack.pop()
                if id(current) in seen or current is None:
                    continue
                seen.add(id(current))
                yield current

                # 一般的なネスト先を探索
                for attr_name in ("base_model", "model", "inner_model", "wrapped_model"):
                    child = getattr(current, attr_name, None)
                    if child is not None and child is not current:
                        stack.append(child)

                get_base_model = getattr(current, "get_base_model", None)
                if callable(get_base_model):
                    try:
                        child = get_base_model()
                    except TypeError:
                        # 一部のPEFT実装は引数付き
                        try:
                            child = get_base_model(return_dict=False)
                        except Exception:
                            child = None
                    if child is not None and child is not current:
                        stack.append(child)

        candidates = list(_collect_model_candidates(model))

        def _extract_quant_config(candidate):
            quant_config = getattr(candidate, "quantization_config", None)
            if quant_config is None:
                quant_config = getattr(candidate, "_quantization_config", None)
            return self.helper.ensure_quantization_config_interface(quant_config)

        is_8bit_quantized = any(
            getattr(candidate, "is_loaded_in_8bit", False)
            for candidate in candidates
        )

        quantization_configs = [
            _extract_quant_config(candidate)
            for candidate in candidates
        ]

        uses_8bit_loader = any(
            getattr(cfg, "load_in_8bit", False)
            for cfg in quantization_configs
            if cfg is not None
        )

        uses_fp32_cpu_offload = any(
            getattr(cfg, "llm_int8_enable_fp32_cpu_offload", False)
            for cfg in quantization_configs
            if cfg is not None
        )

        gradient_checkpointing_kwargs = {"use_reentrant": False}
        setattr(config, "gradient_checkpointing_kwargs", gradient_checkpointing_kwargs)

        should_disable_gc = is_8bit_quantized or uses_8bit_loader or uses_fp32_cpu_offload

        if should_disable_gc:
            if getattr(config, "gradient_checkpointing", True):
                logger.info(
                    "Detected 8bit quantization or CPU offload. Disabling gradient checkpointing "
                    "to prevent CUDA illegal memory access."
                )
            config.gradient_checkpointing = False
            setattr(config, "gradient_checkpointing_kwargs", None)

            for candidate in candidates:
                disable_fn = getattr(candidate, "gradient_checkpointing_disable", None)
                if callable(disable_fn):
                    try:
                        disable_fn()
                    except RuntimeError as err:
                        logger.debug(
                            "Failed to disable gradient checkpointing on %s: %s",
                            type(candidate),
                            err,
                        )
                elif hasattr(candidate, "gradient_checkpointing_enable"):
                    # 一部モデルはdisableメソッドを持たないため、内部状態を直接リセット
                    if hasattr(candidate, "_gradient_checkpointing_func"):
                        candidate._gradient_checkpointing_func = None

        else:
            if getattr(config, "gradient_checkpointing", True):
                for candidate in candidates:
                    enable_fn = getattr(candidate, "gradient_checkpointing_enable", None)
                    if callable(enable_fn):
                        try:
                            enable_fn(**gradient_checkpointing_kwargs)
                            setattr(candidate, "_gradient_checkpointing_activated", True)
                        except TypeError:
                            enable_fn()
                        except RuntimeError as err:
                            logger.warning(
                                "Failed to enable gradient checkpointing with custom kwargs on %s: %s. "
                                "Falling back to disabled state.",
                                type(candidate),
                                err,
                            )
                            config.gradient_checkpointing = False
                            setattr(config, "gradient_checkpointing_kwargs", None)
                            break

        for candidate in candidates:
            candidate_config = getattr(candidate, "config", None)
            if candidate_config is not None and hasattr(candidate_config, "use_cache"):
                if getattr(candidate_config, "use_cache", True):
                    candidate_config.use_cache = False

        return model

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
        learning_rate: float = 2e-5,
        progress_callback: Optional[callable] = None
    ):
        """継続学習タスクを実行"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Continual Learning Task: {task_name}")
        logger.info(f"{'='*50}")
        
        def _unwrap_base_model(candidate_model):
            if candidate_model is None:
                return None
            get_base = getattr(candidate_model, "get_base_model", None)
            if callable(get_base):
                try:
                    base = get_base()
                except TypeError:
                    try:
                        base = get_base(return_dict=False)
                    except Exception:
                        base = None
                if base is not None:
                    return base
            base_attr = getattr(candidate_model, "base_model", None)
            if base_attr is not None and base_attr is not candidate_model:
                return base_attr
            return candidate_model

        def _collect_linear_module_suffixes(root_model):
            suffixes = set()
            try:
                for name, module in root_model.named_modules():
                    weight = getattr(module, "weight", None)
                    if weight is None:
                        continue
                    if getattr(weight, "dim", lambda: 0)() != 2:
                        continue
                    suffixes.add(name.split(".")[-1])
            except Exception:
                pass
            return suffixes

        def _infer_lora_targets(root_model):
            base_model = _unwrap_base_model(root_model)
            if base_model is None:
                return ["q_proj", "k_proj", "v_proj", "o_proj"]

            config = getattr(base_model, "config", None)
            model_type = getattr(config, "model_type", "")
            model_type = (model_type or "").lower()

            preferred_candidates = []
            if any(token in model_type for token in ("llama", "qwen", "mistral", "deepseek", "baichuan")):
                preferred_candidates = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif "gpt-neox" in model_type or "gptneox" in model_type:
                preferred_candidates = [
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ]
            elif "gpt" in model_type:
                preferred_candidates = ["c_attn", "c_proj", "c_fc"]
            elif "bloom" in model_type:
                preferred_candidates = [
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ]
            elif "opt" in model_type:
                preferred_candidates = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                preferred_candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]

            available_suffixes = _collect_linear_module_suffixes(base_model)
            resolved = [name for name in preferred_candidates if name in available_suffixes]
            if resolved:
                return resolved

            # フォールバック: "proj"や"gate"を含む線形層を優先的に選択
            heuristics = [
                name for name in available_suffixes
                if any(token in name for token in ("proj", "gate", "up", "down"))
            ]
            if heuristics:
                return sorted(heuristics)

            return sorted(available_suffixes)[:8] if available_suffixes else ["q_proj", "v_proj"]

        def _auto_tune_training_config(training_config: TrainingConfig, quantized: bool):
            """メモリ制約に応じてバッチ/シーケンス設定を調整"""

            if training_config is None:
                return

            target_batch = max(1, min(batch_size, 2 if quantized else batch_size))
            max_seq = getattr(training_config, "max_seq_length", 256) or 256
            target_seq = min(max_seq, 512 if quantized else 1024)

            adjustments = {}
            try:
                adjustments = training_config.apply_memory_optimizations(
                    target_batch_size=target_batch,
                    target_seq_length=target_seq,
                    preserve_effective_batch=False,
                    min_gradient_accumulation=1,
                )
            except Exception as mem_err:
                logger.debug(f"Failed to auto tune training config: {mem_err}")

            for key, (old_value, new_value) in adjustments.items():
                logger.info(f"Auto memory adjustment -> {key}: {old_value} -> {new_value}")

            if torch.cuda.is_available():
                bf16_supported = False
                if hasattr(torch.cuda, "is_bf16_supported"):
                    try:
                        bf16_supported = bool(torch.cuda.is_bf16_supported())
                    except Exception:
                        bf16_supported = False
                else:
                    try:
                        capability = torch.cuda.get_device_capability()
                        bf16_supported = capability[0] >= 8
                    except Exception:
                        bf16_supported = False

                training_config.mixed_precision = "bf16" if bf16_supported else "fp16"
                training_config.fp16 = not bf16_supported
            else:
                training_config.mixed_precision = "no"
                training_config.fp16 = False
        
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
            num_epochs=epochs,  # 正しいパラメータ名に修正
            batch_size=batch_size,  # 正しいパラメータ名に修正
            gradient_accumulation_steps=16,
            learning_rate=learning_rate,
            warmup_steps=100,  # warmup_ratioではなくwarmup_steps
            fp16=True,
            gradient_checkpointing=True,
            save_steps=500,  # save_strategyではなくsave_steps
            logging_steps=10,
            eval_steps=100,  # eval_stepsを追加
            max_grad_norm=1.0  # max_grad_normを追加
        )

        # データセットの準備
        logger.info(f"Loading dataset from: {train_dataset_path}")
        # ファイルパスからデータを読み込む場合はStreamingTextDatasetを使用
        train_dataset = StreamingTextDataset(
            file_path=train_dataset_path,
            tokenizer=tokenizer,
            max_length=256  # メモリ効率のため短くする
        )
        
        # LoRA継続学習トレーナーの作成
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        import torch

        # 既にPEFTモデルの場合とそうでない場合を判別
        is_peft_model = isinstance(model, PeftModel)
        
        # 量子化モデルのチェック（BitsAndBytesConfigオブジェクトへ互換メソッドを注入）
        is_quantized = False
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            quant_config = self.helper.ensure_quantization_config_interface(
                model.config.quantization_config
            )
            if quant_config:
                load_in_4bit = getattr(quant_config, 'load_in_4bit', None)
                if load_in_4bit is None and isinstance(quant_config, dict):
                    load_in_4bit = quant_config.get('load_in_4bit')
                load_in_8bit = getattr(quant_config, 'load_in_8bit', None)
                if load_in_8bit is None and isinstance(quant_config, dict):
                    load_in_8bit = quant_config.get('load_in_8bit')

                if load_in_4bit:
                    is_quantized = True
                    logger.warning("Model is quantized with 4-bit, will use LoRA for training")
                elif load_in_8bit:
                    is_quantized = True
                    logger.warning("Model is quantized with 8-bit, will use LoRA for training")

        _auto_tune_training_config(config, is_quantized)
        dataset_max_seq = getattr(config, "max_seq_length", 256) or 256
        if hasattr(train_dataset, "set_max_length"):
            train_dataset.set_max_length(dataset_max_seq)

        target_modules = _infer_lora_targets(model)

        if not is_peft_model:
            # まだLoRAが適用されていない場合のみ新しいLoRAアダプターを追加
            # 量子化モデルの場合は必ずLoRAが必要
            if is_quantized:
                # 量子化モデルの場合、prepare_model_for_kbit_trainingを呼び出す
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                logger.info("Prepared quantized model for LoRA training")

            lora_config = LoraConfig(
                r=8,  # LoRAのランク
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            # LoRAアダプターを適用
            logger.info("新しいLoRAアダプターを適用中...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            # 既にPEFTモデルの場合
            logger.info("既存のLoRAアダプターを使用して継続学習を実施")

            try:
                # 1. まず量子化モデルの準備を行う（これがパラメータをフリーズする前に）
                if is_quantized:
                    from peft import prepare_model_for_kbit_training

                    # PEFTモデルのベースモデルを取得
                    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model

                    # prepare_model_for_kbit_trainingを適用
                    # 注意: これはすべてのパラメータをrequires_grad=Falseにする
                    if hasattr(base_model, 'model'):
                        prepare_model_for_kbit_training(base_model.model, use_gradient_checkpointing=True)
                    else:
                        prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

                    logger.info("Prepared quantized model for k-bit training")

                # 2. PEFTの組み込みメソッドを使用してLoRAパラメータのみを学習可能にする
                # これはprepare_model_for_kbit_trainingの後に実行する必要がある
                if hasattr(model, 'mark_only_lora_as_trainable'):
                    # PEFTの推奨メソッドを使用
                    model.mark_only_lora_as_trainable()
                    logger.info("Marked only LoRA parameters as trainable using PEFT built-in method")
                else:
                    # フォールバック: 手動でLoRAパラメータを設定
                    logger.info("Manually setting LoRA parameters as trainable")
                    for name, param in model.named_parameters():
                        # PEFTが使用する標準的なLoRAパラメータ名
                        if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B', 'lora_dropout']):
                            param.requires_grad = True
                            logger.debug(f"Enabled gradient for: {name}")
                        else:
                            param.requires_grad = False

                # 3. gradient checkpointingとinput gradientsを有効化
                if hasattr(model, 'enable_input_require_grads'):
                    model.enable_input_require_grads()
                    logger.info("Enabled input_require_grads for gradient computation")

                # 4. アダプターレイヤーが有効になっていることを確認
                if hasattr(model, 'enable_adapter_layers'):
                    # PeftModelレベルでenable_adapter_layersを呼び出す
                    model.enable_adapter_layers()
                    logger.info("Enabled adapter layers on PeftModel")

                # 5. 学習可能パラメータ数の確認
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in model.parameters())

                # 6. trainable_params == 0の場合の追加処理
                if trainable_params == 0:
                    logger.warning("No trainable parameters found after initial setup!")
                    logger.info("Attempting recovery by re-enabling LoRA parameters...")

                    # 再度LoRAパラメータを有効化（prepare_model_for_kbit_trainingによってフリーズされた可能性）
                    for name, param in model.named_parameters():
                        if 'lora' in name.lower():
                            param.requires_grad = True

                    # 再カウント
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    if trainable_params == 0:
                        raise ValueError("Failed to enable any trainable parameters for continual learning")

                # 7. 最終的な情報をログ出力
                logger.info(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")

                # PEFTモデルの詳細情報を表示
                if hasattr(model, 'print_trainable_parameters'):
                    model.print_trainable_parameters()

            except Exception as e:
                logger.error(f"Error setting up PEFT model for continual learning: {str(e)}")
                logger.error(f"Model type: {type(model)}")
                logger.error(f"Is quantized: {is_quantized}")
                raise

        # gradient_checkpointingの設定
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()

        # use_cacheを無効化（gradient checkpointingと併用不可）
        if hasattr(model, 'config'):
            model.config.use_cache = False

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
        if progress_callback:
            progress_callback(30, "トレーニング開始...")

        trainer.train()

        if progress_callback:
            progress_callback(70, "トレーニング完了")

        # Fisher行列の計算と保存（オプション）
        # 32Bモデルの場合、メモリ不足を防ぐためFisher計算をスキップ可能
        skip_fisher = False
        if is_quantized or "32B" in str(getattr(self, 'base_model_path', '')):
            logger.warning("Large quantized model detected - Fisher matrix computation may cause OOM")
            logger.warning("Skipping Fisher matrix computation to prevent memory issues")
            logger.warning("Note: EWC regularization will not be applied (using standard LoRA fine-tuning)")
            skip_fisher = True

        if not skip_fisher:
            logger.info("Computing Fisher matrix for current task...")
            if progress_callback:
                progress_callback(80, "Fisher行列を計算中...")

            try:
                self._compute_and_save_fisher(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    dataset_path=train_dataset_path,
                    task_name=task_name
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("OOM during Fisher computation - skipping Fisher matrix")
                    skip_fisher = True
                    torch.cuda.empty_cache()
                else:
                    raise
        else:
            logger.info("Fisher matrix computation skipped (memory optimization)")
        
        # モデルの保存（LoRAアダプターとして保存）
        logger.info(f"Saving LoRA adapter to: {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # adapter_config.jsonにベースモデル情報を追加
        adapter_config_path = Path(output_dir) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            # ベースモデルのパスを保存（次回のロード時に使用）
            if hasattr(self, 'base_model_path'):
                adapter_config["base_model_name_or_path"] = self.base_model_path

            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)

        # タスク履歴の更新
        task_info = {
            "task_name": task_name,
            "timestamp": timestamp,
            "model_path": output_dir,
            "model_type": "lora_adapter",  # モデルタイプを記録
            "base_model": getattr(self, 'base_model_path', 'Unknown'),
            "fisher_path": str(self.ewc_data_path / f"fisher_{task_name}.pt"),
            "dataset": train_dataset_path,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "ewc_lambda": fisher_importance,
            "lora_r": 8,  # LoRA設定を記録
            "lora_alpha": 16,
            "quantized": is_quantized  # 量子化状態を記録
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
            dataset = StreamingTextDataset(
                file_path=dataset_path,
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
            dataset = StreamingTextDataset(
                file_path=dataset_path,
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
        
        dataset = StreamingTextDataset(
            file_path=dataset_path,
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

    def get_latest_model_path(self) -> Optional[Path]:
        """最新の継続学習モデルのパスを取得"""
        if self.task_history:
            latest_task = self.task_history[-1]
            model_path = Path(latest_task.get("model_path", ""))
            if model_path.exists():
                return model_path

        # タスク履歴がない場合は、outputsディレクトリから最新を探す
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            continual_models = list(outputs_dir.glob("continual_task_*"))
            if continual_models:
                # 最新のモデルを返す
                return max(continual_models, key=lambda p: p.stat().st_mtime)

        return None
