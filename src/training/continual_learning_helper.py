"""
継続学習のヘルパー関数とユーティリティ
Codex MCPの提案に基づく改善実装
"""

import os
import tempfile
import shutil
import logging
import types
from pathlib import Path
from typing import Dict, Optional, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class ContinualLearningHelper:
    """継続学習のヘルパークラス"""

    def __init__(self):
        self.offload_dirs: List[Path] = []
        self.original_cuda_alloc_conf: Optional[str] = None

    def setup_memory_allocator(self, model_name: str) -> None:
        """
        メモリアロケータの設定（既存値を保持）

        Args:
            model_name: モデル名
        """
        # モデル名を正規化（大文字小文字を統一）
        model_name_lower = model_name.lower()

        # 22B/32Bモデルのチェック
        if "32b" in model_name_lower or "22b" in model_name_lower:
            # 既存のPYTORCH_CUDA_ALLOC_CONFを保存
            self.original_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

            # 新しい設定を追加
            new_conf = "expandable_segments:True,max_split_size_mb:512"

            if self.original_cuda_alloc_conf:
                # 既存の設定がある場合は追加
                if new_conf not in self.original_cuda_alloc_conf:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{self.original_cuda_alloc_conf},{new_conf}"
                    logger.info(f"メモリアロケータ設定を追加: {new_conf}")
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
                logger.info(f"メモリアロケータ設定: {new_conf}")

    def restore_memory_allocator(self) -> None:
        """メモリアロケータ設定を復元"""
        if self.original_cuda_alloc_conf is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.original_cuda_alloc_conf
            logger.info("メモリアロケータ設定を復元")
        elif "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            logger.info("メモリアロケータ設定をクリア")

    def create_offload_dir(self) -> Path:
        """
        オフロードディレクトリを作成（追跡付き）

        Returns:
            作成したディレクトリのパス
        """
        offload_dir = Path(tempfile.mkdtemp(prefix="continual_offload_"))
        self.offload_dirs.append(offload_dir)
        logger.info(f"オフロードディレクトリを作成: {offload_dir}")
        return offload_dir

    def cleanup_offload_dirs(self) -> None:
        """すべてのオフロードディレクトリをクリーンアップ"""
        for offload_dir in self.offload_dirs:
            if offload_dir.exists():
                try:
                    shutil.rmtree(offload_dir, ignore_errors=True)
                    logger.info(f"オフロードディレクトリを削除: {offload_dir}")
                except Exception as e:
                    logger.warning(f"オフロードディレクトリの削除に失敗: {offload_dir}, {e}")
        self.offload_dirs.clear()

    def detect_quantization(self, model_path: str) -> Dict[str, Any]:
        """
        モデルの量子化状態を検出

        Args:
            model_path: モデルのパス

        Returns:
            量子化情報の辞書
        """
        quantization_info = {
            "is_quantized": False,
            "quantization_type": None,
            "bits": None,
            "can_finetune": True
        }

        model_path = Path(model_path)

        # quantization_config.jsonのチェック
        quant_config_path = model_path / "quantization_config.json"
        if quant_config_path.exists():
            import json
            with open(quant_config_path) as f:
                config = json.load(f)
                if "quant_method" in config:
                    quantization_info["is_quantized"] = True
                    quantization_info["quantization_type"] = config["quant_method"]
                    quantization_info["bits"] = config.get("bits", "unknown")
                    quantization_info["can_finetune"] = False
                    logger.warning(f"量子化モデルを検出: {config['quant_method']} {config.get('bits', '')}bit")

        # GGUFファイルのチェック
        if list(model_path.glob("*.gguf")):
            quantization_info["is_quantized"] = True
            quantization_info["quantization_type"] = "GGUF"
            quantization_info["can_finetune"] = False
            logger.warning("GGUFモデルを検出")

        # モデル名からの推測
        model_name_lower = str(model_path).lower()
        if any(pattern in model_name_lower for pattern in ["gptq", "awq", "exl2", "4bit", "8bit"]):
            quantization_info["is_quantized"] = True
            if "gptq" in model_name_lower:
                quantization_info["quantization_type"] = "GPTQ"
            elif "awq" in model_name_lower:
                quantization_info["quantization_type"] = "AWQ"
            elif "exl2" in model_name_lower:
                quantization_info["quantization_type"] = "EXL2"
            quantization_info["can_finetune"] = False
            logger.warning(f"モデル名から量子化を推測: {model_name_lower}")

        return quantization_info

    def check_loaded_model_quantization(self, model) -> Dict[str, Any]:
        """
        ロード済みモデルの量子化状態をチェック

        Args:
            model: ロード済みのモデル

        Returns:
            量子化状態の情報
        """
        quantization_info = {
            "is_quantized": False,
            "quantization_type": None,
            "can_finetune": True
        }

        # 8bit量子化のチェック
        if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
            quantization_info["is_quantized"] = True
            quantization_info["quantization_type"] = "8bit"
            quantization_info["can_finetune"] = False
            logger.warning("8bit量子化モデルが検出されました")

        # 4bit量子化のチェック
        elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
            quantization_info["is_quantized"] = True
            quantization_info["quantization_type"] = "4bit"
            quantization_info["can_finetune"] = False
            logger.warning("4bit量子化モデルが検出されました")

        # Transformersの一部バージョンではquantization_configが辞書インターフェースを期待
        quant_config = getattr(model.config, "quantization_config", None)
        self.ensure_quantization_config_interface(quant_config)

        return quantization_info

    def ensure_quantization_config_interface(self, quant_config: Any) -> Any:
        """BitsAndBytesConfigを辞書互換インターフェースでラップ.

        古いTransformersはquantization_configに対してdict.getを呼び出すため、
        ランタイムでgetメソッドを注入して互換性を確保する。
        """

        if quant_config is None:
            return None

        if isinstance(quant_config, dict):
            return quant_config

        if not hasattr(quant_config, "get"):
            def _dict_get(self_obj, key, default=None):
                return getattr(self_obj, key, default)

            quant_config.get = types.MethodType(_dict_get, quant_config)

        if not hasattr(quant_config, "keys"):
            def _dict_keys(self_obj):
                return [attr for attr in dir(self_obj) if not attr.startswith("_")]

            quant_config.keys = types.MethodType(_dict_keys, quant_config)

        if not hasattr(quant_config, "items"):
            def _dict_items(self_obj):
                return [(key, getattr(self_obj, key)) for key in self_obj.keys()]

            quant_config.items = types.MethodType(_dict_items, quant_config)

        if not hasattr(quant_config, "to_dict"):
            def _to_dict(self_obj):
                return {
                    key: getattr(self_obj, key)
                    for key in self_obj.keys()
                    if not callable(getattr(self_obj, key))
                }

            quant_config.to_dict = types.MethodType(_to_dict, quant_config)

        return quant_config

    def get_optimal_dtype(self, model_path: str) -> torch.dtype:
        """
        モデルに最適なdtypeを取得

        Args:
            model_path: モデルのパス

        Returns:
            最適なtorch.dtype
        """
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)

            # モデル設定から推奨dtypeを取得
            if hasattr(config, "torch_dtype"):
                dtype_str = config.torch_dtype
                if dtype_str == "bfloat16":
                    return torch.bfloat16
                elif dtype_str == "float16":
                    return torch.float16
                elif dtype_str == "float32":
                    return torch.float32

            # モデルサイズに基づくデフォルト
            model_name_lower = str(model_path).lower()
            if "32b" in model_name_lower or "22b" in model_name_lower:
                # 大規模モデルはbfloat16を推奨
                return torch.bfloat16

        except Exception as e:
            logger.warning(f"dtype取得エラー: {e}")

        # デフォルトはfloat16
        return torch.float16

    def get_multi_gpu_memory_map(self) -> Dict[int, str]:
        """
        マルチGPU環境用のメモリマップを取得

        Returns:
            デバイスIDとメモリサイズのマッピング
        """
        memory_map = {}

        if torch.cuda.is_available():
            try:
                from accelerate import utils as accelerate_utils

                # accelerateのget_max_memory()を使用
                max_memory = accelerate_utils.get_max_memory()

                # メモリマップを構築
                for device_id, memory_bytes in max_memory.items():
                    if isinstance(device_id, int):  # GPUデバイス
                        # バイトをGBに変換（余裕を持たせるため0.9倍）
                        memory_gb = int(memory_bytes * 0.9 / (1024**3))
                        memory_map[device_id] = f"{memory_gb}GB"
                        logger.info(f"GPU {device_id}: {memory_gb}GB available")
                    elif device_id == "cpu":
                        # CPUメモリ（余裕を持たせるため0.7倍）
                        memory_gb = int(memory_bytes * 0.7 / (1024**3))
                        memory_map["cpu"] = f"{memory_gb}GB"
                        logger.info(f"CPU: {memory_gb}GB available")

            except ImportError:
                # accelerateが利用できない場合のフォールバック
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = int(props.total_memory * 0.9 / (1024**3))
                    memory_map[i] = f"{memory_gb}GB"
                    logger.info(f"GPU {i}: {memory_gb}GB available")

                # CPUメモリ（デフォルト値）
                memory_map["cpu"] = "40GB"

        else:
            # GPUが利用できない場合
            logger.warning("GPUが利用できません。CPUのみで実行します。")
            memory_map["cpu"] = "40GB"

        return memory_map

    def prepare_model_kwargs(self, model_path: str, force_no_quantization: bool = True, for_peft: bool = False) -> Dict[str, Any]:
        """
        モデルロード用のkwargsを準備

        Args:
            model_path: モデルのパス
            force_no_quantization: 量子化を強制的に無効化するか
            for_peft: PEFTモデル（LoRAアダプター）用かどうか

        Returns:
            model_kwargsの辞書
        """
        # オフロードディレクトリの作成
        offload_dir = self.create_offload_dir()

        # 最適なdtypeを取得
        dtype = self.get_optimal_dtype(model_path)

        # メモリマップを取得（必要に応じて後で調整）
        memory_map = self.get_multi_gpu_memory_map()
        adjusted_memory_map = dict(memory_map)

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "offload_folder": str(offload_dir),
            "max_memory": adjusted_memory_map
        }

        # PEFTモデルの場合のみoffload_dirを追加
        if for_peft:
            model_kwargs["offload_dir"] = str(offload_dir)

        # 継続学習用の量子化設定
        normalized_model_path = str(model_path).lower()
        is_32b_model = "32b" in normalized_model_path

        if force_no_quantization and not is_32b_model:
            model_kwargs["load_in_8bit"] = False
            model_kwargs["load_in_4bit"] = False
            # quantization_configを無効化
            model_kwargs["quantization_config"] = None
        else:
            if is_32b_model:
                # 32Bモデルは8bitではなくbnb 4bitでロード
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                bnb_config = self.ensure_quantization_config_interface(bnb_config)
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs.pop("load_in_8bit", None)
                model_kwargs.pop("load_in_4bit", None)
                model_kwargs["device_map"] = "balanced"

                def _parse_gb(value: Any) -> Optional[int]:
                    if isinstance(value, str):
                        stripped = value.strip().lower()
                        if stripped.endswith("gib"):
                            stripped = stripped[:-3]
                        elif stripped.endswith("gb"):
                            stripped = stripped[:-2]
                        try:
                            return int(float(stripped))
                        except ValueError:
                            return None
                    return None

                # GPU側の上限をやや下げることでCPUオフロードの余地を作る
                for device_idx, limit in list(adjusted_memory_map.items()):
                    if isinstance(device_idx, int):
                        gpu_limit = _parse_gb(limit)
                        if gpu_limit and gpu_limit > 0:
                            adjusted_memory_map[device_idx] = f"{max(int(gpu_limit * 0.85), 12)}GB"

                cpu_limit_gb = _parse_gb(adjusted_memory_map.get("cpu")) or 0
                # CPUオフロードの許容量を引き上げてGPU負荷を緩和
                target_cpu_limit = max(cpu_limit_gb, 192)
                adjusted_memory_map["cpu"] = f"{target_cpu_limit}GB"
                model_kwargs["max_memory"] = adjusted_memory_map
                logger.info("32Bモデルを検出: bnb 4bit量子化を適用します")
            else:
                # 既存のquantization_configがない場合のみデフォルト設定を適用
                if "quantization_config" not in model_kwargs:
                    # モデルサイズに応じた自動量子化
                    model_kwargs["load_in_8bit"] = False
                    model_kwargs["load_in_4bit"] = False

        logger.info(f"モデルロード設定: dtype={dtype}, offload={offload_dir}")

        return model_kwargs

    def suggest_alternative_for_quantized(self, quantization_info: Dict[str, Any]) -> str:
        """
        量子化モデルに対する代替案を提案

        Args:
            quantization_info: 量子化情報

        Returns:
            提案メッセージ
        """
        if not quantization_info["is_quantized"]:
            return ""

        quant_type = quantization_info["quantization_type"]

        suggestions = []
        suggestions.append(f"⚠️ {quant_type}量子化モデルは直接ファインチューニングできません。")
        suggestions.append("\n推奨される代替案：")
        suggestions.append("1. LoRAまたはQLoRAを使用した効率的なファインチューニング")
        suggestions.append("2. 非量子化版のモデルを使用")
        suggestions.append("3. 量子化前のベースモデルから開始")

        if quant_type in ["4bit", "8bit"]:
            suggestions.append(f"\n💡 ヒント: load_in_{quant_type[0]}bit=False を設定して非量子化版をロードできます")

        return "\n".join(suggestions)


# グローバルヘルパーインスタンス
continual_helper = ContinualLearningHelper()
