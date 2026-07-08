"""Shared application-level dependencies and state."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.training.state import training_tasks


logger = logging.getLogger("app.dependencies")


def _resolve_project_root() -> Path:
    """Determine the project root inside and outside containers."""
    workspace = os.environ.get("PROJECT_ROOT")
    if workspace:
        return Path(workspace).resolve()
    return Path(os.getcwd()).resolve()


PROJECT_ROOT = _resolve_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADED_DIR = PROJECT_ROOT / "data" / "uploaded"
CONTINUAL_LEARNING_DIR = PROJECT_ROOT / "data" / "continual_learning"
CONTINUAL_TASKS_FILE = CONTINUAL_LEARNING_DIR / "tasks_state.json"

# NOTE: Maintained as a plain dict for compatibility with legacy callers.
continual_tasks: Dict[str, Any] = {}


# Shared runtime caches/state
model_cache: Dict[str, Any] = {}
executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Feature availability flags (set during app initialisation in main_unified)
# ---------------------------------------------------------------------------
RAG_AVAILABLE: bool = False
METRICS_AVAILABLE: bool = False
OLLAMA_AVAILABLE: bool = False
metrics_collector: Any = None

# Continual learning task manager reference
task_manager: Any = None

# ---------------------------------------------------------------------------
# Template engine
# ---------------------------------------------------------------------------
from fastapi.templating import Jinja2Templates  # noqa: E402

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# RAG application singleton
# ---------------------------------------------------------------------------
from app.rag_application import RAGApplication  # noqa: E402

rag_app = RAGApplication()

# ---------------------------------------------------------------------------
# Available models list
# ---------------------------------------------------------------------------
available_models: List[Dict[str, Any]] = [
    {
        "name": "distilgpt2",
        "description": "軽量な英語モデル（テスト用）",
        "size": "82MB",
        "status": "available",
        "gpu_requirement": "なし"
    },
    {
        "name": "rinna/japanese-gpt2-small",
        "description": "日本語GPT-2 Small（Rinna）",
        "size": "110MB",
        "status": "available",
        "gpu_requirement": "なし"
    },
    {
        "name": "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        "description": "Japanese StableLM 3B Instruct（推奨）",
        "size": "3B",
        "status": "available",
        "gpu_requirement": "8GB"
    },
    {
        "name": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        "description": "ELYZA日本語Llama-2 7B Instruct",
        "size": "7B",
        "status": "gpu-required",
        "gpu_requirement": "16GB"
    },
    {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 2.5 14B Instruct（推奨）",
        "size": "14B",
        "status": "gpu-required",
        "gpu_requirement": "28GB"
    },
    {
        "name": "cyberagent/calm3-22b-chat",
        "description": "CyberAgent CALM3 22B Chat",
        "size": "22B",
        "status": "gpu-required",
        "gpu_requirement": "44GB"
    },
    {
        "name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "description": "DeepSeek R1 Distill Qwen 32B 日本語特化（Ollama推奨）",
        "size": "32B",
        "status": "ollama-recommended",
        "gpu_requirement": "20GB (Ollama使用時)"
    },
    {
        "name": "Qwen/Qwen2.5-17B-Instruct",
        "description": "Qwen 2.5 17B Instruct",
        "size": "17B",
        "status": "gpu-required",
        "gpu_requirement": "34GB"
    },
    {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 2.5 32B Instruct（Ollama推奨）",
        "size": "32B",
        "status": "ollama-recommended",
        "gpu_requirement": "20GB (Ollama使用時)"
    }
]


def get_saved_models() -> List[Dict[str, Any]]:
    """保存済みモデル一覧を取得"""
    saved_models: List[Dict[str, Any]] = []
    if os.path.exists("/workspace"):
        project_root = Path("/workspace")
    else:
        project_root = Path(os.getcwd())

    outputs_path = project_root / "outputs"

    # プロジェクトルートからLoRAモデルを検索
    for model_dir in project_root.glob("lora_demo_*"):
        if model_dir.is_dir():
            if (model_dir / "adapter_config.json").exists() or (model_dir / "adapter_model.safetensors").exists():
                saved_models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "LoRA",
                    "size": "~1.6MB",
                    "base_model": "不明",
                    "training_method": "lora"
                })

    # outputsディレクトリも検索
    if outputs_path.exists():
        for model_dir in outputs_path.iterdir():
            if model_dir.is_dir():
                info_path = model_dir / "training_info.json"
                model_type = "Unknown"
                model_size = "Unknown"
                base_model = "不明"
                training_method = "unknown"
                training_data_size = 0

                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            training_method = info.get("training_method", "unknown")
                            base_model = info.get("base_model", "不明")
                            training_data_size = info.get("training_data_size", 0)

                            if training_method == "full":
                                model_type = "フルファインチューニング"
                                model_size = "~500MB+"
                            elif training_method == "qlora":
                                model_type = "QLoRA (4bit)"
                                model_size = "~1.0MB"
                            elif training_method == "continual_ewc":
                                model_type = "継続学習 (EWC)"
                                model_size = "~500MB+"
                            else:
                                model_type = "LoRA"
                                model_size = "~1.6MB"
                    except Exception as e:
                        logger.warning(f"training_info.jsonの読み込みに失敗: {e}")
                        if "continual_task" in model_dir.name.lower():
                            model_type = "継続学習"
                            training_method = "continual"
                        elif "lora" in model_dir.name.lower():
                            model_type = "LoRA"
                            training_method = "lora"
                        elif "qlora" in model_dir.name.lower():
                            model_type = "QLoRA"
                            training_method = "qlora"
                        elif "full" in model_dir.name.lower():
                            model_type = "フルファインチューニング"
                            training_method = "full"

                has_model_files = (
                    (model_dir / "adapter_model.safetensors").exists() or
                    (model_dir / "pytorch_model.bin").exists() or
                    (model_dir / "model.safetensors").exists() or
                    any(model_dir.glob("*.safetensors")) or
                    any(model_dir.glob("*.bin"))
                )

                has_tokenizer = (
                    (model_dir / "tokenizer.json").exists() or
                    (model_dir / "tokenizer_config.json").exists()
                )

                if has_model_files:
                    saved_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": model_type,
                        "size": model_size,
                        "base_model": base_model,
                        "training_method": training_method,
                        "training_data_size": training_data_size,
                        "has_tokenizer": has_tokenizer,
                        "has_model_files": has_model_files
                    })

    logger.info(f"検出された保存済みモデル: {len(saved_models)}個")
    return saved_models


__all__ = [
    "logger",
    "PROJECT_ROOT",
    "OUTPUTS_DIR",
    "UPLOADED_DIR",
    "CONTINUAL_LEARNING_DIR",
    "CONTINUAL_TASKS_FILE",
    "continual_tasks",
    "model_cache",
    "executor",
    "training_tasks",
    "RAG_AVAILABLE",
    "METRICS_AVAILABLE",
    "OLLAMA_AVAILABLE",
    "metrics_collector",
    "task_manager",
    "templates",
    "rag_app",
    "available_models",
    "get_saved_models",
]
