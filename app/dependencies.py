"""Shared application-level dependencies and state."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

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
]
