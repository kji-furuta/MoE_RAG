"""
共通の依存関係と設定
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

# ロガー設定
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(os.getcwd())

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
UPLOADED_DIR = DATA_DIR / "uploaded"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONTINUAL_LEARNING_DIR = DATA_DIR / "continual_learning"

# モデルキャッシュ
model_cache: Dict[str, Any] = {}

# タスク管理
training_tasks: Dict[str, Any] = {}
continual_tasks: Dict[str, Any] = {}

# 継続学習タスクの永続化ファイル
CONTINUAL_TASKS_FILE = CONTINUAL_LEARNING_DIR / "tasks_state.json"

# エグゼキュータ
executor = ThreadPoolExecutor(max_workers=2)

# RAG設定
RAG_AVAILABLE = True  # デフォルト値、実際はRAGモジュールの初期化で設定される