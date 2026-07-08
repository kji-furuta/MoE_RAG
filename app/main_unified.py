#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Unified Implementation
統合されたWebインターフェース実装

This module serves as the thin orchestrator: it creates the FastAPI app,
registers middleware, mounts static files, includes all routers, and
defines application lifecycle events (startup / shutdown).
"""

# PyTorchメモリ管理の最適化
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # トークナイザーの警告を抑制

import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.dependencies as _deps
from app.dependencies import rag_app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional subsystem imports (set feature-flags in dependencies)
# ---------------------------------------------------------------------------

# RAG system
try:
    os.environ["RAG_DISABLE_MODEL_LOAD"] = "true"
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.rag.core.query_engine import RoadDesignQueryEngine, QueryResult  # noqa: F401
    from src.rag.indexing.metadata_manager import MetadataManager  # noqa: F401
    RAG_AVAILABLE = True
    _deps.RAG_AVAILABLE = True
    logger.info("RAG system components loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    _deps.RAG_AVAILABLE = False
    logger.warning(f"RAG system not available: {e}")

# Prometheus metrics
try:
    from app.monitoring import metrics_collector, get_prometheus_metrics  # noqa: F401
    METRICS_AVAILABLE = True
    _deps.METRICS_AVAILABLE = True
    _deps.metrics_collector = metrics_collector
    logger.info("Prometheusメトリクスをインポートしました")
except Exception as e:
    METRICS_AVAILABLE = False
    logger.warning(f"メトリクスシステムのインポートをスキップ: {e}")

# Ollama integration
try:
    scripts_convert_path = Path(__file__).parent.parent / "scripts" / "convert"
    sys.path.insert(0, str(scripts_convert_path))
    from ollama_integration import OllamaIntegration  # noqa: F401
    _deps.OLLAMA_AVAILABLE = True
    logger.info("Ollama統合が利用可能です")
except ImportError as e:
    logger.warning(f"Ollama統合が利用できません: {e}")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

def _find_static_directory() -> str:
    """静的ファイルディレクトリを検索"""
    static_path = Path(__file__).parent / "static"
    if static_path.is_dir():
        return str(static_path)

    project_root = Path(os.getcwd())
    for candidate in (project_root / "static", project_root / "app" / "static"):
        if candidate.is_dir():
            return str(candidate)

    return str(project_root / "static")


static_dir = _find_static_directory()
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------------------------------------------------------------------------
# Router registration (order-independent; prefix handled inside each router)
# ---------------------------------------------------------------------------

# --- Optional routers (external modules that may not be installed) ---
_optional_routers = [
    ("app.moe_rag_endpoints", "MoE-RAG endpoints", None),
    ("app.moe_training_endpoints", "MoE Training endpoints", None),
]
for _mod, _label, _prefix in _optional_routers:
    try:
        _router = __import__(_mod, fromlist=["router"]).router
        app.include_router(_router) if _prefix is None else app.include_router(_router, prefix=_prefix)
        logger.info(f"{_label} loaded successfully")
    except ImportError as e:
        logger.warning(f"{_label} not available: {e}")

# --- Core routers (always available) ---
from app.routers.finetuning import router as finetuning_router  # noqa: E402
from app.routers.upload import router as upload_router  # noqa: E402
from app.routers.pages import router as pages_router  # noqa: E402
from app.routers.generation import router as generation_router  # noqa: E402
from app.routers.rag import router as rag_router  # noqa: E402
from app.routers.moe_dataset import router as moe_dataset_router  # noqa: E402
from app.routers.monitoring import router as monitoring_router  # noqa: E402
from app.routers.models import router as models_router  # noqa: E402
from app.routers.rlanything import router as rlanything_router  # noqa: E402

app.include_router(finetuning_router)
app.include_router(upload_router)
app.include_router(pages_router)
app.include_router(generation_router)
app.include_router(rag_router)
app.include_router(moe_dataset_router)
app.include_router(monitoring_router)
app.include_router(models_router)
app.include_router(rlanything_router)

# --- Continual learning & DPO (optional) ---
CONTINUAL_TASKS_FILE = Path(os.getcwd()) / "data" / "continual_learning" / "tasks_state.json"
task_manager = None
websocket_endpoint = None

try:
    from app.continual_learning.continual_learning_ui import (
        create_continual_learning_router,
        websocket_endpoint as _ws_ep,
        task_manager as continual_task_manager,
    )

    task_manager = continual_task_manager
    _deps.task_manager = task_manager
    websocket_endpoint = _ws_ep
    app.include_router(create_continual_learning_router(), prefix="/api/continual-learning")

    from app.dpo.preference_ui import router as dpo_router  # noqa: E402
    from app.dpo.training_api import router as dpo_training_router  # noqa: E402

    app.include_router(dpo_router)
    app.include_router(dpo_training_router)
    logger.info("継続学習モジュールとDPO APIを正常にロードしました")
except Exception as e:
    logger.warning(f"継続学習モジュールのロードをスキップ: {str(e)}")

# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


def _load_continual_tasks() -> None:
    """保存された継続学習タスクを読み込み、タスクマネージャーに復元する"""
    if task_manager is None:
        logger.info("継続学習モジュールが無効なため、タスク読み込みをスキップします")
        return

    try:
        if CONTINUAL_TASKS_FILE.exists():
            with open(CONTINUAL_TASKS_FILE, "r", encoding="utf-8") as f:
                stored_tasks = json.load(f)
            if isinstance(stored_tasks, dict):
                task_manager.tasks = stored_tasks
                logger.info("継続学習タスクを読み込みました: %d件", len(task_manager.tasks))
            else:
                logger.warning("継続学習タスクファイルの形式が不正なため、状態を初期化します")
                task_manager.tasks = {}
        else:
            task_manager.tasks = {}
            logger.info("継続学習タスクファイルが存在しません。新規に初期化します。")
    except Exception as e:
        logger.error(f"継続学習タスク読み込みエラー: {str(e)}")
        task_manager.tasks = {}


def _save_continual_tasks() -> None:
    """現在の継続学習タスクを永続化する"""
    if task_manager is None:
        logger.info("継続学習モジュールが無効なため、タスク保存をスキップします")
        return

    try:
        CONTINUAL_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONTINUAL_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(task_manager.tasks, f, ensure_ascii=False, indent=2)
        logger.info("継続学習タスクを保存しました: %d件", len(task_manager.tasks))
    except Exception as e:
        logger.error(f"継続学習タスク保存エラー: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("AI Fine-tuning Toolkit Web API starting with RAG integration...")

    # 必要なディレクトリを作成
    project_root = Path(os.getcwd())
    for subdir in ("data/uploaded", "outputs", "app/static", "data/continual_learning"):
        (project_root / subdir).mkdir(parents=True, exist_ok=True)

    # RAGシステムの初期化
    if RAG_AVAILABLE:
        try:
            await rag_app.initialize()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    else:
        logger.warning("RAG system will not be available in this session")

    # 継続学習タスクを読み込む
    _load_continual_tasks()


@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("AI Fine-tuning Toolkit Web API shutting down...")
    _save_continual_tasks()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# WebSocket (continual learning progress)
# ---------------------------------------------------------------------------


@app.websocket("/ws/continual-learning")
async def continual_learning_websocket(ws: WebSocket):
    """継続学習の進捗をリアルタイムで配信"""
    if websocket_endpoint is None:
        await ws.close(code=1011)
        return
    try:
        await websocket_endpoint(ws)
    except Exception as e:
        logger.error(f"WebSocketエラー: {str(e)}")
        await ws.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
