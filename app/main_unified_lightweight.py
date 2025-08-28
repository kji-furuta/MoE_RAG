#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Lightweight Docker Version
RAGシステムの自動初期化を無効化し、オンデマンドで読み込む
"""

# Docker環境用メモリ管理パッチを最初に適用
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RAGシステムの自動初期化を無効化
os.environ["RAG_DISABLE_AUTO_INIT"] = "true"
os.environ["RAG_LAZY_LOAD"] = "true"

try:
    from src.core.docker_memory_patch import patch_memory_manager_for_docker
    patch_memory_manager_for_docker()
except Exception as e:
    print(f"Warning: Docker memory patch could not be applied: {e}")

# PyTorchメモリ管理の最適化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import uuid
import logging
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import yaml
from datetime import datetime, timezone, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
import io
import random

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Docker環境チェック
IS_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"
if IS_DOCKER:
    logger.info("Running in Docker environment - Lightweight mode enabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Import model utilities
try:
    from app.model_utils import (
        load_model_and_tokenizer,
        handle_model_loading_error,
        get_output_directory,
        load_training_config,
        create_quantization_config,
        load_tokenizer,
        get_device_map
    )
    logger.info("Model utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Model utilities import error: {e}")
    def load_model_and_tokenizer(*args, **kwargs):
        raise NotImplementedError("Model utilities not available")

# RAGシステムの遅延初期化用変数
rag_engine = None
RAG_AVAILABLE = False

def initialize_rag_system():
    """RAGシステムをオンデマンドで初期化"""
    global rag_engine, RAG_AVAILABLE
    
    if rag_engine is not None:
        return True
    
    try:
        logger.info("Initializing RAG system on demand...")
        
        # 軽量な埋め込みモデルに変更
        os.environ["RAG_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
        os.environ["RAG_EMBEDDING_DIMENSION"] = "384"
        os.environ["RAG_USE_CPU"] = "true"  # CPUで埋め込みを実行
        
        from src.rag.core.query_engine import RoadDesignQueryEngine
        from src.rag.indexing.metadata_manager import MetadataManager
        
        # 軽量設定でRAGエンジンを初期化
        rag_engine = RoadDesignQueryEngine(
            model_name=None,  # LLMモデルは使用しない
            use_ollama=False,
            embedding_only=True  # 埋め込みのみモード
        )
        
        RAG_AVAILABLE = True
        logger.info("RAG system initialized successfully (lightweight mode)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        RAG_AVAILABLE = False
        return False

# Prometheusメトリクス
try:
    from app.monitoring import metrics_collector, get_prometheus_metrics
    METRICS_AVAILABLE = True
except:
    METRICS_AVAILABLE = False

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース（Docker版）",
    version="2.0.0-docker"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルのマウント
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).parent.parent / "templates"
if templates_dir.exists():
    app.mount("/templates", StaticFiles(directory=str(templates_dir)), name="templates")

# グローバル変数とパス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADED_DIR = DATA_DIR / "uploaded"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONTINUAL_LEARNING_DIR = DATA_DIR / "continual_learning"
CONTINUAL_TASKS_FILE = CONTINUAL_LEARNING_DIR / "tasks_state.json"

# ディレクトリ作成
for dir_path in [DATA_DIR, UPLOADED_DIR, OUTPUTS_DIR, CONTINUAL_LEARNING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# タスク管理
training_tasks = {}
continual_tasks = {}
model_cache = {}

# ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)

# 各ルーターを登録（エラーを無視）
try:
    from app.routers.models import router as models_router
    app.include_router(models_router)
except Exception as e:
    logger.warning(f"Models router: {e}")

try:
    from app.routers.finetuning import router as finetuning_router
    app.include_router(finetuning_router)
except Exception as e:
    logger.warning(f"Finetuning router: {e}")

try:
    from app.routers.continual import router as continual_router
    app.include_router(continual_router)
except Exception as e:
    logger.warning(f"Continual router: {e}")

# RAGルーター（遅延読み込み）
@app.post("/api/rag/search")
async def rag_search(query: dict):
    """RAG検索エンドポイント（オンデマンド初期化）"""
    if not initialize_rag_system():
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    # RAG検索を実行
    try:
        results = await rag_engine.search(query.get("query", ""))
        return results
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 利用可能なモデル
available_models = [
    "cyberagent/open-calm-small",
    "cyberagent/open-calm-3b",
    "rinna/japanese-gpt-1b"
]

# Ollama check
OLLAMA_AVAILABLE = False
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=1)
    if response.status_code == 200:
        OLLAMA_AVAILABLE = True
except:
    pass

# Models
from app.models.training import TrainingRequest, TrainingStatus, GenerationRequest

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(JST).isoformat(),
        "docker": IS_DOCKER,
        "gpu_available": torch.cuda.is_available(),
        "rag_available": RAG_AVAILABLE,
        "ollama_available": OLLAMA_AVAILABLE
    }

@app.get("/")
async def root():
    """ダッシュボード"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Fine-tuning Toolkit (Docker)</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 90%;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            .docker-badge {
                background: #2496ed;
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.8em;
                display: inline-block;
                margin-left: 10px;
            }
            .status {
                margin-top: 30px;
                padding: 20px;
                background: #f7f7f7;
                border-radius: 10px;
            }
            .status-item {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .feature-card {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                text-decoration: none;
                transition: transform 0.3s;
            }
            .feature-card:hover {
                transform: translateY(-5px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>
                🤖 AI Fine-tuning Toolkit
                <span class="docker-badge">🐳 Docker</span>
            </h1>
            <p style="text-align: center; color: #666;">
                Lightweight Mode - Optimized for Docker
            </p>
            
            <div class="features">
                <a href="/docs" class="feature-card">
                    <h3>📚 API Docs</h3>
                    <p>APIドキュメント</p>
                </a>
                <a href="/static/moe_training.html" class="feature-card">
                    <h3>🎯 Fine-tuning</h3>
                    <p>モデル学習</p>
                </a>
                <a href="/static/continual_learning/index.html" class="feature-card">
                    <h3>🔄 Continual</h3>
                    <p>継続学習</p>
                </a>
                <a href="/static/moe_rag_ui.html" class="feature-card">
                    <h3>🔍 RAG</h3>
                    <p>検索システム</p>
                </a>
            </div>
            
            <div class="status">
                <h2>System Status</h2>
                <div class="status-item">
                    <span>Docker</span>
                    <span>✓ Running</span>
                </div>
                <div class="status-item">
                    <span>GPU</span>
                    <span>""" + ("✓ Available" if torch.cuda.is_available() else "✗ Not Available") + """</span>
                </div>
                <div class="status-item">
                    <span>RAG</span>
                    <span>""" + ("✓ Ready" if RAG_AVAILABLE else "○ On-demand") + """</span>
                </div>
                <div class="status-item">
                    <span>Memory</span>
                    <span>""" + f"{psutil.virtual_memory().percent:.1f}% used" + """</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.on_event("startup")
async def startup_event():
    """起動時処理（軽量化）"""
    logger.info("=" * 60)
    logger.info("AI Fine-tuning Toolkit (Docker Lightweight Mode)")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    
    # メモリ情報
    mem = psutil.virtual_memory()
    logger.info(f"Memory: {mem.total / 1024**3:.1f}GB total, {mem.available / 1024**3:.1f}GB available")
    
    # RAGの自動初期化はスキップ
    logger.info("RAG system: On-demand initialization mode")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """シャットダウン処理"""
    logger.info("Shutting down...")
    executor.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ユーティリティ関数
def get_saved_models():
    """保存済みモデル一覧"""
    models = []
    if OUTPUTS_DIR.exists():
        for model_dir in OUTPUTS_DIR.iterdir():
            if model_dir.is_dir():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                })
    return models

async def run_training_task(task_id: str, request: TrainingRequest):
    """トレーニングタスク（簡易版）"""
    try:
        training_tasks[task_id].status = "running"
        await asyncio.sleep(5)  # Simulate
        training_tasks[task_id].status = "completed"
    except Exception as e:
        training_tasks[task_id].status = "failed"
        training_tasks[task_id].message = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
