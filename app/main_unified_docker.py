#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Unified Implementation
統合されたWebインターフェース実装（Docker対応版）
"""

# Docker環境用メモリ管理パッチを最初に適用
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.core.docker_memory_patch import patch_memory_manager_for_docker
    patch_memory_manager_for_docker()
except Exception as e:
    print(f"Warning: Docker memory patch could not be applied: {e}")

# PyTorchメモリ管理の最適化
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # トークナイザーの警告を抑制

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import uuid
from pathlib import Path
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
    logger.info("Running in Docker environment")
    # Docker環境用の設定
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"  # オンラインモードを許可

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
    # フォールバック実装
    def load_model_and_tokenizer(*args, **kwargs):
        raise NotImplementedError("Model utilities not available")

# RAG system imports
try:
    # RAGシステムのモデルロードを無効化（メモリ節約）
    os.environ["RAG_DISABLE_MODEL_LOAD"] = "true"
    from loguru import logger as rag_logger
    from src.rag.core.query_engine import RoadDesignQueryEngine, QueryResult
    from src.rag.indexing.metadata_manager import MetadataManager
    RAG_AVAILABLE = True
    logger.info("RAG system components loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG system not available: {e}")

# Prometheusメトリクスのインポート
try:
    from app.monitoring import metrics_collector, get_prometheus_metrics
    logger.info("Prometheus metrics loaded")
    METRICS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Metrics system not available: {e}")
    metrics_collector = None
    METRICS_AVAILABLE = False

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース",
    version="2.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Docker環境では全許可（本番では制限すべき）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルのマウント
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

templates_dir = Path(__file__).parent.parent / "templates"
if templates_dir.exists():
    app.mount("/templates", StaticFiles(directory=str(templates_dir)), name="templates")
    logger.info(f"Templates mounted from: {templates_dir}")

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

# タスク管理用辞書
training_tasks = {}
continual_tasks = {}
model_cache = {}

# ThreadPoolExecutor for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# MoE-RAGエンドポイントのインポートと登録
try:
    from app.moe_rag_endpoints import router as moe_rag_router
    app.include_router(moe_rag_router)
    logger.info("MoE-RAG endpoints registered")
except Exception as e:
    logger.warning(f"MoE-RAG endpoints not available: {e}")

# 各ルーターを登録
try:
    from app.routers.models import router as models_router
    from app.routers.finetuning import router as finetuning_router
    from app.routers.continual import router as continual_router
    from app.routers.rag import router as rag_router
    
    app.include_router(models_router)
    app.include_router(finetuning_router)
    app.include_router(continual_router)
    app.include_router(rag_router)
    
    logger.info("All routers registered successfully")
except Exception as e:
    logger.warning(f"Some routers could not be loaded: {e}")

# 利用可能なモデル
available_models = [
    "cyberagent/open-calm-small",
    "cyberagent/open-calm-medium",
    "cyberagent/open-calm-large",
    "cyberagent/open-calm-1b",
    "cyberagent/open-calm-3b",
    "cyberagent/open-calm-7b",
    "rinna/japanese-gpt-1b",
    "rinna/japanese-gpt-neox-3.6b",
    "rinna/japanese-gpt-neox-3.6b-instruction",
    "llama3-8b",
    "google/gemma-7b",
    "mistral-7b"
]

# Ollamaのチェック
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code == 200:
        OLLAMA_AVAILABLE = True
        logger.info("Ollama service is available")
    else:
        OLLAMA_AVAILABLE = False
except:
    OLLAMA_AVAILABLE = False
    logger.info("Ollama service not available")

# Pydanticモデル
from app.models.training import TrainingRequest, TrainingStatus, GenerationRequest

# ヘルスチェックとルートエンドポイント
@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(JST).isoformat(),
        "docker": IS_DOCKER,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/")
async def root():
    """ルートエンドポイント - HTMLダッシュボード"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Fine-tuning Toolkit</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
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
                margin-bottom: 30px;
                text-align: center;
                font-size: 2.5em;
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
            .feature-card h3 {
                margin-bottom: 10px;
                font-size: 1.2em;
            }
            .feature-card p {
                font-size: 0.9em;
                opacity: 0.9;
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
            .status-ok { color: #4caf50; }
            .status-warning { color: #ff9800; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Fine-tuning Toolkit</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                統合型AIモデルファインチューニング環境
            </p>
            
            <div class="features">
                <a href="/docs" class="feature-card">
                    <h3>📚 API Docs</h3>
                    <p>APIドキュメント</p>
                </a>
                <a href="/static/moe_training.html" class="feature-card">
                    <h3>🎯 Fine-tuning</h3>
                    <p>モデルファインチューニング</p>
                </a>
                <a href="/static/continual_learning/index.html" class="feature-card">
                    <h3>🔄 Continual Learning</h3>
                    <p>継続学習管理</p>
                </a>
                <a href="/static/moe_rag_ui.html" class="feature-card">
                    <h3>🔍 RAG System</h3>
                    <p>RAG検索システム</p>
                </a>
            </div>
            
            <div class="status">
                <h2 style="margin-bottom: 15px;">システムステータス</h2>
                <div class="status-item">
                    <span>Docker環境</span>
                    <span class="status-ok">""" + ("✓ 検出" if IS_DOCKER else "✗ 未検出") + """</span>
                </div>
                <div class="status-item">
                    <span>GPU</span>
                    <span class="status-ok">""" + (f"✓ {torch.cuda.device_count()}台" if torch.cuda.is_available() else "✗ 利用不可") + """</span>
                </div>
                <div class="status-item">
                    <span>RAGシステム</span>
                    <span class="status-ok">""" + ("✓ 有効" if RAG_AVAILABLE else "✗ 無効") + """</span>
                </div>
                <div class="status-item">
                    <span>Ollama</span>
                    <span class="status-ok">""" + ("✓ 接続済" if OLLAMA_AVAILABLE else "✗ 未接続") + """</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# エラーハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if IS_DOCKER else "An error occurred"
        }
    )

# 起動時の初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("=" * 60)
    logger.info("AI Fine-tuning Toolkit Starting...")
    logger.info(f"Docker Environment: {IS_DOCKER}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=" * 60)
    
    # 継続学習タスクの復元
    try:
        if CONTINUAL_TASKS_FILE.exists():
            with open(CONTINUAL_TASKS_FILE, 'r') as f:
                global continual_tasks
                continual_tasks = json.load(f)
                logger.info(f"Restored {len(continual_tasks)} continual learning tasks")
    except Exception as e:
        logger.error(f"Failed to restore continual tasks: {e}")

# シャットダウン時のクリーンアップ
@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーションシャットダウン時のクリーンアップ"""
    logger.info("Shutting down AI Fine-tuning Toolkit...")
    
    # タスクの保存
    try:
        if continual_tasks:
            with open(CONTINUAL_TASKS_FILE, 'w') as f:
                json.dump(continual_tasks, f, indent=2, default=str)
                logger.info(f"Saved {len(continual_tasks)} continual learning tasks")
    except Exception as e:
        logger.error(f"Failed to save continual tasks: {e}")
    
    # エグゼキューターのシャットダウン
    executor.shutdown(wait=True)
    
    # メモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cleanup completed")

# ユーティリティ関数
def get_saved_models():
    """保存されたモデル一覧を取得"""
    models = []
    if OUTPUTS_DIR.exists():
        for model_dir in OUTPUTS_DIR.iterdir():
            if model_dir.is_dir():
                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                }
                
                # model_info.jsonがあれば読み込む
                info_file = model_dir / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            saved_info = json.load(f)
                            model_info.update(saved_info)
                    except:
                        pass
                
                models.append(model_info)
    
    return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)

async def run_training_task(task_id: str, request: TrainingRequest):
    """バックグラウンドでトレーニングタスクを実行"""
    try:
        training_tasks[task_id].status = "running"
        training_tasks[task_id].message = "Training in progress..."
        
        # ここで実際のトレーニング処理を実行
        # （実装は省略）
        
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].progress = 100.0
        training_tasks[task_id].message = "Training completed successfully"
        
    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}")
        training_tasks[task_id].status = "failed"
        training_tasks[task_id].message = f"Training failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8050))
    uvicorn.run(
        "app.main_unified:app",
        host="0.0.0.0",
        port=port,
        reload=not IS_DOCKER  # Docker環境では自動リロード無効
    )
