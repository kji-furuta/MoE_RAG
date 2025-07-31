#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Lightweight Implementation
軽量版Webインターフェース実装（RAGシステムの遅延読み込み）
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import uuid
from pathlib import Path
import logging
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import yaml
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
from typing import Optional
import sys
import time
from pathlib import Path as PathlibPath

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG system imports (遅延読み込み)
RAG_AVAILABLE = False
rag_app = None

def initialize_rag_system():
    """RAGシステムを遅延初期化"""
    global RAG_AVAILABLE, rag_app
    try:
        sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
        from loguru import logger as rag_logger
        from src.rag.core.query_engine import RoadDesignQueryEngine, QueryResult
        from src.rag.indexing.metadata_manager import MetadataManager
        
        class RAGApplication:
            def __init__(self):
                self.query_engine: Optional[RoadDesignQueryEngine] = None
                self.metadata_manager: Optional[MetadataManager] = None
                self.is_initialized = False
                self.initialization_error = None
                
            async def initialize(self):
                try:
                    logger.info("Initializing RAG system...")
                    self.query_engine = RoadDesignQueryEngine()
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.query_engine.initialize
                    )
                    self.metadata_manager = MetadataManager()
                    self.is_initialized = True
                    logger.info("RAG system initialized successfully")
                except Exception as e:
                    self.initialization_error = str(e)
                    logger.error(f"Failed to initialize RAG system: {e}")
                    
            def check_initialized(self):
                if not self.is_initialized:
                    if self.initialization_error:
                        raise HTTPException(
                            status_code=500,
                            detail=f"RAG system initialization failed: {self.initialization_error}"
                        )
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="RAG system is not yet initialized"
                        )
        
        rag_app = RAGApplication()
        RAG_AVAILABLE = True
        logger.info("RAG system components loaded successfully")
    except ImportError as e:
        RAG_AVAILABLE = False
        logger.warning(f"RAG system not available: {e}")

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース（軽量版）",
    version="2.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 静的ファイルディレクトリの動的検出
def find_static_directory():
    current_file = Path(__file__)
    static_path = current_file.parent / "static"
    
    if static_path.exists() and static_path.is_dir():
        return str(static_path)
    
    project_root = Path(os.getcwd())
    possible_paths = [
        project_root / "static",
        project_root / "app" / "static",
        project_root / "static"
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return str(path)
    
    return str(project_root / "static")

static_dir = find_static_directory()
print(f"Using static directory: {static_dir}")

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# データモデル定義
class ModelInfo(BaseModel):
    name: str
    description: str
    size: str
    status: str

class TrainingRequest(BaseModel):
    model_name: str
    training_data: List[str]
    training_method: str = "lora"
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]

class GenerationRequest(BaseModel):
    model_path: str
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    model_path: Optional[str] = None

# グローバル変数
training_tasks = {}
available_models = []

@app.get("/")
async def root(request: Request):
    """メインページ"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Fine-tuning Toolkit</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .logo { max-width: 300px; height: auto; }
            .nav { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
            .nav a { padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
            .nav a:hover { background: #0056b3; }
            .status { background: #e9ecef; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .feature { margin-bottom: 20px; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/static/logo_teikoku.png" alt="Logo" class="logo">
                <h1>AI Fine-tuning Toolkit</h1>
                <p>日本語LLMファインチューニング + RAGシステム統合Webツールキット</p>
            </div>
            
            <div class="nav">
                <a href="/finetune">ファインチューニング</a>
                <a href="/rag">RAGシステム</a>
                <a href="/models">モデル管理</a>
                <a href="/manual">マニュアル</a>
            </div>
            
            <div class="status">
                <h3>システム状況</h3>
                <p>✅ Webサーバー: 稼働中</p>
                <p>✅ GPU: 利用可能</p>
                <p>⏳ RAGシステム: 初期化中（初回アクセス時に自動開始）</p>
            </div>
            
            <div class="feature">
                <h3>🚀 主要機能</h3>
                <ul>
                    <li><strong>ファインチューニング</strong>: LoRA、QLoRA、フルファインチューニング</li>
                    <li><strong>RAGシステム</strong>: 土木道路設計特化型検索・質問応答</li>
                    <li><strong>モデル管理</strong>: 学習済みモデルの管理・変換</li>
                    <li><strong>リアルタイム監視</strong>: 学習進捗とGPU使用状況</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📚 ドキュメント</h3>
                <ul>
                    <li><a href="/manual">利用マニュアル</a> - 詳細な使用方法</li>
                    <li><a href="/system-overview">システム概要</a> - 技術仕様</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/finetune")
async def finetune_page(request: Request):
    """ファインチューニングページ"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ファインチューニング - AI Fine-tuning Toolkit</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .logo { max-width: 200px; height: auto; }
            .nav { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
            .nav a { padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
            .nav a:hover { background: #0056b3; }
            .content { text-align: center; padding: 40px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/static/logo_teikoku.png" alt="Logo" class="logo">
                <h1>ファインチューニング</h1>
            </div>
            
            <div class="nav">
                <a href="/">ホーム</a>
                <a href="/finetune">ファインチューニング</a>
                <a href="/rag">RAGシステム</a>
                <a href="/models">モデル管理</a>
            </div>
            
            <div class="content">
                <h2>🚧 工事中</h2>
                <p>ファインチューニング機能は現在準備中です。</p>
                <p>完全版のWebインターフェースをご利用ください。</p>
                <p><a href="/">ホームに戻る</a></p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/rag")
async def rag_page(request: Request):
    """RAGシステムページ"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGシステム - AI Fine-tuning Toolkit</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .logo { max-width: 200px; height: auto; }
            .nav { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
            .nav a { padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
            .nav a:hover { background: #0056b3; }
            .content { text-align: center; padding: 40px; }
            .status { background: #e9ecef; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/static/logo_teikoku.png" alt="Logo" class="logo">
                <h1>RAGシステム</h1>
                <p>土木道路設計特化型検索・質問応答システム</p>
            </div>
            
            <div class="nav">
                <a href="/">ホーム</a>
                <a href="/finetune">ファインチューニング</a>
                <a href="/rag">RAGシステム</a>
                <a href="/models">モデル管理</a>
            </div>
            
            <div class="content">
                <div class="status">
                    <h3>⏳ RAGシステム初期化中</h3>
                    <p>初回アクセス時にRAGシステムの初期化が開始されます。</p>
                    <p>初期化には数分かかる場合があります。</p>
                    <p><button onclick="location.reload()">状態を更新</button></p>
                </div>
                
                <h2>🚧 工事中</h2>
                <p>RAGシステム機能は現在準備中です。</p>
                <p>完全版のWebインターフェースをご利用ください。</p>
                <p><a href="/">ホームに戻る</a></p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/models")
async def models_page(request: Request):
    """モデル管理ページ"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>モデル管理 - AI Fine-tuning Toolkit</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .logo { max-width: 200px; height: auto; }
            .nav { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
            .nav a { padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
            .nav a:hover { background: #0056b3; }
            .content { text-align: center; padding: 40px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/static/logo_teikoku.png" alt="Logo" class="logo">
                <h1>モデル管理</h1>
            </div>
            
            <div class="nav">
                <a href="/">ホーム</a>
                <a href="/finetune">ファインチューニング</a>
                <a href="/rag">RAGシステム</a>
                <a href="/models">モデル管理</a>
            </div>
            
            <div class="content">
                <h2>🚧 工事中</h2>
                <p>モデル管理機能は現在準備中です。</p>
                <p>完全版のWebインターフェースをご利用ください。</p>
                <p><a href="/">ホームに戻る</a></p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/api/system-info")
async def get_system_info():
    """システム情報取得"""
    try:
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i)
                })
        
        return {
            "status": "success",
            "system_info": {
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_info": gpu_info,
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "rag_available": RAG_AVAILABLE,
                "rag_initialized": rag_app.is_initialized if rag_app else False
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/rag/health")
async def rag_health_check():
    """RAGシステムヘルスチェック"""
    if not RAG_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "RAG system not available",
            "timestamp": datetime.now().isoformat()
        }
    
    if not rag_app:
        return {
            "status": "initializing",
            "message": "RAG system is initializing",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "healthy" if rag_app.is_initialized else "initializing",
        "message": "RAG system is ready" if rag_app.is_initialized else "RAG system is initializing",
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    logger.info("Starting AI Fine-tuning Toolkit Web API (Lightweight)...")
    
    # RAGシステムの初期化を開始（バックグラウンド）
    if RAG_AVAILABLE:
        asyncio.create_task(rag_app.initialize())

if __name__ == "__main__":
    import uvicorn
    # RAGシステムの初期化
    initialize_rag_system()
    uvicorn.run(app, host="0.0.0.0", port=8050, reload=True) 