#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Improved with Dependency Management
依存関係管理機能を統合した改善版Webインターフェース
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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
from datetime import datetime, timezone, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import sys
import time
from pathlib import Path as PathlibPath

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))

# プロジェクトルートをパスに追加
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    MODEL_UTILS_AVAILABLE = True
except ImportError as e:
    MODEL_UTILS_AVAILABLE = False
    logger.warning(f"Model utilities not available: {e}")

# ===== 新規追加: 依存関係管理のインポート =====
try:
    from src.rag.dependencies.dependency_manager import (
        RAGDependencyManager,
        DependencyLevel,
        DependencyCheckResult
    )
    DEPENDENCY_MANAGER_AVAILABLE = True
    logger.info("Dependency manager loaded successfully")
except ImportError as e:
    DEPENDENCY_MANAGER_AVAILABLE = False
    logger.warning(f"Dependency manager not available: {e}")

# グローバル変数
dependency_manager: Optional[RAGDependencyManager] = None
dependency_check_result: Optional[DependencyCheckResult] = None
rag_system = None  # RAGシステムのインスタンス

# RAG system imports (依存関係チェック後に動的にロード)
RAG_AVAILABLE = False
RoadDesignQueryEngine = None
QueryResult = None
MetadataManager = None


def check_and_load_rag_system():
    """依存関係をチェックしてRAGシステムをロード"""
    global RAG_AVAILABLE, RoadDesignQueryEngine, QueryResult, MetadataManager
    
    if not DEPENDENCY_MANAGER_AVAILABLE:
        logger.warning("Dependency manager not available, skipping RAG system check")
        return False
    
    try:
        # 依存関係マネージャーの初期化
        global dependency_manager, dependency_check_result
        dependency_manager = RAGDependencyManager()
        
        # 依存関係チェック
        logger.info("Checking RAG system dependencies...")
        dependency_check_result = dependency_manager.check_all_dependencies(use_cache=True)
        
        # レポートをログに出力
        logger.info("\n" + dependency_manager.get_dependency_report())
        
        if not dependency_check_result.can_run:
            logger.error("Cannot run RAG system due to missing dependencies:")
            for dep in dependency_check_result.missing_core:
                logger.error(f"  - Missing core: {dep}")
            for dep in dependency_check_result.missing_infrastructure:
                logger.error(f"  - Missing infrastructure: {dep}")
            return False
        
        # 警告があれば表示
        for warning in dependency_check_result.warnings:
            logger.warning(f"Dependency warning: {warning}")
        
        # RAGシステムのインポートを試みる
        from loguru import logger as rag_logger
        from src.rag.core.query_engine import RoadDesignQueryEngine as _RQE, QueryResult as _QR
        from src.rag.indexing.metadata_manager import MetadataManager as _MM
        
        RoadDesignQueryEngine = _RQE
        QueryResult = _QR
        MetadataManager = _MM
        RAG_AVAILABLE = True
        
        logger.success("RAG system loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load RAG system: {e}")
        if os.environ.get("DEBUG"):
            logger.exception("Detailed error:")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時
    logger.info("Starting application...")
    
    # 依存関係チェックとRAGシステムのロード
    rag_loaded = check_and_load_rag_system()
    
    if rag_loaded:
        logger.info("RAG system is available")
    else:
        logger.warning("RAG system is not available - running in degraded mode")
    
    # その他の初期化処理
    logger.info("Application startup complete")
    
    yield
    
    # シャットダウン時
    logger.info("Shutting down application...")
    
    # クリーンアップ処理
    if rag_system:
        logger.info("Cleaning up RAG system...")
        # RAGシステムのクリーンアップ
    
    logger.info("Application shutdown complete")


# FastAPIアプリケーション初期化（lifespan追加）
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース（依存関係管理機能付き）",
    version="2.1.0",
    lifespan=lifespan
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
    """静的ファイルディレクトリを検索"""
    current_file = Path(__file__)
    static_path = current_file.parent / "static"
    
    if static_path.exists() and static_path.is_dir():
        return str(static_path)
    
    project_root = Path(os.getcwd())
    possible_paths = [
        project_root / "static",
        project_root / "app" / "static",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return str(path)
    
    return str(project_root / "static")

static_dir = find_static_directory()
logger.info(f"Using static directory: {static_dir}")

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ===== データモデル定義 =====
class DependencyStatus(BaseModel):
    """依存関係ステータス"""
    available: bool
    can_run: bool
    missing_core: List[str]
    missing_infrastructure: List[str]
    missing_optional: List[str]
    alternatives_used: Dict[str, str]
    warnings: List[str]
    report: str


class SystemHealth(BaseModel):
    """システムヘルス情報"""
    status: str  # healthy, degraded, unhealthy
    components: Dict[str, Dict[str, Any]]
    dependencies: Optional[DependencyStatus]
    timestamp: str


# ===== 新規追加: 依存関係管理エンドポイント =====

@app.get("/api/dependencies/check", response_model=DependencyStatus)
async def check_dependencies():
    """依存関係をチェック"""
    if not DEPENDENCY_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency manager is not available"
        )
    
    try:
        if not dependency_manager:
            dependency_manager = RAGDependencyManager()
        
        result = dependency_manager.check_all_dependencies(use_cache=False)
        
        return DependencyStatus(
            available=True,
            can_run=result.can_run,
            missing_core=result.missing_core,
            missing_infrastructure=result.missing_infrastructure,
            missing_optional=result.missing_optional,
            alternatives_used=result.alternatives_used,
            warnings=result.warnings,
            report=dependency_manager.get_dependency_report(format="text")
        )
        
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dependency check failed: {str(e)}"
        )


@app.get("/api/dependencies/report")
async def get_dependency_report(format: str = Query("text", pattern="^(text|json|markdown)$")):
    """依存関係レポートを取得"""
    if not DEPENDENCY_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency manager is not available"
        )
    
    try:
        if not dependency_manager:
            dependency_manager = RAGDependencyManager()
        
        report = dependency_manager.get_dependency_report(format=format)
        
        if format == "json":
            return JSONResponse(content=json.loads(report))
        elif format == "markdown":
            return PlainTextResponse(content=report, media_type="text/markdown")
        else:
            return PlainTextResponse(content=report)
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@app.post("/api/dependencies/install")
async def install_dependencies(
    level: Optional[str] = Query(None, pattern="^(core|infrastructure|optional)$"),
    dry_run: bool = Query(False)
):
    """不足している依存関係をインストール"""
    if not DEPENDENCY_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency manager is not available"
        )
    
    try:
        if not dependency_manager:
            dependency_manager = RAGDependencyManager()
        
        dep_level = None
        if level:
            dep_level = DependencyLevel[level.upper()]
        
        results = dependency_manager.install_missing_dependencies(
            level=dep_level,
            dry_run=dry_run
        )
        
        return {
            "dry_run": dry_run,
            "level": level,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Installation failed: {str(e)}"
        )


# ===== 改善されたヘルスチェックエンドポイント =====

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """システムヘルスチェック（依存関係情報付き）"""
    components = {}
    
    # 基本システム情報
    components["system"] = {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # GPU情報
    if torch.cuda.is_available():
        components["gpu"] = {
            "status": "healthy",
            "available": True,
            "count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        }
        
        # GPU メモリ情報
        for i in range(torch.cuda.device_count()):
            mem_info = torch.cuda.mem_get_info(i)
            components[f"gpu_{i}"] = {
                "status": "healthy",
                "free_memory_gb": mem_info[0] / (1024**3),
                "total_memory_gb": mem_info[1] / (1024**3)
            }
    else:
        components["gpu"] = {
            "status": "unavailable",
            "available": False
        }
    
    # RAGシステム
    components["rag"] = {
        "status": "healthy" if RAG_AVAILABLE else "unavailable",
        "available": RAG_AVAILABLE
    }
    
    # 依存関係情報
    dep_status = None
    if DEPENDENCY_MANAGER_AVAILABLE and dependency_check_result:
        dep_status = DependencyStatus(
            available=True,
            can_run=dependency_check_result.can_run,
            missing_core=dependency_check_result.missing_core,
            missing_infrastructure=dependency_check_result.missing_infrastructure,
            missing_optional=dependency_check_result.missing_optional,
            alternatives_used=dependency_check_result.alternatives_used,
            warnings=dependency_check_result.warnings,
            report=""  # 簡潔にするため省略
        )
    
    # 全体のステータス判定
    overall_status = "healthy"
    if any(c.get("status") == "unhealthy" for c in components.values()):
        overall_status = "unhealthy"
    elif any(c.get("status") == "degraded" for c in components.values()):
        overall_status = "degraded"
    elif not RAG_AVAILABLE:
        overall_status = "degraded"
    
    return SystemHealth(
        status=overall_status,
        components=components,
        dependencies=dep_status,
        timestamp=datetime.now(JST).isoformat()
    )


# ===== RAGエンドポイント（依存関係チェック付き） =====

@app.get("/rag/health")
async def rag_health():
    """RAGシステムのヘルスチェック"""
    if not RAG_AVAILABLE:
        # 依存関係の詳細情報を提供
        detail = "RAG system is not available"
        
        if DEPENDENCY_MANAGER_AVAILABLE and dependency_check_result:
            if dependency_check_result.missing_core:
                detail += f". Missing core dependencies: {', '.join(dependency_check_result.missing_core)}"
            if dependency_check_result.missing_infrastructure:
                detail += f". Missing infrastructure: {', '.join(dependency_check_result.missing_infrastructure)}"
        
        raise HTTPException(
            status_code=503,
            detail=detail
        )
    
    return {
        "status": "healthy",
        "message": "RAG system is operational",
        "timestamp": datetime.now(JST).isoformat()
    }


# 既存のエンドポイントはそのまま保持...
# （以下、必要な既存のエンドポイントを含める）

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_unified_improved:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        log_level="info"
    )
