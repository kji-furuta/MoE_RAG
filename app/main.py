"""
AI Fine-tuning Toolkit - メインアプリケーション（リファクタリング版）
統合Web API：Fine-tuning、RAG、継続学習、モデル管理
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ルーターのインポート
from app.routers import (
    finetuning_router,
    rag_router,
    continual_router,
    models_router
)
from app.routers.continual import load_continual_tasks, save_continual_tasks
from app.dependencies import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    logger.info("AI Fine-tuning Toolkit Web API starting...")
    
    # 必要なディレクトリを作成
    (DATA_DIR / "uploaded").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "continual_learning").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "rag" / "uploads").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "app" / "static").mkdir(parents=True, exist_ok=True)
    
    # 継続学習タスクを読み込む
    load_continual_tasks()
    
    logger.info("Initialization complete")
    
    yield  # アプリケーション実行
    
    # 終了時の処理
    logger.info("AI Fine-tuning Toolkit Web API shutting down...")
    
    # 継続学習タスクを保存
    save_continual_tasks()
    
    logger.info("Shutdown complete")


# FastAPIアプリケーションの作成
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="統合AI開発プラットフォーム：Fine-tuning、RAG、継続学習",
    version="3.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルのマウント
def find_static_directory():
    """静的ファイルディレクトリを検索"""
    possible_paths = [
        PROJECT_ROOT / "app" / "static",
        PROJECT_ROOT / "static",
        Path("/workspace/app/static"),
        Path("/workspace/static")
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"静的ファイルディレクトリを発見: {path}")
            return path
    
    # デフォルトパスを作成
    default_path = PROJECT_ROOT / "app" / "static"
    default_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"静的ファイルディレクトリを作成: {default_path}")
    return default_path


static_dir = find_static_directory()
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# テンプレート設定
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# ルーターの登録
app.include_router(finetuning_router)
app.include_router(rag_router)
app.include_router(continual_router)
app.include_router(models_router)


# ============================================
# Webページルート
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """ホームページ（総合システム）"""
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/finetune", response_class=HTMLResponse)
async def finetune_page(request: Request):
    """ファインチューニングページ"""
    return templates.TemplateResponse("finetune.html", {"request": request})


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """モデル管理ページ"""
    return templates.TemplateResponse("models.html", {"request": request})


@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    """RAGシステムページ"""
    return templates.TemplateResponse("rag.html", {"request": request})


@app.get("/manual", response_class=HTMLResponse)
async def manual_page(request: Request):
    """マニュアルページ"""
    manual_path = PROJECT_ROOT / "static" / "manual.html"
    if manual_path.exists():
        return FileResponse(str(manual_path))
    else:
        return templates.TemplateResponse("manual.html", {"request": request})


@app.get("/system-overview", response_class=HTMLResponse)
async def system_overview_page(request: Request):
    """システム概要ページ"""
    overview_path = PROJECT_ROOT / "static" / "system_overview.html"
    if overview_path.exists():
        return FileResponse(str(overview_path))
    else:
        return templates.TemplateResponse("system_overview.html", {"request": request})


@app.get("/documentation/{doc_type}")
async def serve_documentation(doc_type: str):
    """ドキュメントの提供"""
    doc_files = {
        "readme": "README.md",
        "architecture": "docs/architecture.md",
        "api": "docs/api_reference.md",
        "tutorial": "docs/tutorial.md"
    }
    
    if doc_type in doc_files:
        file_path = PROJECT_ROOT / doc_files[doc_type]
        if file_path.exists():
            # Markdownファイルをテキストとして返す
            return FileResponse(
                str(file_path),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
    
    return {"error": "Documentation not found"}


# ============================================
# WebSocketエンドポイント
# ============================================

@app.websocket("/ws/continual-learning/{task_id}")
async def continual_learning_websocket(websocket: WebSocket, task_id: str):
    """継続学習の進捗をWebSocketで配信"""
    await websocket.accept()
    
    try:
        while True:
            # タスクの状態を取得
            from app.dependencies import continual_tasks
            
            if task_id in continual_tasks:
                task = continual_tasks[task_id]
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.get("status"),
                    "progress": task.get("progress", 0),
                    "message": task.get("message", ""),
                    "current_epoch": task.get("current_epoch"),
                    "total_epochs": task.get("total_epochs")
                })
                
                # タスクが完了または失敗したら終了
                if task.get("status") in ["completed", "failed"]:
                    await websocket.send_json({
                        "task_id": task_id,
                        "status": task.get("status"),
                        "message": "タスクが終了しました",
                        "final": True
                    })
                    break
            else:
                await websocket.send_json({
                    "error": f"Task {task_id} not found"
                })
                break
            
            # 1秒待機
            import asyncio
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocketエラー: {str(e)}")
        await websocket.close()


# ============================================
# ヘルスチェック
# ============================================

@app.get("/health")
async def health_check():
    """システムヘルスチェック"""
    return {
        "status": "healthy",
        "service": "AI Fine-tuning Toolkit",
        "version": "3.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        log_level="info"
    )