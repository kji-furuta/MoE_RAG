#!/usr/bin/env python3
"""
Simplified AI Fine-tuning Toolkit Web API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import uuid
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory="/workspace/app/static"), name="static")

# データモデル定義
class ModelInfo(BaseModel):
    name: str
    description: str
    size: str
    status: str

class TrainingRequest(BaseModel):
    model_name: str
    training_data: List[str]
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]

class GenerationRequest(BaseModel):
    model_path: str
    prompt: str
    max_length: int = 100
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
available_models = [
    {
        "name": "distilgpt2",
        "description": "軽量な英語モデル（学習用）",
        "size": "82MB",
        "status": "available"
    },
    {
        "name": "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        "description": "日本語対応3Bモデル",
        "size": "3B",
        "status": "available"
    },
    {
        "name": "calm3-22b",
        "description": "CALM3 22B日本語チャットモデル",
        "size": "22B",
        "status": "available"
    }
]

def get_saved_models():
    """保存済みモデル一覧を取得"""
    saved_models = []
    workspace_path = Path("/workspace")
    
    # LoRAモデルを検索
    for model_dir in workspace_path.glob("lora_demo_*"):
        if model_dir.is_dir():
            saved_models.append({
                "name": model_dir.name,
                "path": str(model_dir),
                "type": "LoRA",
                "size": "~1.6MB"
            })
    
    return saved_models

# API エンドポイント

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """メインページ"""
    return FileResponse('/workspace/app/static/index.html')

@app.get("/api/models")
async def get_models():
    """利用可能なモデル一覧を取得"""
    return {
        "available_models": available_models,
        "saved_models": get_saved_models()
    }

@app.post("/api/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """トレーニングデータをアップロード"""
    try:
        # ファイル保存
        upload_dir = Path("/workspace/data/uploaded")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # ファイル形式の検証
        if file.filename.endswith('.jsonl'):
            # JSONL形式の検証
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sample_data = []
                for i, line in enumerate(lines[:5]):  # 最初の5行をサンプルとして表示
                    try:
                        data = json.loads(line.strip())
                        sample_data.append(data)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail=f"Invalid JSON at line {i+1}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "sample_data": sample_data[:3] if 'sample_data' in locals() else []
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def start_training(request: TrainingRequest):
    """ファインチューニングを開始"""
    task_id = str(uuid.uuid4())
    
    # モデル保存ディレクトリを作成（デモ）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"/workspace/lora_demo_{timestamp}"
    
    # ディレクトリ作成
    Path(model_path).mkdir(exist_ok=True)
    
    # ダミーのアダプター設定ファイルを作成
    adapter_config = {
        "model_type": "lora",
        "base_model": request.model_name,
        "r": request.lora_config.get("r", 16),
        "task_type": "CAUSAL_LM"
    }
    
    with open(Path(model_path) / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f)
    
    # シンプルなモックタスク
    training_tasks[task_id] = TrainingStatus(
        task_id=task_id,
        status="completed",
        progress=100.0,
        message="ファインチューニング完了（デモ）",
        model_path=model_path
    )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/training-status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_tasks[task_id]

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    """テキスト生成（デモ）"""
    try:
        # デモ用の簡単な生成
        demo_responses = {
            "質問: 日本の首都は": "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
            "質問: AI": "質問: AIとは何ですか？\n回答: AI（人工知能）は、人間の知能を機械で再現する技術です。",
            "Hello": "Hello! How can I help you today?",
        }
        
        # プロンプトの一部にマッチする応答を探す
        generated_text = request.prompt
        for key, response in demo_responses.items():
            if key in request.prompt:
                generated_text = response
                break
        else:
            generated_text = request.prompt + " [デモ応答: ファインチューニング済みモデルからの生成結果]"
        
        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "model_path": request.model_path
        }
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-info")
async def get_system_info():
    """システム情報を取得"""
    try:
        import torch
        import psutil
        
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory // 1024**3,
                    "memory_used": torch.cuda.memory_allocated(i) // 1024**3
                })
        
        return {
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_info": gpu_info,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // 1024**3,
            "memory_used": psutil.virtual_memory().used // 1024**3
        }
    except Exception as e:
        return {
            "gpu_count": 0,
            "gpu_info": [],
            "cpu_count": 4,
            "memory_total": 16,
            "memory_used": 8,
            "error": str(e)
        }

# アプリケーション起動時の初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("AI Fine-tuning Toolkit Web API starting...")
    
    # 必要なディレクトリを作成
    Path("/workspace/data/uploaded").mkdir(parents=True, exist_ok=True)
    Path("/workspace/app/static").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)