#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API
FastAPIベースのWebインターフェース
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import asyncio
import uuid
from pathlib import Path
import logging

# カスタムモジュールのインポート
import sys
sys.path.append('/workspace/src')

from models.japanese_model import JapaneseModel
from training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from training.training_utils import TrainingConfig

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
available_models = []

# ユーティリティ関数
def load_available_models():
    """利用可能なモデル一覧を読み込み"""
    global available_models
    
    # JapaneseModelクラスから対応モデルを取得
    from models.japanese_model import JapaneseModel
    
    # モデルサイズ別カテゴリ
    models_by_size = JapaneseModel.list_models_by_size()
    
    available_models = []
    
    # コンパクトモデル（1-3B）
    available_models.append({
        "name": "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        "description": "日本語StableLM 3B Instruct - 軽量で高速",
        "size": "3B",
        "status": "available",
        "recommended": True
    })
    
    # 小～中規模モデル（3-7B）
    available_models.append({
        "name": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        "description": "ELYZA日本語Llama-2 7B - バランス型",
        "size": "7B",
        "status": "available"
    })
    available_models.append({
        "name": "rinna/youri-7b-chat",
        "description": "Rinna Youri 7B Chat - 対話特化",
        "size": "7B",
        "status": "available"
    })
    
    # 中規模モデル（8-10B）
    available_models.append({
        "name": "elyza/Llama-3-ELYZA-JP-8B",
        "description": "Llama-3 ELYZA日本語 8B - 高性能",
        "size": "8B",
        "status": "available"
    })
    
    # 大規模モデル（17B）
    available_models.append({
        "name": "tokyotech-llm/Swallow-13b-instruct-hf",
        "description": "Swallow 13B Instruct - 東工大開発",
        "size": "13B",
        "status": "requires_auth"
    })
    available_models.append({
        "name": "Qwen/Qwen2.5-17B-Instruct",
        "description": "Qwen 2.5 17B - 多言語対応（日本語可）",
        "size": "17B",
        "status": "available",
        "gpu_required": "A100 40GB+"
    })
    
    # 超大規模モデル（32B）
    available_models.append({
        "name": "cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B",
        "description": "CyberAgent DeepSeek-R1 32B - 最高性能日本語モデル",
        "size": "32B",
        "status": "available",
        "gpu_required": "A100 80GB",
        "warning": "4bit量子化推奨"
    })
    available_models.append({
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 2.5 32B - 最新の超大規模モデル",
        "size": "32B",
        "status": "available",
        "gpu_required": "A100 80GB",
        "warning": "4bit量子化推奨"
    })
    
    # テスト用軽量モデル
    available_models.append({
        "name": "distilgpt2",
        "description": "軽量な英語モデル（テスト用）",
        "size": "82MB",
        "status": "available",
        "test_only": True
    })

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
    
    # outputsディレクトリも検索
    outputs_path = workspace_path / "outputs"
    if outputs_path.exists():
        for model_dir in outputs_path.glob("*_lora_*"):
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
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """ファインチューニングを開始"""
    task_id = str(uuid.uuid4())
    
    # タスクを背景で実行
    background_tasks.add_task(run_training_task, task_id, request)
    
    training_tasks[task_id] = TrainingStatus(
        task_id=task_id,
        status="starting",
        progress=0.0,
        message="ファインチューニングを開始しています..."
    )
    
    return {"task_id": task_id, "status": "started"}

async def run_training_task(task_id: str, request: TrainingRequest):
    """バックグラウンドでトレーニングを実行"""
    try:
        # ステータス更新
        training_tasks[task_id].status = "preparing"
        training_tasks[task_id].message = "モデルを準備中..."
        training_tasks[task_id].progress = 10.0
        
        # モデル初期化
        model = JapaneseModel(model_name=request.model_name)
        
        training_tasks[task_id].message = "LoRA設定を適用中..."
        training_tasks[task_id].progress = 20.0
        
        # LoRA設定
        lora_config = LoRAConfig(**request.lora_config)
        training_config = TrainingConfig(**request.training_config)
        
        training_tasks[task_id].message = "トレーニング開始..."
        training_tasks[task_id].progress = 30.0
        training_tasks[task_id].status = "training"
        
        # トレーナー初期化
        trainer = LoRAFinetuningTrainer(
            model=model,
            lora_config=lora_config,
            training_config=training_config
        )
        
        # トレーニング実行
        trained_model = trainer.train(train_texts=request.training_data)
        
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].progress = 100.0
        training_tasks[task_id].message = "ファインチューニング完了"
        training_tasks[task_id].model_path = str(training_config.output_dir)
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        training_tasks[task_id].status = "failed"
        training_tasks[task_id].message = f"エラー: {str(e)}"

@app.get("/api/training-status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_tasks[task_id]

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    """テキスト生成"""
    try:
        # モデル読み込み（簡単な実装）
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # ベースモデル名を取得（実際の実装では設定から取得）
        base_model_name = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, request.model_path)
        
        # テキスト生成
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=request.max_length,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": request.prompt,
            "generated_text": result,
            "model_path": request.model_path
        }
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-info")
async def get_system_info():
    """システム情報を取得"""
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

# アプリケーション起動時の初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("AI Fine-tuning Toolkit Web API starting...")
    load_available_models()
    
    # 必要なディレクトリを作成
    Path("/workspace/data/uploaded").mkdir(parents=True, exist_ok=True)
    Path("/workspace/app/static").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)