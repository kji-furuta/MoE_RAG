"""
Fine-tuning関連のAPIルーター
"""

import os
import json
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse

from ..models.training import TrainingRequest, TrainingStatus, GenerationRequest
from ..dependencies import (
    logger, PROJECT_ROOT, UPLOADED_DIR, OUTPUTS_DIR, 
    training_tasks, model_cache, executor
)

# モデル関連のインポート（main_unified.pyから移動が必要）
# これらの関数は後でサービス層に移動する予定
from ..main_unified import (
    get_saved_models,
    run_training_task,
    load_tokenizer,
    create_quantization_config,
    get_device_map,
    available_models,
    OLLAMA_AVAILABLE
)

router = APIRouter(prefix="/api", tags=["finetuning"])


@router.get("/models")
async def get_models():
    """利用可能なモデル一覧を取得"""
    return {
        "available_models": available_models,
        "saved_models": get_saved_models()
    }


@router.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """トレーニングデータをアップロード"""
    try:
        logger.info(f"ファイルアップロード開始: {file.filename}")
        
        # ファイル名とサイズの検証
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が不正です")
        
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB制限
            raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます (最大100MB)")
        
        # ファイル保存
        UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOADED_DIR / file.filename
        content = await file.read()
        
        logger.info(f"ファイル保存: {file_path}, サイズ: {len(content)} bytes")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # ファイル形式の検証
        sample_data = []
        data_count = 0
        
        if file.filename.endswith('.jsonl'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data_count = len(lines)
                    
                    for i, line in enumerate(lines[:5]):  # 最初の5行をサンプルとして取得
                        line = line.strip()
                        if line:  # 空行をスキップ
                            try:
                                data = json.loads(line)
                                sample_data.append(data)
                            except json.JSONDecodeError as je:
                                logger.error(f"JSON parse error at line {i+1}: {str(je)}")
                                raise HTTPException(status_code=400, detail=f"行 {i+1} でJSONパースエラー: {str(je)}")
                        
                logger.info(f"JSONL解析完了: {data_count}行, サンプル: {len(sample_data)}件")
                
            except UnicodeDecodeError:
                logger.error("ファイルエンコーディングエラー")
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)")
        
        elif file.filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data_count = len(data)
                        sample_data = data[:3]  # 最初の3件をサンプルとして取得
                    else:
                        data_count = 1
                        sample_data = [data]
                        
                logger.info(f"JSON解析完了: {data_count}件, サンプル: {len(sample_data)}件")
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON parse error: {str(je)}")
                raise HTTPException(status_code=400, detail=f"JSONパースエラー: {str(je)}")
            except UnicodeDecodeError:
                logger.error("ファイルエンコーディングエラー")
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)")
        else:
            raise HTTPException(status_code=400, detail="サポートされていないファイル形式です (.jsonl または .json を使用してください)")
        
        result = {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "data_count": data_count,
            "sample_data": sample_data[:3]
        }
        
        logger.info(f"アップロード成功: {result}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"アップロードエラー: {str(e)}")


@router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """ファインチューニングを開始"""
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"ファインチューニング開始リクエスト受信: task_id={task_id}")
        logger.info(f"リクエスト内容: model_name={request.model_name}, method={request.training_method}")
        logger.info(f"LoRA設定: {request.lora_config}")
        logger.info(f"トレーニング設定: {request.training_config}")
        
        # リクエストの検証
        if not request.model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        if not request.training_data:
            raise HTTPException(status_code=400, detail="training_data is required")
        
        # 初期ステータスを設定
        training_tasks[task_id] = TrainingStatus(
            task_id=task_id,
            status="starting",
            progress=0.0,
            message="ファインチューニングを開始しています..."
        )
        
        # バックグラウンドでトレーニングを実行
        background_tasks.add_task(run_training_task, task_id, request)
        
        return {"task_id": task_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"トレーニング開始エラー: {str(e)}")
        logger.error(f"エラー詳細: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_tasks[task_id]


@router.post("/save-verification")
async def save_verification_results(verification_data: dict):
    """ファインチューニング済みモデルの検証結果を保存"""
    try:
        # 保存ディレクトリの作成
        verification_dir = PROJECT_ROOT / "verification_results"
        verification_dir.mkdir(exist_ok=True)
        
        # ファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = verification_data.get("model_path", "unknown").split("/")[-1]
        filename = f"verification_{model_name}_{timestamp}.json"
        
        # 検証結果を保存
        output_path = verification_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(verification_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"検証結果を保存: {output_path}")
        
        return {
            "status": "success",
            "saved_path": str(output_path),
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"検証結果保存エラー: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }