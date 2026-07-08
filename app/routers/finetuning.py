"""Fine-tuning training API router."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..dependencies import PROJECT_ROOT, logger, training_tasks, available_models, get_saved_models
from ..training.models import TrainingRequest

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/api", tags=["finetuning"])


@router.get("/models")
async def get_models():
    """利用可能なモデル一覧を取得"""
    available = available_models

    saved_models = []
    try:
        saved_models = get_saved_models()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"保存済みモデルの取得に失敗: {exc}")

    return {
        "available_models": available,
        "saved_models": saved_models
    }


@router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """ファインチューニングを開始"""
    try:
        from ..training.service import create_training_task, run_training_task  # local import to avoid optional deps at module load

        task_id = create_training_task(request)

        # バックグラウンドでトレーニングを実行
        background_tasks.add_task(run_training_task, task_id, request)
        
        return {"task_id": task_id, "status": "started"}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
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
