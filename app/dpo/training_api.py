#!/usr/bin/env python3
"""
DPO Training API Router
DPO学習のためのREST APIエンドポイント
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.training.service import run_training_task, get_training_status
from src.training.dpo_trainer import DPOTrainingConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dpo", tags=["dpo"])


class DPOTrainingRequest(BaseModel):
    """DPO学習リクエスト"""
    model_name: str = Field(..., description="ベースモデル名")
    dataset_path: str = Field(..., description="DPOデータセットパス")
    output_dir: str = Field(default="outputs/dpo_model", description="出力ディレクトリ")

    # 学習パラメータ
    num_train_epochs: int = Field(default=1, description="学習エポック数")
    per_device_train_batch_size: int = Field(default=1, description="バッチサイズ")
    gradient_accumulation_steps: int = Field(default=16, description="勾配累積ステップ (メモリ最適化: 16推奨)")
    learning_rate: float = Field(default=5e-7, description="学習率")
    max_prompt_length: int = Field(default=512, description="最大プロンプト長 (メモリ最適化: 512推奨)")
    max_length: int = Field(default=1024, description="最大シーケンス長 (メモリ最適化: 1024推奨)")

    # LoRAパラメータ
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")

    # メモリ設定
    max_memory_per_gpu: str = Field(default="20GiB", description="GPU毎の最大メモリ")
    max_cpu_memory: str = Field(default="30GiB", description="CPU最大メモリ")

    # DPO固有パラメータ
    beta: float = Field(default=0.1, description="DPO beta parameter")


class DPOTrainingResponse(BaseModel):
    """DPO学習レスポンス"""
    task_id: str
    status: str
    message: str


class DPOStatsResponse(BaseModel):
    """DPO統計レスポンス"""
    total_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int


@router.post("/train", response_model=DPOTrainingResponse)
async def start_dpo_training(
    request: DPOTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    DPO学習を開始
    """
    try:
        # DPO設定を作成
        config = DPOTrainingConfig(
            model_name=request.model_name,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir,
            num_train_epochs=request.num_train_epochs,
            per_device_train_batch_size=request.per_device_train_batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            learning_rate=request.learning_rate,
            max_prompt_length=request.max_prompt_length,  # Added for memory optimization
            max_length=request.max_length,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            max_memory_per_gpu=request.max_memory_per_gpu,
            max_cpu_memory=request.max_cpu_memory,
            beta=request.beta,
        )

        # タスクIDを生成
        import uuid
        task_id = str(uuid.uuid4())

        # バックグラウンドタスクとして学習を実行
        background_tasks.add_task(
            run_training_task,
            task_id=task_id,
            training_type="dpo",
            config=config
        )

        logger.info(f"DPO学習タスク開始: {task_id}")

        return DPOTrainingResponse(
            task_id=task_id,
            status="started",
            message="DPO training task started successfully"
        )

    except Exception as e:
        logger.error(f"DPO学習開始エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_dpo_training_status(task_id: str):
    """
    DPO学習タスクのステータスを取得
    """
    try:
        status = get_training_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ステータス取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DPOStatsResponse)
async def get_dpo_stats():
    """
    DPO学習タスクの統計を取得
    """
    try:
        from app.training.service import training_tasks

        total = len(training_tasks)
        running = sum(1 for t in training_tasks.values() if t["status"] == "running")
        completed = sum(1 for t in training_tasks.values() if t["status"] == "completed")
        failed = sum(1 for t in training_tasks.values() if t["status"] == "failed")

        return DPOStatsResponse(
            total_tasks=total,
            running_tasks=running,
            completed_tasks=completed,
            failed_tasks=failed
        )
    except Exception as e:
        logger.error(f"統計取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
