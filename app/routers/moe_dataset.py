"""MoE dataset management and training endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import psutil
import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import logger

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/api/moe", tags=["moe_dataset"])


# ---------------------------------------------------------------------------
# Helper: access app.state.moe_tasks via Request
# ---------------------------------------------------------------------------

def _get_moe_tasks(request: Request) -> dict:
    """Return the moe_tasks dict from app.state, creating it if needed."""
    if not hasattr(request.app.state, "moe_tasks"):
        request.app.state.moe_tasks = {}
    return request.app.state.moe_tasks


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@router.get("/dataset/stats/{dataset_name}")
async def get_dataset_stats(dataset_name: str):
    """データセットの統計情報を取得"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }

        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail="Dataset not found")

        file_path = Path(dataset_paths[dataset_name])

        if not file_path.exists():
            return {
                "sample_count": 0,
                "expert_distribution": "データなし",
                "last_updated": "未作成",
                "file_size": 0
            }

        # ファイル統計
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        # 最終更新日時をJSTで表示
        last_modified = datetime.fromtimestamp(file_stat.st_mtime, tz=JST).strftime("%Y/%m/%d %H:%M JST")

        # サンプル数とエキスパート分布を計算
        sample_count = 0
        expert_counts: Dict[str, int] = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample_count += 1
                    try:
                        data = json.loads(line)
                        expert = data.get('expert_domain', '不明')
                        expert_counts[expert] = expert_counts.get(expert, 0) + 1
                    except Exception:
                        pass

        # エキスパート分布の文字列化
        expert_distribution = ", ".join([f"{k}: {v}" for k, v in expert_counts.items()])

        return {
            "sample_count": sample_count,
            "expert_distribution": expert_distribution or "不明",
            "last_updated": last_modified,
            "file_size": file_size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/update")
async def update_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """データセットを更新（既存データセットのバックアップ付き）"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }

        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=400, detail="Invalid dataset name")

        file_path = Path(dataset_paths[dataset_name])

        # バックアップの作成
        backup_path = None
        if file_path.exists():
            # バックアップファイル名のタイムスタンプもJSTで統一
            timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{dataset_name}_{timestamp}.jsonl"

            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")

        # ファイル内容の読み取りと検証
        content = await file.read()

        # JSONLファイルの検証
        lines = content.decode('utf-8').strip().split('\n')
        valid_samples = []
        invalid_lines = []

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # 必須フィールドの確認
                required_fields = ['question', 'answer']
                if all(field in data for field in required_fields):
                    valid_samples.append(line)
                else:
                    invalid_lines.append(f"Line {i}: Missing required fields")
            except json.JSONDecodeError as e:
                invalid_lines.append(f"Line {i}: {str(e)}")

        if not valid_samples:
            raise HTTPException(
                status_code=400,
                detail=f"No valid samples found. Errors: {'; '.join(invalid_lines[:5])}"
            )

        # 新しいデータセットを保存
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                f.write(sample + '\n')

        validation_result = "成功" if not invalid_lines else f"警告: {len(invalid_lines)}行スキップ"

        return {
            "status": "success",
            "backup_path": str(backup_path) if backup_path else None,
            "sample_count": len(valid_samples),
            "validation_result": validation_result,
            "invalid_lines": len(invalid_lines),
            "message": f"Dataset {dataset_name} updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/download/{dataset_name}")
async def download_dataset(dataset_name: str):
    """データセットをダウンロード"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }

        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail="Dataset not found")

        file_path = Path(dataset_paths[dataset_name])

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        def iterfile():
            with open(file_path, 'rb') as f:
                yield from f

        filename = f"{dataset_name}_dataset_{datetime.now(JST).strftime('%Y%m%d')}.jsonl"

        return StreamingResponse(
            iterfile(),
            media_type='application/x-jsonlines',
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

async def _run_moe_training_task(task_info: dict, config: Dict[str, Any]):
    """MoEトレーニングタスクを実行（バックグラウンド）"""
    try:
        task_info["status"] = "running"
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training started")

        # ここで実際のトレーニングロジックを実装
        # デモ用のダミー処理
        await asyncio.sleep(5)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Loading model...")
        await asyncio.sleep(3)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training epoch 1/3...")
        task_info["current_epoch"] = 1
        task_info["current_loss"] = 0.5
        await asyncio.sleep(3)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training completed")

        task_info["status"] = "completed"
        task_info["end_time"] = datetime.now(JST).isoformat()
        task_info["progress"] = 100

    except Exception as e:
        task_info["status"] = "failed"
        task_info["error"] = str(e)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Error: {str(e)}")


@router.post("/training/start")
async def start_moe_training(
    request: Request,
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
):
    """MoEトレーニングを開始"""
    try:
        task_id = str(uuid.uuid4())

        # タスク情報を保存
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "config": body,
            "start_time": datetime.now(JST).isoformat(),
            "logs": []
        }

        moe_tasks = _get_moe_tasks(request)
        moe_tasks[task_id] = task_info

        # タスクを非同期で実行
        background_tasks.add_task(_run_moe_training_task, task_info, body)

        return {"task_id": task_id, "status": "started"}

    except Exception as e:
        logger.error(f"MoE training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status/{task_id}")
async def get_moe_training_status(request: Request, task_id: str):
    """MoEトレーニングのステータスを取得"""
    moe_tasks = _get_moe_tasks(request)
    if task_id not in moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = moe_tasks[task_id]

    # 進捗計算
    if task_info["status"] == "completed":
        progress = 100
    elif task_info["status"] == "running":
        current_epoch = task_info.get("current_epoch", 0)
        total_epochs = task_info["config"].get("epochs", 3)
        progress = (current_epoch / total_epochs) * 100
    else:
        progress = 0

    return {
        **task_info,
        "progress": progress
    }


@router.post("/training/stop/{task_id}")
async def stop_moe_training(request: Request, task_id: str):
    """MoEトレーニングを停止"""
    moe_tasks = _get_moe_tasks(request)
    if task_id not in moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = moe_tasks[task_id]
    task_info["status"] = "stopped"
    task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training stopped by user")

    return {"status": "stopped", "task_id": task_id}


@router.get("/training/logs/{task_id}")
async def get_moe_training_logs(request: Request, task_id: str, tail: int = Query(50)):
    """MoEトレーニングのログを取得"""
    moe_tasks = _get_moe_tasks(request)
    if task_id not in moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = moe_tasks[task_id]
    logs = task_info.get("logs", [])

    if tail > 0:
        logs = logs[-tail:]

    return {"task_id": task_id, "logs": logs}


@router.get("/training/gpu-status")
async def get_gpu_status():
    """GPU状態を取得"""
    try:
        gpu_info: Dict[str, Any] = {
            "gpus": [],
            "cpu": None,
            "memory": None
        }

        # GPU情報
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_used": torch.cuda.memory_allocated(i) // (1024**2),  # MB
                    "memory_total": torch.cuda.get_device_properties(i).total_memory // (1024**2),  # MB
                    "memory_percent": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100,
                    "temperature": 0,  # nvidia-smiから取得する必要がある
                    "gpu_load": 0  # nvidia-smiから取得する必要がある
                }
                gpu_info["gpus"].append(gpu)

        # CPU情報
        gpu_info["cpu"] = {
            "percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count()
        }

        # メモリ情報
        mem = psutil.virtual_memory()
        gpu_info["memory"] = {
            "total": mem.total,
            "used": mem.used,
            "percent": mem.percent
        }

        return gpu_info

    except Exception as e:
        logger.error(f"GPU status error: {str(e)}")
        return {"error": str(e)}


@router.get("/training/history")
async def get_moe_training_history(request: Request, limit: int = Query(20)):
    """MoEトレーニング履歴を取得"""
    try:
        moe_tasks = _get_moe_tasks(request)
        if not moe_tasks:
            return {"history": []}

        # タスクをリストに変換してソート
        tasks = list(moe_tasks.values())
        tasks.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        # 制限を適用
        if limit > 0:
            tasks = tasks[:limit]

        return {"history": tasks}

    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/deploy/{task_id}")
async def deploy_moe_model(request: Request, task_id: str):
    """MoEモデルをデプロイ"""
    moe_tasks = _get_moe_tasks(request)
    if task_id not in moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = moe_tasks[task_id]

    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")

    # デプロイロジック（実際の実装が必要）
    model_path = f"outputs/moe_model_{task_id}"

    return {
        "status": "deployed",
        "model_path": model_path,
        "task_id": task_id
    }
