"""
MoE Training API Endpoints
トレーニング管理のバックエンドAPI
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
import subprocess
import uuid
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import psutil
import GPUtil

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APIルーター
router = APIRouter(prefix="/api/moe/training", tags=["MoE Training"])

# トレーニングタスクの管理
training_tasks = {}

class TrainingConfig(BaseModel):
    """トレーニング設定モデル"""
    training_type: str = "demo"  # demo, full, lora, continual
    base_model: str = "cyberagent/open-calm-small"  # ベースモデル
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.0001
    warmup_steps: int = 100
    save_steps: int = 500
    dataset: str = "demo"
    experts: List[str] = []
    custom_data_path: Optional[str] = None

class TrainingTask:
    """トレーニングタスク管理クラス"""
    def __init__(self, task_id: str, config: TrainingConfig):
        self.task_id = task_id
        self.config = config
        self.status = "pending"
        self.progress = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.process = None
        self.error = None
        
    def to_dict(self):
        """タスク情報を辞書に変換"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "config": self.config.dict(),
            "logs": self.logs[-100:],  # 最新100行のログ
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error
        }

@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """トレーニングを開始"""
    # タスクID生成
    task_id = str(uuid.uuid4())
    
    # タスク作成
    task = TrainingTask(task_id, config)
    training_tasks[task_id] = task
    
    # バックグラウンドでトレーニング実行
    background_tasks.add_task(execute_training, task)
    
    logger.info(f"Started training task: {task_id}")
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Training started with {len(config.experts)} experts"
    }

async def execute_training(task: TrainingTask):
    """実際のトレーニング実行"""
    try:
        task.status = "running"
        task.start_time = datetime.now()
        task.logs.append(f"[{datetime.now().isoformat()}] トレーニングを開始しました")
        
        # エキスパート設定をファイルに保存
        expert_config_path = f"/workspace/temp/expert_config_{task.task_id}.json"
        os.makedirs("/workspace/temp", exist_ok=True)
        
        with open(expert_config_path, 'w') as f:
            json.dump({
                "experts": task.config.experts,
                "training_type": task.config.training_type,
                "base_model": task.config.base_model  # ベースモデルも含める
            }, f)
        
        # トレーニングコマンドの構築
        cmd = [
            "bash",
            "/workspace/scripts/moe/train_moe.sh",
            task.config.training_type,
            str(task.config.epochs),
            str(task.config.batch_size)
        ]
        
        # 環境変数の設定
        env = os.environ.copy()
        env["EXPERT_CONFIG"] = expert_config_path
        env["BASE_MODEL"] = task.config.base_model  # ベースモデルを環境変数として渡す
        env["LEARNING_RATE"] = str(task.config.learning_rate)
        env["WARMUP_STEPS"] = str(task.config.warmup_steps)
        env["SAVE_STEPS"] = str(task.config.save_steps)
        
        if task.config.dataset == "custom" and task.config.custom_data_path:
            env["CUSTOM_DATA_PATH"] = task.config.custom_data_path
        
        # プロセス実行
        task.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # ログの非同期読み取り
        async def read_output(stream, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8').strip()
                if line_str:
                    task.logs.append(f"[{datetime.now().isoformat()}] {prefix}: {line_str}")
                    
                    # 進捗の解析
                    if "epoch" in line_str.lower():
                        # エポック情報の抽出
                        try:
                            parts = line_str.split()
                            for i, part in enumerate(parts):
                                if "epoch" in part.lower() and i + 1 < len(parts):
                                    epoch_str = parts[i + 1].strip(':,')
                                    if '/' in epoch_str:
                                        current, total = epoch_str.split('/')
                                        task.current_epoch = int(current)
                                        task.progress = (task.current_epoch / task.config.epochs) * 100
                        except:
                            pass
                    
                    if "loss" in line_str.lower():
                        # 損失値の抽出
                        try:
                            parts = line_str.split()
                            for i, part in enumerate(parts):
                                if "loss" in part.lower() and i + 1 < len(parts):
                                    loss_str = parts[i + 1].strip(':,')
                                    task.current_loss = float(loss_str)
                        except:
                            pass
        
        # 並行して出力を読み取る
        await asyncio.gather(
            read_output(task.process.stdout, "OUT"),
            read_output(task.process.stderr, "ERR")
        )
        
        # プロセスの終了を待つ
        return_code = await task.process.wait()
        
        if return_code == 0:
            task.status = "completed"
            task.progress = 100
            task.logs.append(f"[{datetime.now().isoformat()}] トレーニングが正常に完了しました")
        else:
            task.status = "failed"
            task.error = f"Process exited with code {return_code}"
            task.logs.append(f"[{datetime.now().isoformat()}] エラー: {task.error}")
        
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.logs.append(f"[{datetime.now().isoformat()}] エラー: {task.error}")
        logger.error(f"Training task {task.task_id} failed: {e}")
    
    finally:
        task.end_time = datetime.now()
        # 一時ファイルのクリーンアップ
        if os.path.exists(expert_config_path):
            os.remove(expert_config_path)

@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    return task.to_dict()

@router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """トレーニングを停止"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    
    if task.process and task.status == "running":
        try:
            task.process.terminate()
            await asyncio.sleep(1)
            if task.process.returncode is None:
                task.process.kill()
            task.status = "stopped"
            task.logs.append(f"[{datetime.now().isoformat()}] ユーザーによって停止されました")
            return {"status": "stopped", "message": "Training stopped successfully"}
        except Exception as e:
            logger.error(f"Failed to stop training {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": task.status, "message": "Training is not running"}

@router.get("/history")
async def get_training_history(limit: int = 10):
    """トレーニング履歴を取得"""
    # 完了したタスクのみフィルタリング
    completed_tasks = [
        task.to_dict() for task in training_tasks.values()
        if task.status in ["completed", "failed", "stopped"]
    ]
    
    # 開始時刻でソート（新しい順）
    completed_tasks.sort(key=lambda x: x["start_time"] or "", reverse=True)
    
    return {
        "history": completed_tasks[:limit],
        "total": len(completed_tasks)
    }

@router.get("/gpu-status")
async def get_gpu_status():
    """GPU状態を取得"""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_free": gpu.memoryFree,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "gpu_load": gpu.load * 100,
                "temperature": gpu.temperature
            })
        
        # CPU情報も追加
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "gpus": gpu_info,
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "used": memory.used,
                "total": memory.total,
                "percent": memory.percent
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        # GPUtilsが使えない場合はnvidia-smiを試す
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                gpu_info = []
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_info.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "memory_used": float(parts[2]),
                            "memory_total": float(parts[3]),
                            "memory_free": float(parts[4]),
                            "memory_percent": (float(parts[2]) / float(parts[3])) * 100,
                            "gpu_load": float(parts[5]),
                            "temperature": float(parts[6])
                        })
                
                return {
                    "gpus": gpu_info,
                    "timestamp": datetime.now().isoformat()
                }
        except:
            pass
        
        return {
            "error": "GPU information not available",
            "timestamp": datetime.now().isoformat()
        }

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """カスタムデータセットをアップロード"""
    try:
        # アップロードディレクトリ作成
        upload_dir = Path("/workspace/data/custom_datasets")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル保存
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # ファイル検証（JSON/JSONL/CSVかチェック）
        if file.filename.endswith(('.json', '.jsonl', '.csv')):
            return {
                "status": "success",
                "file_path": str(file_path),
                "filename": file.filename,
                "size": len(content)
            }
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload JSON, JSONL, or CSV file.")
    
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/{task_id}")
async def get_training_logs(task_id: str, tail: int = 100):
    """トレーニングログを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    logs = task.logs[-tail:] if tail > 0 else task.logs
    
    return {
        "task_id": task_id,
        "logs": logs,
        "total_lines": len(task.logs)
    }

@router.get("/stream-logs/{task_id}")
async def stream_training_logs(task_id: str):
    """トレーニングログをストリーミング"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    
    async def generate():
        last_index = 0
        while task.status == "running" or last_index < len(task.logs):
            if last_index < len(task.logs):
                new_logs = task.logs[last_index:]
                last_index = len(task.logs)
                for log in new_logs:
                    yield f"data: {json.dumps({'log': log})}\n\n"
            else:
                await asyncio.sleep(1)
        
        # 最終ステータス送信
        yield f"data: {json.dumps({'status': task.status, 'completed': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/deploy/{task_id}")
async def deploy_model(task_id: str):
    """トレーニング済みモデルをデプロイ"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Training must be completed before deployment")
    
    try:
        # モデルのパスを特定
        if task.config.training_type == "demo":
            model_path = "/workspace/outputs/moe_demo"
        else:
            model_path = "/workspace/outputs/moe_civil"
        
        # デプロイメント設定を作成
        deploy_config = {
            "model_path": model_path,
            "task_id": task_id,
            "experts": task.config.experts,
            "deployed_at": datetime.now().isoformat()
        }
        
        # デプロイメント設定を保存
        deploy_config_path = "/workspace/config/deployed_moe.json"
        with open(deploy_config_path, 'w') as f:
            json.dump(deploy_config, f, indent=2)
        
        return {
            "status": "deployed",
            "model_path": model_path,
            "message": "Model deployed successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# クリーンアップタスク（古いタスクを削除）
async def cleanup_old_tasks():
    """24時間以上経過したタスクをクリーンアップ"""
    while True:
        await asyncio.sleep(3600)  # 1時間ごとにチェック
        
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task in training_tasks.items():
            if task.end_time:
                elapsed = current_time - task.end_time
                if elapsed.total_seconds() > 86400:  # 24時間
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del training_tasks[task_id]
            logger.info(f"Cleaned up old task: {task_id}")

# バックグラウンドクリーンアップの開始
# asyncio.create_task(cleanup_old_tasks())