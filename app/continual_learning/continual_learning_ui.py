# app/continual_learning/continual_learning_ui.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import json
import uuid
from pathlib import Path
import aiofiles
import logging
from concurrent.futures import ThreadPoolExecutor
import torch

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# タスク管理
class TaskManager:
    """非同期タスク管理"""
    def __init__(self):
        self.tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def create_task(self, task_type: str) -> str:
        """新しいタスクを作成"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "metrics": {},
            "messages": []
        }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """タスクの状態を更新"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """タスク情報を取得"""
        return self.tasks.get(task_id)

# グローバルタスクマネージャー
task_manager = TaskManager()

# リクエストモデル
class ContinualLearningRequest(BaseModel):
    """継続学習リクエスト"""
    base_model: str
    task_name: str
    use_previous_tasks: bool = True
    ewc_lambda: float = 5000.0
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    warmup_ratio: float = 0.1
    use_memory_efficient: bool = True
    
class TaskStatusResponse(BaseModel):
    """タスクステータスレスポンス"""
    task_id: str
    status: str
    progress: float
    metrics: Dict
    messages: List[str]
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

# 継続学習の実行関数
async def run_continual_learning_task(
    task_id: str,
    base_model: str,
    task_name: str,
    dataset_path: str,
    config: ContinualLearningRequest
):
    """継続学習タスクの非同期実行"""
    try:
        # タスク開始
        task_manager.update_task(task_id, status="running", progress=0)
        
        # 継続学習パイプラインのインポート
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from src.training.continual_learning_pipeline import ContinualLearningPipeline
        
        # パイプラインの初期化
        pipeline = ContinualLearningPipeline(base_model_path=base_model)
        
        # 進捗コールバック
        def progress_callback(progress: float, message: str):
            task_manager.update_task(
                task_id,
                progress=progress,
                messages=task_manager.get_task(task_id)["messages"] + [message]
            )
        
        # モデルのロード
        task_manager.update_task(task_id, progress=10, messages=["モデルをロード中..."])
        
        if base_model.startswith("outputs/"):
            model = pipeline.load_finetuned_model(base_model)
        else:
            # Hugging Faceモデルの場合
            from src.models.base_model import BaseModel
            model = BaseModel.from_pretrained(base_model)
        
        # 継続学習の実行
        task_manager.update_task(task_id, progress=20, messages=["継続学習を開始..."])
        
        model = pipeline.run_continual_task(
            model=model,
            task_name=task_name,
            train_dataset_path=dataset_path,
            epochs=config.epochs,
            use_previous_fisher=config.use_previous_tasks,
            fisher_importance=config.ewc_lambda,
            progress_callback=progress_callback
        )
        
        # 評価の実行
        task_manager.update_task(task_id, progress=90, messages=["評価を実行中..."])
        
        from src.evaluation.continual_metrics import ContinualLearningEvaluator
        evaluator = ContinualLearningEvaluator()
        
        # 破滅的忘却の評価
        if len(pipeline.task_history) > 1:
            forgetting_results = evaluator.evaluate_forgetting(
                model, pipeline.task_history
            )
            
            # レポート生成
            report_path = evaluator.generate_report(forgetting_results)
            
            task_manager.update_task(
                task_id,
                metrics={
                    "forgetting_results": forgetting_results,
                    "report_path": str(report_path)
                }
            )
        
        # タスク完了
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            completed_at=datetime.now().isoformat(),
            messages=task_manager.get_task(task_id)["messages"] + ["継続学習が完了しました"]
        )
        
    except Exception as e:
        logger.error(f"継続学習エラー: {str(e)}", exc_info=True)
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now().isoformat()
        )

# FastAPIルーター
def create_continual_learning_router():
    """継続学習用のルーターを作成"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/api/continual-learning", tags=["continual-learning"])
    
    @router.post("/start")
    async def start_continual_learning(
        background_tasks: BackgroundTasks,
        dataset: UploadFile = File(...),
        config: str = Form(...),  # JSON文字列として受け取る
    ):
        """継続学習を開始"""
        try:
            # 設定のパース
            config_dict = json.loads(config)
            cl_config = ContinualLearningRequest(**config_dict)
            
            # データセットの保存
            dataset_dir = Path("data/continual")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_path = dataset_dir / f"{cl_config.task_name}_{dataset.filename}"
            
            async with aiofiles.open(dataset_path, 'wb') as f:
                content = await dataset.read()
                await f.write(content)
            
            # タスクの作成
            task_id = task_manager.create_task("continual_learning")
            
            # バックグラウンドで実行
            background_tasks.add_task(
                run_continual_learning_task,
                task_id,
                cl_config.base_model,
                cl_config.task_name,
                str(dataset_path),
                cl_config
            )
            
            return JSONResponse({
                "task_id": task_id,
                "status": "started",
                "message": f"継続学習タスク '{cl_config.task_name}' を開始しました"
            })
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/status/{task_id}")
    async def get_continual_learning_status(task_id: str):
        """継続学習の進捗確認"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="タスクが見つかりません")
        
        return TaskStatusResponse(**task)
    
    @router.get("/tasks")
    async def list_continual_learning_tasks():
        """すべての継続学習タスクをリスト"""
        return list(task_manager.tasks.values())
    
    @router.delete("/task/{task_id}")
    async def cancel_continual_learning_task(task_id: str):
        """継続学習タスクをキャンセル"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="タスクが見つかりません")
        
        if task["status"] == "running":
            task_manager.update_task(task_id, status="cancelled")
        
        return {"message": "タスクをキャンセルしました"}
    
    @router.get("/models")
    async def list_available_models():
        """利用可能なモデルをリスト"""
        models = []
        
        # outputsディレクトリのモデル
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for model_dir in outputs_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    training_info_path = model_dir / "training_info.json"
                    info = {}
                    if training_info_path.exists():
                        with open(training_info_path) as f:
                            info = json.load(f)
                    
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": "finetuned",
                        "created_at": datetime.fromtimestamp(
                            model_dir.stat().st_mtime
                        ).isoformat(),
                        "info": info
                    })
        
        # デフォルトモデル
        models.extend([
            {
                "name": "deepseek-ai/deepseek-llm-7b-base",
                "path": "deepseek-ai/deepseek-llm-7b-base",
                "type": "base"
            },
            {
                "name": "deepseek-ai/deepseek-llm-7b-chat",
                "path": "deepseek-ai/deepseek-llm-7b-chat",
                "type": "chat"
            }
        ])
        
        return models
    
    @router.get("/tasks/history")
    async def get_task_history():
        """継続学習の履歴を取得"""
        history_path = Path("outputs/ewc_data/task_history.json")
        if history_path.exists():
            with open(history_path) as f:
                return json.load(f)
        return []
    
    @router.get("/evaluation/report/{task_id}")
    async def get_evaluation_report(task_id: str):
        """評価レポートを取得"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="タスクが見つかりません")
        
        if "metrics" in task and "report_path" in task["metrics"]:
            report_path = Path(task["metrics"]["report_path"])
            if report_path.exists():
                return StreamingResponse(
                    open(report_path, "rb"),
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"inline; filename={report_path.name}"
                    }
                )
        
        raise HTTPException(status_code=404, detail="レポートが見つかりません")
    
    @router.post("/config/validate")
    async def validate_config(config: ContinualLearningRequest):
        """設定の検証"""
        errors = []
        
        # モデルパスの検証
        if config.base_model.startswith("outputs/"):
            if not Path(config.base_model).exists():
                errors.append(f"モデルが見つかりません: {config.base_model}")
        
        # パラメータの検証
        if config.epochs < 1:
            errors.append("エポック数は1以上である必要があります")
        
        if config.learning_rate <= 0:
            errors.append("学習率は正の値である必要があります")
        
        if config.ewc_lambda < 0:
            errors.append("EWCラムダは非負である必要があります")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "message": "設定は有効です"}
    
    return router

# WebSocketサポート（リアルタイム進捗更新）
async def websocket_endpoint(websocket):
    """WebSocket経由でのリアルタイム進捗更新"""
    await websocket.accept()
    
    try:
        # クライアントからタスクIDを受信
        data = await websocket.receive_json()
        task_id = data.get("task_id")
        
        if not task_id:
            await websocket.send_json({"error": "task_id is required"})
            return
        
        # 進捗を定期的に送信
        while True:
            task = task_manager.get_task(task_id)
            if task:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task["status"],
                    "progress": task["progress"],
                    "messages": task["messages"][-5:]  # 最新5件のメッセージ
                })
                
                if task["status"] in ["completed", "failed", "cancelled"]:
                    break
            
            await asyncio.sleep(1)  # 1秒ごとに更新
            
    except Exception as e:
        logger.error(f"WebSocketエラー: {str(e)}")
    finally:
        await websocket.close()
