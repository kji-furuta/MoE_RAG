"""RLAnything Training API Router.

RLAnythingフレームワークのAPI エンドポイントを提供する。
既存のfinetuning.pyルーターパターンに準拠。
"""

from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import PROJECT_ROOT, logger, training_tasks

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/api/rl", tags=["rlanything"])


# ── リクエスト/レスポンスモデル ────────────────────────


class RLAnythingTrainRequest(BaseModel):
    """RLAnythingトレーニング開始リクエスト"""

    model_name: str = Field(
        default="cyberagent/calm3-22b-chat",
        description="ベースモデル名",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="学習データセットのパス",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="出力ディレクトリ",
    )

    # GRPO設定
    grpo_num_generations: int = Field(default=8, ge=2, le=64)
    grpo_temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    grpo_beta: float = Field(default=0.04, ge=0.0, le=1.0)
    policy_learning_rate: float = Field(default=1e-6, gt=0.0)
    policy_max_steps: int = Field(default=500, ge=1)

    # LoRA設定
    use_lora: bool = True
    lora_r: int = Field(default=64, ge=4, le=256)
    lora_alpha: int = Field(default=128, ge=4, le=512)

    # 報酬設定
    reward_outcome_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    reward_self_consistency_k: int = Field(default=5, ge=1)

    # 環境設定
    env_success_rate_low: float = Field(default=0.2, ge=0.0, le=1.0)
    env_success_rate_high: float = Field(default=0.8, ge=0.0, le=1.0)
    env_max_difficulty: int = Field(default=5, ge=1, le=10)

    # オーケストレーション設定
    num_iterations: int = Field(default=10, ge=1, le=100)
    trajectories_per_iteration: int = Field(default=64, ge=1, le=1000)
    early_stopping_patience: int = Field(default=3, ge=1)

    # メモリ最適化
    use_quantization: bool = True
    quantization_bits: int = Field(default=4, ge=4, le=8)
    gradient_checkpointing: bool = True
    bf16: bool = True

    # タスクデータ（難易度別のプロンプト群）
    tasks: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="難易度別タスクプロンプト: {'1': ['prompt1', ...], '2': [...]}",
    )


class RLAnythingStatusResponse(BaseModel):
    """RLAnythingステータスレスポンス"""

    task_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ── バックグラウンドタスク ────────────────────────────


def _run_rlanything_training(task_id: str, request: RLAnythingTrainRequest) -> None:
    """バックグラウンドでRLAnythingトレーニングを実行"""
    try:
        from src.training.rl import RLAnythingConfig, RLAnythingOrchestrator

        training_tasks[task_id]["status"] = "running"
        training_tasks[task_id]["started_at"] = datetime.now(JST).isoformat()

        # 設定構築
        output_dir = request.output_dir or str(
            PROJECT_ROOT / "outputs" / f"rlanything_{task_id}"
        )

        config = RLAnythingConfig(
            model_name=request.model_name,
            output_dir=output_dir,
            # LoRA
            use_lora=request.use_lora,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            # GRPO
            grpo_num_generations=request.grpo_num_generations,
            grpo_temperature=request.grpo_temperature,
            grpo_beta=request.grpo_beta,
            policy_learning_rate=request.policy_learning_rate,
            policy_max_steps=request.policy_max_steps,
            # 報酬
            reward_outcome_weight=request.reward_outcome_weight,
            reward_process_weight=1.0 - request.reward_outcome_weight,
            reward_self_consistency_k=request.reward_self_consistency_k,
            # 環境
            env_success_rate_low=request.env_success_rate_low,
            env_success_rate_high=request.env_success_rate_high,
            env_max_difficulty=request.env_max_difficulty,
            # オーケストレーション
            num_iterations=request.num_iterations,
            trajectories_per_iteration=request.trajectories_per_iteration,
            early_stopping_patience=request.early_stopping_patience,
            # メモリ
            use_quantization=request.use_quantization,
            quantization_bits=request.quantization_bits,
            gradient_checkpointing=request.gradient_checkpointing,
            bf16=request.bf16,
            # データ
            dataset_path=request.dataset_path,
        )

        # オーケストレータ初期化
        orchestrator = RLAnythingOrchestrator(config)

        # ステータスコールバック
        def status_callback(message: str):
            training_tasks[task_id]["progress_message"] = message

        orchestrator.set_callbacks(
            on_iteration_complete=lambda it, m: _update_iteration_progress(
                task_id, it, m
            ),
            status_callback=status_callback,
        )

        # セットアップ
        orchestrator.setup()

        # タスクプールの登録
        if request.tasks:
            tasks_by_difficulty = {
                int(k): v for k, v in request.tasks.items()
            }
            orchestrator.environment.register_all_tasks(tasks_by_difficulty)
        elif request.dataset_path:
            # データセットからプロンプトを読み込んでタスクとして登録
            _load_tasks_from_dataset(orchestrator, request.dataset_path)

        # 実行
        result = orchestrator.run()

        # アダプタ保存
        adapter_path = orchestrator.policy.save_adapter()

        training_tasks[task_id]["status"] = "completed"
        training_tasks[task_id]["completed_at"] = datetime.now(JST).isoformat()
        training_tasks[task_id]["result"] = {
            "adapter_path": adapter_path,
            "final_mean_reward": result.get("final_mean_reward", 0.0),
            "final_success_rate": result.get("final_success_rate", 0.0),
            "iterations_completed": result.get("num_iterations_completed", 0),
            "elapsed_time": result.get("elapsed_time", 0.0),
        }

    except Exception as e:
        logger.error(f"RLAnythingトレーニングエラー [task={task_id}]: {e}")
        logger.error(traceback.format_exc())
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["error"] = str(e)
        training_tasks[task_id]["failed_at"] = datetime.now(JST).isoformat()


def _update_iteration_progress(
    task_id: str, iteration: int, metrics: Dict[str, Any]
) -> None:
    """イテレーション完了時にプログレスを更新"""
    if task_id in training_tasks:
        training_tasks[task_id]["progress"] = {
            "current_iteration": iteration + 1,
            "mean_reward": metrics.get("mean_reward", 0.0),
            "success_rate": metrics.get("success_rate", 0.0),
            "difficulty": metrics.get("difficulty", 1),
        }


def _load_tasks_from_dataset(orchestrator, dataset_path: str) -> None:
    """データセットファイルからタスクを読み込んで環境に登録"""
    try:
        import json as _json
        from pathlib import Path

        path = Path(dataset_path)
        if not path.exists():
            logger.warning(f"データセットファイルが見つかりません: {dataset_path}")
            return

        # フィールド優先順位: prompt > instruction > text
        PROMPT_FIELDS = ("prompt", "instruction", "text")

        prompts: List[str] = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = _json.loads(line.strip())
                    for field in PROMPT_FIELDS:
                        if field in record:
                            prompts.append(record[field])
                            break
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
                if isinstance(data, list):
                    for record in data:
                        if isinstance(record, dict):
                            for field in PROMPT_FIELDS:
                                if field in record:
                                    prompts.append(record[field])
                                    break
                        elif isinstance(record, str):
                            prompts.append(record)

        if prompts:
            # 全プロンプトをデフォルト難易度で登録
            orchestrator.environment.register_tasks(prompts)
            logger.info(f"データセットから{len(prompts)}プロンプトを登録")

    except Exception as e:
        logger.warning(f"データセット読み込み失敗: {e}")


# ── API エンドポイント ────────────────────────────────


@router.post("/train")
async def start_rlanything_training(
    request: RLAnythingTrainRequest,
    background_tasks: BackgroundTasks,
):
    """RLAnythingトレーニングを開始

    閉ループ強化学習をバックグラウンドで実行し、タスクIDを返す。
    """
    try:
        task_id = f"rl_{uuid.uuid4().hex[:12]}"

        training_tasks[task_id] = {
            "task_id": task_id,
            "type": "rlanything",
            "status": "queued",
            "model_name": request.model_name,
            "num_iterations": request.num_iterations,
            "created_at": datetime.now(JST).isoformat(),
            "progress": None,
            "progress_message": "キューに追加されました",
            "result": None,
            "error": None,
        }

        background_tasks.add_task(_run_rlanything_training, task_id, request)

        return {"task_id": task_id, "status": "started"}

    except Exception as e:
        logger.error(f"RLAnythingトレーニング開始エラー: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/status/{task_id}")
async def get_rlanything_status(task_id: str):
    """RLAnythingトレーニングのステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return training_tasks[task_id]


@router.get("/tasks")
async def list_rlanything_tasks():
    """RLAnythingトレーニングタスク一覧を取得"""
    rl_tasks = {
        k: v for k, v in training_tasks.items()
        if v.get("type") == "rlanything"
    }
    return {"tasks": rl_tasks, "count": len(rl_tasks)}


@router.post("/stop/{task_id}")
async def stop_rlanything_training(task_id: str):
    """RLAnythingトレーニングを停止

    注意: バックグラウンドタスクの停止は次のイテレーション完了時に反映される。
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = training_tasks[task_id]
    if task["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not running (current: {task['status']})",
        )

    # 停止フラグを設定（オーケストレータが次のイテレーションで確認）
    training_tasks[task_id]["status"] = "stopping"
    return {"task_id": task_id, "status": "stopping"}


@router.get("/config/default")
async def get_default_config():
    """RLAnythingのデフォルト設定を取得"""
    try:
        from src.training.rl import RLAnythingConfig
        config = RLAnythingConfig()
        return config.to_dict()
    except Exception as e:
        logger.error(f"デフォルト設定取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-config")
async def validate_config(config_data: Dict[str, Any]):
    """RLAnything設定のバリデーション"""
    try:
        from src.training.rl import RLAnythingConfig

        # 有効なフィールドのみフィルタ
        valid_fields = {f.name for f in RLAnythingConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}

        config = RLAnythingConfig(**filtered)
        warnings = config.validate()

        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "config": config.to_dict(),
        }
    except Exception as e:
        return {
            "valid": False,
            "warnings": [str(e)],
            "config": None,
        }
