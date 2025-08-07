"""
継続学習関連のAPIルーター
"""

import os
import json
import uuid
import asyncio
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form

from ..dependencies import (
    logger, PROJECT_ROOT, CONTINUAL_LEARNING_DIR,
    continual_tasks, CONTINUAL_TASKS_FILE
)

router = APIRouter(prefix="/api/continual-learning", tags=["continual_learning"])


def load_continual_tasks():
    """保存された継続学習タスクを読み込む"""
    global continual_tasks
    try:
        if CONTINUAL_TASKS_FILE.exists():
            with open(CONTINUAL_TASKS_FILE, 'r', encoding='utf-8') as f:
                continual_tasks = json.load(f)
                logger.info(f"継続学習タスクを読み込みました: {len(continual_tasks)}件")
        else:
            continual_tasks = {}
            logger.info("継続学習タスクファイルが存在しません。新規作成します。")
    except Exception as e:
        logger.error(f"継続学習タスク読み込みエラー: {str(e)}")
        continual_tasks = {}


def save_continual_tasks():
    """継続学習タスクを保存する"""
    try:
        # ディレクトリが存在しない場合は作成
        CONTINUAL_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CONTINUAL_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(continual_tasks, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"継続学習タスクを保存しました: {len(continual_tasks)}件")
    except Exception as e:
        logger.error(f"継続学習タスク保存エラー: {str(e)}")


async def run_continual_learning_background(task_id: str, config: dict, dataset_path: str):
    """バックグラウンドで継続学習を実行"""
    try:
        # タスクステータスを更新
        continual_tasks[task_id]["status"] = "running"
        continual_tasks[task_id]["message"] = "継続学習を準備中..."
        save_continual_tasks()  # タスクの状態を保存
        
        logger.info(f"継続学習タスク開始: {task_id}")
        logger.info(f"設定: {config}")
        
        # 実際の継続学習処理をここに実装
        # 現在はシミュレーション
        total_epochs = config.get("epochs", 3)
        base_model_path = config.get("base_model")
        
        # モデルの存在確認
        if not base_model_path:
            raise ValueError("ベースモデルが指定されていません")
        
        # ファインチューニング済みモデルの場合はパスを確認
        if "/" not in base_model_path and os.path.exists(base_model_path):
            logger.info(f"ファインチューニング済みモデルを使用: {base_model_path}")
        else:
            logger.info(f"ベースモデルを使用: {base_model_path}")
        
        for epoch in range(total_epochs):
            # プログレス更新
            progress = int((epoch + 1) / total_epochs * 100)
            continual_tasks[task_id]["progress"] = progress
            continual_tasks[task_id]["current_epoch"] = epoch + 1
            continual_tasks[task_id]["total_epochs"] = total_epochs
            continual_tasks[task_id]["message"] = f"エポック {epoch + 1}/{total_epochs} を実行中..."
            save_continual_tasks()  # 進捗を保存
            
            logger.info(f"タスク {task_id}: エポック {epoch + 1}/{total_epochs}")
            
            # 実際の学習処理をここに追加
            await asyncio.sleep(5)  # シミュレーション用の待機
            
            # TODO: 実際の継続学習実装
            # 1. モデルとトークナイザーの読み込み
            # 2. データセットの準備
            # 3. EWC設定
            # 4. トレーニングループ
            # 5. チェックポイント保存
        
        # 完了
        continual_tasks[task_id]["status"] = "completed"
        continual_tasks[task_id]["message"] = "継続学習が完了しました"
        continual_tasks[task_id]["completed_at"] = datetime.now(JST).isoformat()
        save_continual_tasks()  # 完了状態を保存
        
        # 出力パスを設定（モデル管理と同じ形式）
        output_dir = f"outputs/continual_{config.get('task_name')}_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}"
        continual_tasks[task_id]["output_path"] = output_dir
        
        # モデル管理に登録するための情報を保存
        model_info = {
            "name": f"continual_{config.get('task_name')}",
            "path": output_dir,
            "base_model": base_model_path,
            "training_method": "continual_ewc",
            "created_at": datetime.now(JST).isoformat(),
            "training_params": {
                "epochs": config.get("epochs", 3),
                "learning_rate": config.get("learning_rate", 2e-5),
                "ewc_lambda": config.get("ewc_lambda", 5000),
                "use_previous_tasks": config.get("use_previous_tasks", True)
            },
            "task_history": config.get("task_name")
        }
        
        # モデル情報をJSON ファイルとして保存（モデル管理が読み取れるように）
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        continual_tasks[task_id]["model_info"] = model_info
        save_continual_tasks()  # モデル情報を保存
        
        logger.info(f"継続学習タスク完了: {task_id}")
        
    except Exception as e:
        logger.error(f"継続学習エラー (タスク {task_id}): {str(e)}")
        logger.exception("詳細なエラー情報:")
        continual_tasks[task_id]["status"] = "failed"
        continual_tasks[task_id]["message"] = f"エラー: {str(e)}"
        continual_tasks[task_id]["error"] = str(e)
        save_continual_tasks()  # エラー状態を保存


def get_saved_models():
    """保存されたモデル一覧を取得（簡易版）"""
    models = []
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if outputs_dir.exists():
        for model_dir in outputs_dir.iterdir():
            if model_dir.is_dir():
                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                }
                
                # model_info.jsonがあれば読み込む
                info_file = model_dir / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            saved_info = json.load(f)
                            model_info.update(saved_info)
                    except:
                        pass
                
                models.append(model_info)
    
    return models


@router.get("/models")
async def get_continual_learning_models():
    """継続学習用の利用可能モデル一覧を取得"""
    try:
        # ファインチューニング済みモデルを取得
        saved_models = get_saved_models()
        
        # ベースモデルも含める
        base_models = [
            {
                "name": "cyberagent/calm3-22b-chat",
                "path": "cyberagent/calm3-22b-chat",
                "type": "base",
                "description": "日本語特化型22Bモデル（推奨）"
            },
            {
                "name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
                "path": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
                "type": "base",
                "description": "日本語特化型32Bモデル"
            },
            {
                "name": "Qwen/Qwen2.5-14B-Instruct",
                "path": "Qwen/Qwen2.5-14B-Instruct",
                "type": "base",
                "description": "多言語対応14Bモデル"
            },
            {
                "name": "Qwen/Qwen2.5-32B-Instruct",
                "path": "Qwen/Qwen2.5-32B-Instruct",
                "type": "base",
                "description": "多言語対応32Bモデル"
            }
        ]
        
        # ファインチューニング済みモデルを継続学習用形式に変換
        continual_models = []
        
        # ベースモデルを追加
        for model in base_models:
            continual_models.append({
                "name": model["name"],
                "path": model["path"],
                "type": "base",
                "description": model["description"]
            })
        
        # ファインチューニング済みモデルを追加
        for model in saved_models:
            continual_models.append({
                "name": f"{model['name']} (ファインチューニング済み)",
                "path": model["path"],
                "type": "finetuned",
                "description": f"学習日時: {model.get('created_at', '不明')}"
            })
        
        logger.info(f"継続学習用モデル一覧を取得: {len(continual_models)}個")
        return continual_models
        
    except Exception as e:
        logger.error(f"継続学習用モデル取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_continual_learning(
    background_tasks: BackgroundTasks,
    config: str = Form(...),
    dataset: UploadFile = File(...)
):
    """継続学習を開始"""
    try:
        # 設定をパース
        config_data = json.loads(config)
        
        # データセットを保存
        CONTINUAL_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        
        dataset_path = CONTINUAL_LEARNING_DIR / f"{uuid.uuid4()}_{dataset.filename}"
        with open(dataset_path, "wb") as f:
            content = await dataset.read()
            f.write(content)
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # タスク情報を保存
        continual_tasks[task_id] = {
            "task_id": task_id,
            "task_name": config_data["task_name"],
            "status": "pending",
            "progress": 0,
            "started_at": datetime.now(JST).isoformat(),
            "config": config_data,
            "dataset_path": str(dataset_path)
        }
        save_continual_tasks()  # 新しいタスクを保存
        
        # バックグラウンドで継続学習を実行
        background_tasks.add_task(
            run_continual_learning_background,
            task_id,
            config_data,
            str(dataset_path)
        )
        
        return {
            "task_id": task_id,
            "message": f"継続学習タスク '{config_data['task_name']}' を開始しました"
        }
        
    except Exception as e:
        logger.error(f"継続学習開始エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def get_continual_tasks():
    """継続学習タスクの一覧を取得"""
    try:
        # アクティブなタスクのみを返す
        active_tasks = []
        for task_id, task in continual_tasks.items():
            if task["status"] in ["pending", "running", "completed", "failed"]:
                active_tasks.append(task)
        
        # 新しい順にソート
        active_tasks.sort(key=lambda x: x["started_at"], reverse=True)
        
        return active_tasks[:10]  # 最新10件を返す
        
    except Exception as e:
        logger.error(f"タスク一覧取得エラー: {str(e)}")
        return []


@router.get("/history")
async def get_continual_history():
    """継続学習の履歴を取得"""
    try:
        history = []
        
        # 完了したタスクを履歴として返す
        for task_id, task in continual_tasks.items():
            if task["status"] == "completed":
                history.append({
                    "task_name": task["task_name"],
                    "base_model": task["config"].get("base_model", "unknown"),
                    "completed_at": task.get("completed_at"),
                    "epochs": task["config"].get("epochs", 0),
                    "final_loss": random.uniform(0.1, 0.5),  # ダミーデータ
                    "output_path": task.get("output_path")
                })
        
        # 新しい順にソート
        history.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
        
        return history
        
    except Exception as e:
        logger.error(f"履歴取得エラー: {str(e)}")
        return []