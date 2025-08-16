# app/continual_learning/task_scheduler.py
"""
継続学習のタスクスケジューラー
定期実行、優先度管理、依存関係処理を提供
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import heapq
from enum import Enum
import json
from pathlib import Path
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """タスク優先度"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

class TaskStatus(Enum):
    """タスクステータス"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScheduledTask:
    """スケジュールされたタスク"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        config: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        scheduled_time: Optional[datetime] = None,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        max_retries: int = 3
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.config = config
        self.priority = priority
        self.dependencies = dependencies or []
        self.scheduled_time = scheduled_time
        self.cron_expression = cron_expression
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self.retry_count = 0
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "config": self.config,
            "priority": self.priority.name,
            "dependencies": self.dependencies,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }
    
    def __lt__(self, other):
        """優先度比較（ヒープキュー用）"""
        return self.priority.value < other.priority.value

class ContinualLearningScheduler:
    """継続学習タスクスケジューラー"""
    
    def __init__(self, max_concurrent_tasks: int = 2):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.scheduler = AsyncIOScheduler()
        self.task_queue: List[ScheduledTask] = []  # 優先度キュー
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.task_executors: Dict[str, Callable] = {}
        self.persistence_path = Path("outputs/scheduler_state.json")
        
    async def initialize(self):
        """スケジューラーを初期化"""
        # 永続化されたタスクを読み込み
        self._load_state()
        
        # APSchedulerを開始
        self.scheduler.start()
        
        # タスク実行ループを開始
        asyncio.create_task(self._task_execution_loop())
        
        logger.info("Task scheduler initialized")
    
    def register_executor(self, task_type: str, executor: Callable):
        """タスク実行関数を登録"""
        self.task_executors[task_type] = executor
    
    def schedule_task(
        self,
        task_id: str,
        task_type: str,
        config: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        scheduled_time: Optional[datetime] = None,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None
    ) -> ScheduledTask:
        """タスクをスケジュール"""
        task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            config=config,
            priority=priority,
            dependencies=dependencies,
            scheduled_time=scheduled_time,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds
        )
        
        # 定期実行の設定
        if cron_expression:
            self.scheduler.add_job(
                self._enqueue_task,
                CronTrigger.from_crontab(cron_expression),
                args=[task],
                id=f"cron_{task_id}"
            )
            task.status = TaskStatus.SCHEDULED
        elif interval_seconds:
            self.scheduler.add_job(
                self._enqueue_task,
                IntervalTrigger(seconds=interval_seconds),
                args=[task],
                id=f"interval_{task_id}"
            )
            task.status = TaskStatus.SCHEDULED
        elif scheduled_time:
            self.scheduler.add_job(
                self._enqueue_task,
                'date',
                run_date=scheduled_time,
                args=[task],
                id=f"scheduled_{task_id}"
            )
            task.status = TaskStatus.SCHEDULED
        else:
            # 即時実行
            heapq.heappush(self.task_queue, task)
        
        self._save_state()
        return task
    
    async def _enqueue_task(self, task: ScheduledTask):
        """タスクをキューに追加"""
        # 新しいインスタンスを作成（定期実行用）
        new_task = ScheduledTask(
            task_id=f"{task.task_id}_{datetime.now().timestamp()}",
            task_type=task.task_type,
            config=task.config.copy(),
            priority=task.priority,
            dependencies=task.dependencies.copy()
        )
        heapq.heappush(self.task_queue, new_task)
        self._save_state()
    
    async def _task_execution_loop(self):
        """タスク実行ループ"""
        while True:
            try:
                # 実行可能なタスクをチェック
                if len(self.running_tasks) < self.max_concurrent_tasks and self.task_queue:
                    # 優先度が最も高いタスクを取得
                    task = heapq.heappop(self.task_queue)
                    
                    # 依存関係をチェック
                    if self._check_dependencies(task):
                        asyncio.create_task(self._execute_task(task))
                    else:
                        # 依存関係が満たされていない場合は再度キューに戻す
                        heapq.heappush(self.task_queue, task)
                
                await asyncio.sleep(1)  # 1秒ごとにチェック
                
            except Exception as e:
                logger.error(f"Task execution loop error: {e}")
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """依存関係をチェック"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _execute_task(self, task: ScheduledTask):
        """タスクを実行"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.task_id] = task
        self._save_state()
        
        try:
            # タスク実行関数を取得
            executor = self.task_executors.get(task.task_type)
            if not executor:
                raise ValueError(f"No executor registered for task type: {task.task_type}")
            
            # タスクを実行
            await executor(task.task_id, task.config)
            
            # 完了処理
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self.completed_tasks[task.task_id] = task
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.error_message = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # リトライ
                task.status = TaskStatus.PENDING
                heapq.heappush(self.task_queue, task)
                logger.info(f"Task {task.task_id} scheduled for retry ({task.retry_count}/{task.max_retries})")
            else:
                # 失敗
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                self.completed_tasks[task.task_id] = task
        
        finally:
            # 実行中タスクから削除
            self.running_tasks.pop(task.task_id, None)
            self._save_state()
    
    def cancel_task(self, task_id: str) -> bool:
        """タスクをキャンセル"""
        # スケジュールされたジョブを削除
        for job_id in [f"cron_{task_id}", f"interval_{task_id}", f"scheduled_{task_id}"]:
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass
        
        # キューから削除
        self.task_queue = [t for t in self.task_queue if t.task_id != task_id]
        heapq.heapify(self.task_queue)
        
        # 実行中タスクをキャンセル
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self.completed_tasks[task_id] = task
            self.running_tasks.pop(task_id, None)
            self._save_state()
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """タスクのステータスを取得"""
        # 実行中タスク
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        
        # 完了タスク
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # キュー内のタスク
        for task in self.task_queue:
            if task.task_id == task_id:
                return task
        
        return None
    
    def get_all_tasks(self) -> Dict[str, List[ScheduledTask]]:
        """すべてのタスクを取得"""
        return {
            "queued": sorted(self.task_queue, key=lambda t: t.priority.value),
            "running": list(self.running_tasks.values()),
            "completed": list(self.completed_tasks.values())
        }
    
    def _save_state(self):
        """スケジューラーの状態を保存"""
        state = {
            "task_queue": [t.to_dict() for t in self.task_queue],
            "running_tasks": {k: v.to_dict() for k, v in self.running_tasks.items()},
            "completed_tasks": {k: v.to_dict() for k, v in self.completed_tasks.items()}
        }
        
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """スケジューラーの状態を読み込み"""
        if not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                state = json.load(f)
            
            # 状態を復元（実装は簡略化）
            logger.info(f"Loaded scheduler state with {len(state.get('task_queue', []))} queued tasks")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")

# グローバルスケジューラーインスタンス
scheduler = ContinualLearningScheduler()
