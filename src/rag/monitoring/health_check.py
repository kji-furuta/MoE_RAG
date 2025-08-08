"""
ヘルスチェックシステム

各コンポーネントの健全性を監視し、
問題を早期に検出するためのシステム。
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import psutil
import torch
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """ヘルスステータス"""
    HEALTHY = "healthy"        # 正常
    DEGRADED = "degraded"      # 劣化（一部機能制限）
    UNHEALTHY = "unhealthy"    # 異常
    UNKNOWN = "unknown"        # 不明


@dataclass
class ComponentHealth:
    """コンポーネントのヘルス情報"""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    check_duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "check_duration_ms": self.check_duration_ms,
            "metadata": self.metadata
        }


@dataclass
class HealthCheckResult:
    """ヘルスチェック結果"""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "overall_status": self.overall_status.value,
            "components": {
                name: comp.to_dict() 
                for name, comp in self.components.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "warnings": self.warnings
        }
    
    def is_healthy(self) -> bool:
        """健全かどうか"""
        return self.overall_status == HealthStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        """劣化しているかどうか"""
        return self.overall_status == HealthStatus.DEGRADED


class HealthChecker:
    """個別のヘルスチェッカーの基底クラス"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def check(self) -> ComponentHealth:
        """ヘルスチェックを実行"""
        start_time = datetime.now()
        
        try:
            status, message, metadata = await self._perform_check()
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                last_check=datetime.now(),
                metadata=metadata,
                check_duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                last_check=datetime.now(),
                check_duration_ms=duration_ms
            )
    
    async def _perform_check(self) -> tuple:
        """実際のチェックを実行（サブクラスで実装）"""
        raise NotImplementedError


class SystemHealthChecker(HealthChecker):
    """システムリソースのヘルスチェッカー"""
    
    def __init__(self):
        super().__init__("system")
        self.cpu_threshold = 90  # CPU使用率の閾値
        self.memory_threshold = 90  # メモリ使用率の閾値
        self.disk_threshold = 90  # ディスク使用率の閾値
    
    async def _perform_check(self) -> tuple:
        """システムリソースをチェック"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        issues = []
        metadata = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # CPU チェック
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent}%")
        
        # メモリチェック
        if memory.percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory.percent}%")
        
        # ディスクチェック
        if disk.percent > self.disk_threshold:
            issues.append(f"Low disk space: {disk.percent}% used")
        
        if issues:
            return HealthStatus.DEGRADED, " | ".join(issues), metadata
        else:
            return HealthStatus.HEALTHY, "System resources are healthy", metadata


class GPUHealthChecker(HealthChecker):
    """GPUのヘルスチェッカー"""
    
    def __init__(self):
        super().__init__("gpu")
        self.memory_threshold_gb = 2  # 最小必要メモリ (GB)
    
    async def _perform_check(self) -> tuple:
        """GPU状態をチェック"""
        if not torch.cuda.is_available():
            return HealthStatus.UNHEALTHY, "GPU not available", {"available": False}
        
        metadata = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": []
        }
        
        issues = []
        
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            mem_info = torch.cuda.mem_get_info(i)
            free_gb = mem_info[0] / (1024**3)
            total_gb = mem_info[1] / (1024**3)
            used_percent = ((total_gb - free_gb) / total_gb) * 100
            
            device_info = {
                "index": i,
                "name": device_name,
                "free_memory_gb": free_gb,
                "total_memory_gb": total_gb,
                "used_percent": used_percent
            }
            metadata["devices"].append(device_info)
            
            if free_gb < self.memory_threshold_gb:
                issues.append(f"GPU {i}: Low memory ({free_gb:.2f}GB free)")
        
        if issues:
            return HealthStatus.DEGRADED, " | ".join(issues), metadata
        else:
            return HealthStatus.HEALTHY, "GPU resources are healthy", metadata


class VectorStoreHealthChecker(HealthChecker):
    """ベクトルストアのヘルスチェッカー"""
    
    def __init__(self, vector_store=None):
        super().__init__("vector_store")
        self.vector_store = vector_store
    
    async def _perform_check(self) -> tuple:
        """ベクトルストアの状態をチェック"""
        if not self.vector_store:
            return HealthStatus.UNKNOWN, "Vector store not configured", {}
        
        try:
            # Qdrantの場合
            if hasattr(self.vector_store, 'client'):
                from qdrant_client import QdrantClient
                
                if isinstance(self.vector_store.client, QdrantClient):
                    collections = self.vector_store.client.get_collections()
                    
                    metadata = {
                        "type": "qdrant",
                        "collections_count": len(collections.collections),
                        "collections": [c.name for c in collections.collections]
                    }
                    
                    return HealthStatus.HEALTHY, "Qdrant is operational", metadata
            
            # インメモリの場合
            if hasattr(self.vector_store, 'documents'):
                metadata = {
                    "type": "in_memory",
                    "document_count": len(self.vector_store.documents)
                }
                
                return HealthStatus.HEALTHY, "In-memory store is operational", metadata
            
            return HealthStatus.UNKNOWN, "Unknown vector store type", {}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Vector store error: {str(e)}", {}


class DependencyHealthChecker(HealthChecker):
    """依存関係のヘルスチェッカー"""
    
    def __init__(self, dependency_service=None):
        super().__init__("dependencies")
        self.dependency_service = dependency_service
    
    async def _perform_check(self) -> tuple:
        """依存関係の状態をチェック"""
        if not self.dependency_service:
            return HealthStatus.UNKNOWN, "Dependency service not configured", {}
        
        try:
            result = self.dependency_service.check_dependencies(use_cache=True)
            
            metadata = {
                "can_run": result.can_run,
                "missing_core": len(result.missing_core),
                "missing_infrastructure": len(result.missing_infrastructure),
                "missing_optional": len(result.missing_optional),
                "alternatives_used": len(result.alternatives_used)
            }
            
            if not result.can_run:
                missing = result.missing_core + result.missing_infrastructure
                return HealthStatus.UNHEALTHY, f"Missing critical dependencies: {', '.join(missing[:3])}", metadata
            
            if result.alternatives_used or result.missing_optional:
                return HealthStatus.DEGRADED, "Using alternative packages or missing optional dependencies", metadata
            
            if result.is_satisfied:
                return HealthStatus.HEALTHY, "All dependencies are satisfied", metadata
            
            return HealthStatus.DEGRADED, "Some dependencies are missing", metadata
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Dependency check error: {str(e)}", {}


class RAGHealthChecker:
    """RAGシステム全体のヘルスチェッカー"""
    
    def __init__(self, container=None):
        """
        Args:
            container: DIコンテナ（サービスの取得用）
        """
        self.container = container
        self.checkers: List[HealthChecker] = []
        self.check_interval = 60  # 秒
        self.last_result: Optional[HealthCheckResult] = None
        self._running = False
        self._task = None
        
        # デフォルトのチェッカーを登録
        self._register_default_checkers()
    
    def _register_default_checkers(self):
        """デフォルトのヘルスチェッカーを登録"""
        self.register(SystemHealthChecker())
        self.register(GPUHealthChecker())
        
        # コンテナからサービスを取得してチェッカーを登録
        if self.container:
            try:
                from ..dependencies.services import IVectorStore, DependencyCheckService
                
                vector_store = self.container.get_service(IVectorStore)
                if vector_store:
                    self.register(VectorStoreHealthChecker(vector_store))
                
                dep_service = self.container.get_service(DependencyCheckService)
                if dep_service:
                    self.register(DependencyHealthChecker(dep_service))
                    
            except Exception as e:
                logger.warning(f"Failed to register service checkers: {e}")
    
    def register(self, checker: HealthChecker):
        """ヘルスチェッカーを登録"""
        self.checkers.append(checker)
        logger.debug(f"Registered health checker: {checker.name}")
    
    async def check_all(self) -> HealthCheckResult:
        """全コンポーネントのヘルスチェック"""
        components = {}
        warnings = []
        
        # 並列でチェックを実行
        tasks = [checker.check() for checker in self.checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for checker, result in zip(self.checkers, results):
            if isinstance(result, Exception):
                # エラーの場合
                components[checker.name] = ComponentHealth(
                    name=checker.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(result)}",
                    last_check=datetime.now()
                )
                warnings.append(f"{checker.name}: {str(result)}")
            else:
                components[checker.name] = result
                
                # 警告を収集
                if result.status == HealthStatus.DEGRADED:
                    warnings.append(f"{checker.name}: {result.message}")
        
        # 全体のステータスを決定
        overall_status = self._determine_overall_status(components)
        
        result = HealthCheckResult(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.now(),
            warnings=warnings
        )
        
        self.last_result = result
        return result
    
    def _determine_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """全体のステータスを決定"""
        if not components:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in components.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def start_monitoring(self):
        """定期的なモニタリングを開始"""
        if self._running:
            logger.warning("Monitoring is already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def _monitoring_loop(self):
        """モニタリングループ"""
        while self._running:
            try:
                result = await self.check_all()
                
                # 結果をログ
                if result.overall_status == HealthStatus.UNHEALTHY:
                    logger.error(f"System unhealthy: {result.warnings}")
                elif result.overall_status == HealthStatus.DEGRADED:
                    logger.warning(f"System degraded: {result.warnings}")
                else:
                    logger.debug("System healthy")
                
                # 結果を保存（オプション）
                await self._save_result(result)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            # 次のチェックまで待機
            await asyncio.sleep(self.check_interval)
    
    async def stop_monitoring(self):
        """モニタリングを停止"""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _save_result(self, result: HealthCheckResult):
        """結果を保存（オプション）"""
        try:
            # ファイルに保存
            log_dir = Path("logs/health")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"health_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # 古いログを削除（7日以上前）
            cutoff = datetime.now() - timedelta(days=7)
            for file in log_dir.glob("health_*.json"):
                if file.stat().st_mtime < cutoff.timestamp():
                    file.unlink()
                    
        except Exception as e:
            logger.debug(f"Failed to save health result: {e}")
    
    def get_last_result(self) -> Optional[HealthCheckResult]:
        """最後のチェック結果を取得"""
        return self.last_result
