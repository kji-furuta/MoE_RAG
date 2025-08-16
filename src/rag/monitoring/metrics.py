"""
メトリクス収集システム

パフォーマンスメトリクスを収集し、
システムの最適化に活用するための機能。
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import time
import statistics
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """メトリクスタイプ"""
    COUNTER = "counter"      # カウンター（増加のみ）
    GAUGE = "gauge"          # ゲージ（増減あり）
    HISTOGRAM = "histogram"  # ヒストグラム（分布）
    TIMER = "timer"          # タイマー（実行時間）


@dataclass
class Metric:
    """メトリクス"""
    name: str
    type: MetricType
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class MetricSummary:
    """メトリクスサマリー"""
    name: str
    type: MetricType
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: Optional[float] = None
    stddev: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "type": self.type.value,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "stddev": self.stddev,
            "p95": self.p95,
            "p99": self.p99
        }


class MetricStore:
    """メトリクスストア"""
    
    def __init__(self, max_size: int = 10000, ttl_minutes: int = 60):
        """
        Args:
            max_size: 最大保存数
            ttl_minutes: 保存期間（分）
        """
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.metrics: deque = deque(maxlen=max_size)
    
    def add(self, metric: Metric):
        """メトリクスを追加"""
        self.metrics.append(metric)
        self._cleanup()
    
    def _cleanup(self):
        """古いメトリクスを削除"""
        cutoff = datetime.now() - self.ttl
        
        while self.metrics and self.metrics[0].timestamp < cutoff:
            self.metrics.popleft()
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        type: Optional[MetricType] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None
    ) -> List[Metric]:
        """メトリクスを取得"""
        self._cleanup()
        
        results = []
        for metric in self.metrics:
            # フィルタリング
            if name and metric.name != name:
                continue
            if type and metric.type != type:
                continue
            if tags and not all(metric.tags.get(k) == v for k, v in tags.items()):
                continue
            if since and metric.timestamp < since:
                continue
            
            results.append(metric)
        
        return results
    
    def get_summary(self, name: str, type: MetricType) -> Optional[MetricSummary]:
        """メトリクスのサマリーを取得"""
        metrics = self.get_metrics(name=name, type=type)
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        summary = MetricSummary(
            name=name,
            type=type,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values)
        )
        
        if len(values) > 1:
            summary.median = statistics.median(values)
            summary.stddev = statistics.stdev(values)
            
            # パーセンタイル
            sorted_values = sorted(values)
            p95_index = int(len(sorted_values) * 0.95)
            p99_index = int(len(sorted_values) * 0.99)
            
            summary.p95 = sorted_values[p95_index] if p95_index < len(sorted_values) else sorted_values[-1]
            summary.p99 = sorted_values[p99_index] if p99_index < len(sorted_values) else sorted_values[-1]
        
        return summary


class MetricsCollector:
    """メトリクス収集器"""
    
    def __init__(self, store: Optional[MetricStore] = None):
        """
        Args:
            store: メトリクスストア
        """
        self.store = store or MetricStore()
        self._timers = {}
    
    def increment(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """カウンターをインクリメント"""
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.store.add(metric)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = None):
        """ゲージを記録"""
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.store.add(metric)
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """ヒストグラムに値を追加"""
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.store.add(metric)
    
    def timer_start(self, name: str) -> str:
        """タイマーを開始"""
        timer_id = f"{name}_{time.time()}"
        self._timers[timer_id] = time.perf_counter()
        return timer_id
    
    def timer_end(self, timer_id: str, tags: Dict[str, str] = None):
        """タイマーを終了"""
        if timer_id not in self._timers:
            logger.warning(f"Timer {timer_id} not found")
            return
        
        start_time = self._timers.pop(timer_id)
        duration = time.perf_counter() - start_time
        
        name = timer_id.rsplit('_', 1)[0]
        
        metric = Metric(
            name=name,
            type=MetricType.TIMER,
            value=duration * 1000,  # ミリ秒
            timestamp=datetime.now(),
            tags=tags or {},
            unit="ms"
        )
        self.store.add(metric)
    
    def timer(self, name: str):
        """タイマーのコンテキストマネージャー"""
        class Timer:
            def __init__(self, collector, name):
                self.collector = collector
                self.name = name
                self.timer_id = None
            
            def __enter__(self):
                self.timer_id = self.collector.timer_start(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.timer_id:
                    self.collector.timer_end(self.timer_id)
        
        return Timer(self, name)
    
    def get_metrics(self, **kwargs) -> List[Metric]:
        """メトリクスを取得"""
        return self.store.get_metrics(**kwargs)
    
    def get_summary(self, name: str, type: MetricType) -> Optional[MetricSummary]:
        """サマリーを取得"""
        return self.store.get_summary(name, type)
    
    def export(self, format: str = "json") -> str:
        """メトリクスをエクスポート"""
        metrics = self.store.get_metrics()
        
        if format == "json":
            data = {
                "metrics": [m.to_dict() for m in metrics],
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(data, indent=2)
        
        elif format == "prometheus":
            # Prometheus形式
            lines = []
            for metric in metrics:
                tags_str = ",".join(f'{k}="{v}"' for k, v in metric.tags.items())
                if tags_str:
                    tags_str = f"{{{tags_str}}}"
                
                lines.append(f"{metric.name}{tags_str} {metric.value} {int(metric.timestamp.timestamp() * 1000)}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_to_file(self, filepath: Path, format: str = "json"):
        """メトリクスをファイルに保存"""
        data = self.export(format=format)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(data)


# グローバルコレクター
_global_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """グローバルコレクターを取得"""
    return _global_collector


# 便利な関数
def increment(name: str, value: float = 1, tags: Dict[str, str] = None):
    """カウンターをインクリメント"""
    _global_collector.increment(name, value, tags)


def gauge(name: str, value: float, tags: Dict[str, str] = None, unit: str = None):
    """ゲージを記録"""
    _global_collector.gauge(name, value, tags, unit)


def histogram(name: str, value: float, tags: Dict[str, str] = None):
    """ヒストグラムに値を追加"""
    _global_collector.histogram(name, value, tags)


def timer(name: str):
    """タイマーのデコレーター/コンテキストマネージャー"""
    return _global_collector.timer(name)


# デコレーター
def timed(name: Optional[str] = None):
    """関数の実行時間を計測するデコレーター"""
    def decorator(func):
        metric_name = name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            with timer(metric_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
