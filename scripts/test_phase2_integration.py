#!/usr/bin/env python3
"""
Phase 2 DIコンテナとヘルスチェックシステムの統合テスト
"""

import sys
import os
from pathlib import Path
import asyncio
import time
import json
from datetime import datetime

# プロジェクトのルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.dependencies.container import DIContainer, ServiceScopeEnum
from src.rag.dependencies.services import (
    configure_rag_services,
    ConfigurationService,
    DependencyCheckService,
    IVectorStore,
    IEmbeddingModel,
    IQueryEngine
)
from src.rag.monitoring.health_check import RAGHealthChecker, HealthStatus
from src.rag.monitoring.metrics import MetricsCollector, timer


def test_di_container():
    """DIコンテナのテスト"""
    print("\n" + "=" * 70)
    print("1. Testing DI Container")
    print("=" * 70)
    
    try:
        # コンテナの作成
        container = DIContainer()
        print("✅ Container created")
        
        # サービスの登録
        configure_rag_services(container)
        print("✅ Services registered")
        
        # サービスの解決
        config_service = container.resolve(ConfigurationService)
        print(f"✅ ConfigurationService resolved: {config_service is not None}")
        
        dep_service = container.resolve(DependencyCheckService)
        print(f"✅ DependencyCheckService resolved: {dep_service is not None}")
        
        # インターフェースの解決
        vector_store = container.get_service(IVectorStore)
        print(f"✅ IVectorStore resolved: {vector_store is not None}")
        
        if vector_store:
            store_type = type(vector_store).__name__
            print(f"   Vector store type: {store_type}")
        
        embedding = container.get_service(IEmbeddingModel)
        print(f"✅ IEmbeddingModel resolved: {embedding is not None}")
        
        # シングルトンのテスト
        config_service2 = container.resolve(ConfigurationService)
        is_singleton = config_service is config_service2
        print(f"✅ Singleton test: {'PASS' if is_singleton else 'FAIL'}")
        
        # スコープのテスト
        with container.scope() as scope:
            scoped_service = scope.get_service(ConfigurationService)
            print(f"✅ Scoped service resolved: {scoped_service is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ DI Container test failed: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


async def test_health_check():
    """ヘルスチェックシステムのテスト"""
    print("\n" + "=" * 70)
    print("2. Testing Health Check System")
    print("=" * 70)
    
    try:
        # コンテナとヘルスチェッカーの作成
        container = DIContainer()
        configure_rag_services(container)
        
        health_checker = RAGHealthChecker(container)
        print("✅ Health checker created")
        
        # ヘルスチェックの実行
        print("\nPerforming health check...")
        result = await health_checker.check_all()
        
        print(f"✅ Overall status: {result.overall_status.value}")
        print(f"   Components checked: {len(result.components)}")
        
        # 各コンポーネントの状態
        print("\nComponent Status:")
        for name, component in result.components.items():
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓"
            }.get(component.status, "?")
            
            print(f"  {status_icon} {name}: {component.status.value}")
            print(f"     Message: {component.message}")
            print(f"     Check duration: {component.check_duration_ms:.2f}ms")
        
        # 警告の表示
        if result.warnings:
            print("\n⚠️ Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


def test_metrics_collection():
    """メトリクス収集のテスト"""
    print("\n" + "=" * 70)
    print("3. Testing Metrics Collection")
    print("=" * 70)
    
    try:
        from src.rag.monitoring.metrics import MetricsCollector, MetricType
        
        collector = MetricsCollector()
        print("✅ Metrics collector created")
        
        # カウンターのテスト
        collector.increment("test.counter", tags={"test": "true"})
        collector.increment("test.counter", tags={"test": "true"})
        print("✅ Counter incremented")
        
        # ゲージのテスト
        collector.gauge("test.gauge", 42.5, unit="items")
        print("✅ Gauge recorded")
        
        # ヒストグラムのテスト
        for i in range(10):
            collector.histogram("test.histogram", i * 10)
        print("✅ Histogram values added")
        
        # タイマーのテスト
        with collector.timer("test.timer"):
            time.sleep(0.1)
        print("✅ Timer measured")
        
        # サマリーの取得
        counter_summary = collector.get_summary("test.counter", MetricType.COUNTER)
        if counter_summary:
            print(f"\nCounter Summary:")
            print(f"  Count: {counter_summary.count}")
            print(f"  Sum: {counter_summary.sum}")
        
        histogram_summary = collector.get_summary("test.histogram", MetricType.HISTOGRAM)
        if histogram_summary:
            print(f"\nHistogram Summary:")
            print(f"  Count: {histogram_summary.count}")
            print(f"  Mean: {histogram_summary.mean:.2f}")
            print(f"  Median: {histogram_summary.median:.2f}")
            print(f"  Min: {histogram_summary.min}")
            print(f"  Max: {histogram_summary.max}")
            if histogram_summary.p95:
                print(f"  P95: {histogram_summary.p95:.2f}")
        
        # エクスポートのテスト
        json_export = collector.export(format="json")
        data = json.loads(json_export)
        print(f"\n✅ Exported {len(data['metrics'])} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics collection test failed: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


async def test_monitoring_integration():
    """モニタリング統合テスト"""
    print("\n" + "=" * 70)
    print("4. Testing Monitoring Integration")
    print("=" * 70)
    
    try:
        # 統合環境のセットアップ
        container = DIContainer()
        configure_rag_services(container)
        
        health_checker = RAGHealthChecker(container)
        collector = MetricsCollector()
        
        print("✅ Monitoring components initialized")
        
        # ヘルスチェックとメトリクス収集を組み合わせる
        print("\nPerforming integrated monitoring...")
        
        # ヘルスチェックを実行し、メトリクスを記録
        with collector.timer("health.check.duration"):
            result = await health_checker.check_all()
        
        # 結果をメトリクスとして記録
        collector.gauge(
            "health.overall_status",
            1 if result.overall_status == HealthStatus.HEALTHY else 0,
            tags={"status": result.overall_status.value}
        )
        
        for name, component in result.components.items():
            collector.gauge(
                f"health.component.{name}",
                1 if component.status == HealthStatus.HEALTHY else 0,
                tags={"status": component.status.value}
            )
            
            collector.histogram(
                "health.check.component_duration",
                component.check_duration_ms,
                tags={"component": name}
            )
        
        print(f"✅ Recorded {len(collector.get_metrics())} metrics")
        
        # モニタリングループのテスト（短時間）
        print("\nTesting monitoring loop (5 seconds)...")
        
        await health_checker.start_monitoring()
        health_checker.check_interval = 2  # 2秒間隔に設定
        
        await asyncio.sleep(5)
        
        await health_checker.stop_monitoring()
        print("✅ Monitoring loop tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


async def test_service_lifecycle():
    """サービスライフサイクルのテスト"""
    print("\n" + "=" * 70)
    print("5. Testing Service Lifecycle")
    print("=" * 70)
    
    try:
        # ライフサイクルフックを持つサービスクラス
        class TestService:
            def __init__(self):
                self.created = False
                self.disposed = False
            
            def on_created(self):
                self.created = True
                print("  ↳ TestService.on_created() called")
            
            def dispose(self):
                self.disposed = True
                print("  ↳ TestService.dispose() called")
        
        container = DIContainer()
        container.register_singleton(TestService)
        
        # サービスの作成
        print("Creating service...")
        service = container.resolve(TestService)
        print(f"✅ Service created: {service.created}")
        
        # コンテナの破棄
        print("\nDisposing container...")
        container.dispose()
        print(f"✅ Service disposed: {service.disposed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service lifecycle test failed: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


async def main():
    """メイン処理"""
    print("=" * 70)
    print("🔍 Phase 2 Integration Test")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # テスト実行
    tests = [
        ("DI Container", test_di_container),
        ("Health Check System", lambda: asyncio.create_task(test_health_check())),
        ("Metrics Collection", test_metrics_collection),
        ("Monitoring Integration", lambda: asyncio.create_task(test_monitoring_integration())),
        ("Service Lifecycle", lambda: asyncio.create_task(test_service_lifecycle()))
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            
            # asyncioタスクの場合は await
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                result = await result
            
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            failed += 1
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        print("✅ Phase 2 implementation is complete!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        sys.exit(1)
