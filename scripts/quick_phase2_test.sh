#!/bin/bash
# Phase 2の主要機能の簡易動作確認

echo "======================================================================"
echo "🧪 Phase 2 Quick Validation Test"
echo "======================================================================"
echo ""

# DIコンテナのテスト
echo "1. Testing DI Container..."
docker exec ai-ft-container python3 -c "
import sys
sys.path.insert(0, '/workspace')

from src.rag.dependencies.container import DIContainer, ServiceScopeEnum
from src.rag.dependencies.services import ConfigurationService

container = DIContainer()
container.register_singleton(ConfigurationService)
service = container.resolve(ConfigurationService)
print(f'✅ DI Container: Working')
print(f'   Service resolved: {service is not None}')
"

echo ""
echo "2. Testing Health Check..."
docker exec ai-ft-container python3 -c "
import sys
import asyncio
sys.path.insert(0, '/workspace')

from src.rag.monitoring.health_check import RAGHealthChecker, HealthStatus

async def test():
    checker = RAGHealthChecker()
    result = await checker.check_all()
    print(f'✅ Health Check: Working')
    print(f'   Overall status: {result.overall_status.value}')
    print(f'   Components checked: {len(result.components)}')
    return result

asyncio.run(test())
"

echo ""
echo "3. Testing Metrics Collection..."
docker exec ai-ft-container python3 -c "
import sys
sys.path.insert(0, '/workspace')

from src.rag.monitoring.metrics import MetricsCollector, MetricType

collector = MetricsCollector()
collector.increment('test.counter')
collector.gauge('test.gauge', 42.5)

summary = collector.get_summary('test.counter', MetricType.COUNTER)
print(f'✅ Metrics Collection: Working')
print(f'   Metrics collected: {len(collector.get_metrics())}')
"

echo ""
echo "======================================================================"
echo "✅ Quick Validation Complete"
echo "======================================================================"
