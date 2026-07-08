"""System monitoring, health, and metrics endpoints."""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

import psutil
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ..dependencies import logger, model_cache, training_tasks
import app.dependencies as _deps

router = APIRouter(tags=["monitoring"])


@router.get("/api/system-info")
async def get_system_info():
    """システム情報を取得"""
    try:
        # GPU情報
        gpu_info = []
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_memory_free = gpu_memory_total - gpu_memory_used

                    gpu_info.append({
                        "device": i,
                        "name": gpu_name,
                        "memory": f"{gpu_memory_total:.1f}GB",
                        "memory_used": f"{gpu_memory_used:.1f}GB",
                        "memory_free": f"{gpu_memory_free:.1f}GB",
                        "available": True,
                    })
            else:
                gpu_info = [{"device": 0, "name": "No GPU", "memory": "0GB", "available": False}]
        else:
            gpu_info = [{"device": 0, "name": "CUDA Not Available", "memory": "0GB", "available": False}]

        cuda_info = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda if torch.cuda.is_available() else "Not Available",
        }

        pytorch_info = {"version": torch.__version__}

        cpu_info = {"name": "CPU", "cores": os.cpu_count()}

        memory = psutil.virtual_memory()
        ram_info = {
            "total": f"{memory.total / (1024**3):.1f}GB",
            "used": f"{memory.used / (1024**3):.1f}GB",
            "free": f"{memory.available / (1024**3):.1f}GB",
            "percent": f"{memory.percent:.1f}%",
        }

        cache_info = {
            "status": f"{len(model_cache)} models cached" if len(model_cache) > 0 else "No models cached"
        }

        return {
            "gpu": gpu_info,
            "cuda": cuda_info,
            "pytorch": pytorch_info,
            "cpu": cpu_info,
            "ram": ram_info,
            "cache": cache_info,
        }
    except Exception as e:
        logger.error(f"システム情報取得エラー: {e}")
        return {
            "error": str(e),
            "gpu": {"name": "Error", "memory": "Unknown", "available": False},
            "cuda": {"available": False, "version": "Unknown"},
            "pytorch": {"version": "Unknown"},
            "cpu": {"name": "Unknown", "cores": "Unknown"},
            "ram": {"total": "Unknown", "used": "Unknown", "free": "Unknown", "percent": "Unknown"},
            "cache": {"status": "Unknown"},
        }


@router.get("/metrics")
async def get_metrics():
    """Prometheusメトリクスエンドポイント"""
    if _deps.METRICS_AVAILABLE and _deps.metrics_collector:
        try:
            _deps.metrics_collector.update_system_metrics()

            active_tasks = sum(1 for task in training_tasks.values() if task["status"] == "running")
            _deps.metrics_collector.set_active_training_tasks(active_tasks)

            if _deps.RAG_AVAILABLE:
                try:
                    from src.rag.indexing.metadata_manager import MetadataManager
                    metadata_manager = MetadataManager()
                    doc_count = len(metadata_manager.get_all_documents())
                    _deps.metrics_collector.set_rag_documents_count(doc_count)
                except Exception:
                    pass

            from app.monitoring import get_prometheus_metrics
            return get_prometheus_metrics()
        except Exception as e:
            logger.error(f"メトリクス生成エラー: {e}")
            return Response(content="# Error generating metrics\n", media_type="text/plain")
    else:
        return Response(content="# Metrics not available\n", media_type="text/plain")


@router.post("/api/clear_cache")
async def clear_model_cache():
    """モデルキャッシュをクリア"""
    try:
        cache_size = len(model_cache)
        model_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"モデルキャッシュをクリアしました: {cache_size}個のモデルを解放")
        return {"status": "success", "cleared_models": cache_size}
    except Exception as e:
        logger.error(f"キャッシュクリアエラー: {str(e)}")
        return {"status": "error", "error": str(e)}


@router.post("/api/monitoring/start")
async def start_monitoring():
    """監視システムを起動"""
    try:
        # Web用の監視制御スクリプトを使用
        script_path = Path("/workspace/scripts/web_monitoring_controller.sh")

        if not script_path.exists():
            # スクリプトがない場合は作成
            script_content = '''#!/bin/bash
# 監視システムをコンテナ内から起動するスクリプト
echo "🚀 Grafana監視システムを起動中..."

# Grafanaが起動しているか確認
if curl -s http://ai-ft-grafana:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana: 既に起動しています"
    exit 0
else
    echo "⚠️ Grafana: ホスト側でdocker-compose -f docker/docker-compose-monitoring.yml up -d を実行してください"
    exit 1
fi
'''
            script_path.write_text(script_content)
            os.chmod(script_path, 0o755)

        # スクリプトを実行
        result = subprocess.run(
            [str(script_path), "start"],
            capture_output=True,
            text=True
        )

        # Grafanaの状態を直接確認
        import requests
        grafana_running = False
        prometheus_running = False

        try:
            # Grafana確認（コンテナ間通信）
            resp = requests.get("http://ai-ft-grafana:3000/api/health", timeout=2)
            grafana_running = resp.status_code == 200
        except Exception:
            pass

        try:
            # Prometheus確認（コンテナ間通信）
            resp = requests.get("http://ai-ft-prometheus:9090/-/healthy", timeout=2)
            prometheus_running = resp.status_code == 200
        except Exception:
            pass

        if grafana_running or prometheus_running:
            return JSONResponse(content={
                "status": "success",
                "message": "監視システムが利用可能です",
                "services": {
                    "grafana": "http://localhost:3000" if grafana_running else None,
                    "prometheus": "http://localhost:9090" if prometheus_running else None
                },
                "note": "既に起動済みか、ホスト側で起動されています"
            })
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "監視システムが起動していません。ホスト側で以下のコマンドを実行してください:\ndocker-compose -f docker/docker-compose-monitoring.yml up -d",
                    "command": "docker-compose -f docker/docker-compose-monitoring.yml up -d"
                },
                status_code=503
            )
    except Exception as e:
        logger.error(f"監視システム起動エラー: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


@router.post("/api/monitoring/stop")
async def stop_monitoring():
    """監視システムを停止"""
    try:
        import requests

        # 現在の状態を確認
        grafana_running = False
        prometheus_running = False

        try:
            resp = requests.get("http://ai-ft-grafana:3000/api/health", timeout=2)
            grafana_running = resp.status_code == 200
        except Exception:
            pass

        try:
            resp = requests.get("http://ai-ft-prometheus:9090/-/healthy", timeout=2)
            prometheus_running = resp.status_code == 200
        except Exception:
            pass

        if not grafana_running and not prometheus_running:
            return JSONResponse(content={
                "status": "success",
                "message": "監視システムは既に停止しています"
            })

        # コンテナ内から停止コマンドを実行できないため、手順を案内
        return JSONResponse(content={
            "status": "info",
            "message": "監視システムを停止するには、ホスト側で以下のコマンドを実行してください",
            "command": "docker stop ai-ft-grafana ai-ft-prometheus ai-ft-redis",
            "alternative": "または: docker-compose -f docker/docker-compose-monitoring.yml down"
        })

    except Exception as e:
        logger.error(f"監視システム停止エラー: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


@router.get("/api/monitoring/status")
async def monitoring_status():
    """監視システムの状態を確認"""
    try:
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        compose_file = docker_dir / "docker-compose-monitoring.yml"

        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "ps", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(docker_dir)
        )

        if result.returncode == 0:
            services_running = "grafana" in result.stdout.lower()
            return JSONResponse(content={
                "status": "success",
                "running": services_running,
                "message": "監視システムは稼働中" if services_running else "監視システムは停止中"
            })
        else:
            return JSONResponse(content={
                "status": "success",
                "running": False,
                "message": "監視システムは停止中"
            })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "running": False,
            "message": str(e)
        })
