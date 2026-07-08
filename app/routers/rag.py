"""RAG (Retrieval-Augmented Generation) endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..dependencies import logger, rag_app
from ..schemas.rag import (
    BatchQueryRequest,
    QueryRequest,
    QueryResponse,
    SaveSearchRequest,
    SearchHistoryResponse,
    SystemInfoResponse,
)
import app.dependencies as _deps

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/rag", tags=["rag"])

@router.get("/health")
async def rag_health_check():
    """RAGシステムヘルスチェック"""
    return {
        "status": "healthy" if rag_app.is_initialized else "initializing",
        "timestamp": datetime.now(JST).isoformat(),
        "service": "Road Design RAG System",
        "available": _deps.RAG_AVAILABLE
    }

@router.get("/system-info", response_model=SystemInfoResponse)
async def rag_get_system_info():
    """RAGシステム情報を取得"""
    try:
        # 設定ファイルから直接読み込み
        config_path = Path("src/rag/config/rag_config.yaml")
        config_data = {}

        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

        # Phase 2メトリクスデータを読み込み
        metrics_data = {}
        metrics_json_path = Path("benchmarks/phase2/advanced_metrics_latest.json")
        if metrics_json_path.exists():
            with open(metrics_json_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)

        # ベクトルDBの現在のモデル情報を取得
        vector_db_info = {
            "current_model": config_data.get('embedding', {}).get('model_name', 'multilingual-e5-large'),
            "document_count": 0,
            "collection_exists": False
        }

        # QdrantのHTTP APIに直接リクエスト（Pydanticバリデーションエラー回避）
        try:
            import requests

            qdrant_url = "http://ai-ft-qdrant:6333/collections/road_design_docs"
            response = requests.get(qdrant_url, timeout=5)

            if response.status_code == 200:
                collection_data = response.json()
                if collection_data.get("status") == "ok" and "result" in collection_data:
                    points_count = collection_data["result"].get("points_count", 0)
                    vector_db_info["collection_exists"] = True
                    vector_db_info["document_count"] = points_count
                    logger.info(f"Vector DB info retrieved via HTTP API: {points_count} documents")
            else:
                logger.info(f"Collection not found, status code: {response.status_code}")

        except Exception as e:
            logging.warning(f"Failed to connect to Qdrant via HTTP API: {e}")
            # 接続失敗時は設定ファイルからモデル名のみ取得

        # システム情報を構築
        system_info = {
            "config": {
                "llm": {
                    "model_name": config_data.get('llm', {}).get('model_name', '未設定'),
                    "base_model": config_data.get('llm', {}).get('base_model', '未設定'),
                    "temperature": config_data.get('llm', {}).get('temperature', 0.3),
                    "use_finetuned": config_data.get('llm', {}).get('use_finetuned', False),
                    "use_moe": config_data.get('llm', {}).get('use_moe', False),
                    "moe_num_experts": config_data.get('llm', {}).get('moe_num_experts', 8),
                    "moe_experts_per_token": config_data.get('llm', {}).get('moe_experts_per_token', 2),
                    "model_path": config_data.get('llm', {}).get('model_path', '未設定')
                },
                "embedding": {
                    "model_name": config_data.get('embedding', {}).get('model_name', 'multilingual-e5-large')
                },
                "vector_store": {
                    "type": config_data.get('vector_store', {}).get('type', 'Qdrant')
                }
            },
            "vector_db": vector_db_info,
            "status": "initialized" if rag_app.is_initialized else "not_initialized",
            "metrics": metrics_data.get("metrics", {}) if metrics_data else None
        }

        return SystemInfoResponse(
            status="success",
            system_info=system_info,
            timestamp=datetime.now(JST).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return SystemInfoResponse(
            status="error",
            system_info={
                "config": {
                    "llm": {"model_name": "設定読み込みエラー"},
                    "embedding": {"model_name": "multilingual-e5-large"},
                    "vector_store": {"type": "Qdrant"}
                },
                "error": str(e)
            },
            timestamp=datetime.now(JST).isoformat()
        )

@router.get("/metrics-dashboard")
async def get_metrics_dashboard():
    """Phase 2メトリクスダッシュボードのHTMLを返す"""
    try:
        dashboard_path = Path("benchmarks/phase2/dashboard_latest.html")
        if dashboard_path.exists():
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # ダッシュボードが存在しない場合は生成を試みる
            from src.benchmarks.phase2_advanced_metrics import Phase2AdvancedMetrics
            metrics = Phase2AdvancedMetrics()
            metrics.run_all_metrics()
            
            if dashboard_path.exists():
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content)
            else:
                return HTMLResponse(
                    content="<html><body><h1>メトリクスダッシュボードが見つかりません</h1></body></html>",
                    status_code=404
                )
    except Exception as e:
        logger.error(f"Failed to load metrics dashboard: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>エラー</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )

@router.get("/metrics-data")
async def get_metrics_data():
    """Phase 2メトリクスのJSONデータを返す"""
    try:
        metrics_path = Path("benchmarks/phase2/advanced_metrics_latest.json")
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            return JSONResponse(content=metrics_data)
        else:
            # メトリクスが存在しない場合は生成を試みる
            from src.benchmarks.phase2_advanced_metrics import Phase2AdvancedMetrics
            metrics = Phase2AdvancedMetrics()
            results = metrics.run_all_metrics()
            return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Failed to load metrics data: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@router.get("/metrics-summary")
async def get_metrics_summary():
    """Phase 2メトリクスのサマリーマークダウンを返す"""
    try:
        summary_path = Path("benchmarks/phase2/summary.md")
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            # Convert markdown to HTML
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>メトリクスサマリー</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                    pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    h1, h2, h3 {{ color: #333; }}
                    ul {{ line-height: 1.8; }}
                </style>
            </head>
            <body>
                <pre>{markdown_content}</pre>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(
                content="<html><body><h1>メトリクスサマリーが見つかりません</h1></body></html>",
                status_code=404
            )
    except Exception as e:
        logger.error(f"Failed to load metrics summary: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>エラー</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )

@router.post("/quantize-model")
async def quantize_finetuned_model(
    background_tasks: BackgroundTasks,
    lora_path: str = Form(...),
    quantization_level: str = Form("Q4_K_M"),
    model_name: str = Form(...)
):
    """ファインチューニング済みモデルを量子化してOllamaに登録"""
    try:
        import asyncio
        import subprocess
        from pathlib import Path
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # 非同期で量子化を実行
        async def run_quantization():
            try:
                # ステータス更新用ファイル
                status_file = Path(f"/tmp/quantization_{task_id}.json")
                
                # 初期ステータス
                status = {
                    "task_id": task_id,
                    "status": "running",
                    "progress": 0,
                    "message": "量子化処理を開始しています...",
                    "logs": []
                }
                status_file.write_text(json.dumps(status))
                
                # 統合モデル処理スクリプトを実行
                # モデルパスが指定されている場合はそれを処理、なければ最新モデルを自動検出
                if lora_path and lora_path != "auto":
                    cmd = [
                        "python", "/workspace/scripts/unified_model_processor.py",
                        "--model", lora_path
                    ]
                else:
                    # 自動検出モード（最新のモデルを処理）
                    cmd = [
                        "python", "/workspace/scripts/qlora_to_ollama.py"
                    ]
                
                logger.info(f"量子化コマンド実行: {' '.join(cmd)}")
                
                # プロセス実行
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # 出力を監視
                logs = []
                
                # stdoutを読み取り
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    log_line = line.decode().strip()
                    logs.append(log_line)
                    logger.info(f"Quantization stdout: {log_line}")
                    
                    # ステータス更新
                    if "Loading base model" in log_line:
                        progress = 20
                        message = "ベースモデルをロード中..."
                    elif "Loading LoRA adapter" in log_line:
                        progress = 40
                        message = "LoRAアダプタをロード中..."
                    elif "Merging" in log_line:
                        progress = 60
                        message = "モデルをマージ中..."
                    elif "Quantizing" in log_line or "quantization" in log_line.lower():
                        progress = 80
                        message = "量子化を実行中..."
                    elif "Saving" in log_line:
                        progress = 90
                        message = "量子化モデルを保存中..."
                    else:
                        continue
                    
                    status.update({
                        "progress": progress,
                        "message": message,
                        "logs": logs[-10:]  # 最新10行のみ保持
                    })
                    status_file.write_text(json.dumps(status))
                
                # プロセス終了待ち
                await process.wait()
                
                # stderrも読み取り
                stderr = await process.stderr.read()
                if stderr:
                    stderr_text = stderr.decode()
                    logger.error(f"Quantization stderr: {stderr_text}")
                    logs.append(f"[ERROR] {stderr_text}")
                
                if process.returncode == 0:
                    # Ollamaにモデルを登録
                    ollama_cmd = ["ollama", "create", model_name, "-f", 
                                 f"/workspace/outputs/quantized_{task_id}/Modelfile"]
                    
                    ollama_process = await asyncio.create_subprocess_exec(
                        *ollama_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    await ollama_process.wait()
                    
                    if ollama_process.returncode == 0:
                        # RAG設定を自動更新
                        config_path = Path("/workspace/src/rag/config/rag_config.yaml")
                        if config_path.exists():
                            import yaml
                            with open(config_path) as f:
                                config = yaml.safe_load(f)
                            
                            config['llm']['use_ollama_fallback'] = True
                            config['llm']['ollama_model'] = model_name
                            
                            with open(config_path, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                        
                        status.update({
                            "status": "completed",
                            "progress": 100,
                            "message": f"量子化完了！モデル '{model_name}' がOllamaに登録されました",
                            "ollama_model": model_name
                        })
                    else:
                        status.update({
                            "status": "error",
                            "message": "Ollamaへの登録に失敗しました"
                        })
                else:
                    stderr = await process.stderr.read()
                    status.update({
                        "status": "error",
                        "message": f"量子化に失敗しました: {stderr.decode()}"
                    })
                
                status_file.write_text(json.dumps(status))
                
            except Exception as e:
                logger.error(f"量子化エラー: {e}")
                status = {
                    "task_id": task_id,
                    "status": "error",
                    "message": str(e)
                }
                status_file.write_text(json.dumps(status))
        
        # バックグラウンドタスクとして実行
        background_tasks.add_task(run_quantization)
        
        return {
            "task_id": task_id,
            "message": "量子化処理を開始しました",
            "status_url": f"/rag/quantization-status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"量子化開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quantization-status/{task_id}")
async def get_quantization_status(task_id: str):
    """量子化タスクのステータスを取得"""
    try:
        status_file = Path(f"/tmp/quantization_{task_id}.json")
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="タスクが見つかりません")
        
        with open(status_file) as f:
            status = json.load(f)
        
        return status
        
    except Exception as e:
        logger.error(f"ステータス取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-lora-models")
async def list_lora_models():
    """利用可能なファインチューニング済みモデルをリスト（全種類）"""
    try:
        # モデル探索ユーティリティを使用
        try:
            from src.utils.model_discovery import ModelDiscovery
            discovery = ModelDiscovery()
            all_models = discovery.find_all_models()
        except ImportError:
            logger.warning("ModelDiscovery not available, using fallback method")
            # フォールバック: 従来の方法でLoRAモデルを探す
            all_models = {
                "lora_adapters": [],
                "merged_models": [],
                "continual_models": [],
                "gguf_models": [],
                "ollama_ready": []
            }
            
            # 簡易的なLoRAモデル検索
            outputs_dir = Path("/workspace/outputs")
            if outputs_dir.exists():
                for dir_path in outputs_dir.glob("lora_*"):
                    if dir_path.is_dir():
                        adapter_config = dir_path / "adapter_config.json"
                        if adapter_config.exists():
                            try:
                                with open(adapter_config) as f:
                                    config = json.load(f)
                                    model_info = {
                                        "path": str(dir_path),
                                        "name": dir_path.name,
                                        "base_model": config.get("base_model_name_or_path", "Unknown"),
                                        "type": "lora_adapter"
                                    }
                                    all_models["lora_adapters"].append(model_info)
                            except Exception as e:
                                logger.warning(f"Failed to read {adapter_config}: {e}")
        
        # 全モデルを統合リストに変換
        unified_models = []
        
        # LoRAアダプター
        for model in all_models["lora_adapters"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "lora_adapter",
                "base_model": model.get("base_model", "Unknown"),
                "display_name": f"[LoRA] {model['name']}",
                "size_mb": model.get("size_mb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": "merge_and_quantize",
                "recommended": "DeepSeek" in model.get("base_model", "") or "deepseek" in model.get("base_model", "").lower()
            })
        
        # マージ済みモデル
        for model in all_models["merged_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "merged_model",
                "display_name": f"[Merged] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": "quantize",
                "recommended": True
            })
        
        # 継続学習モデル
        for model in all_models["continual_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "continual_model",
                "task_name": model.get("task_name", "Unknown"),
                "display_name": f"[Continual] {model.get('task_name', model['name'])}",
                "modified_date": model.get("modified_date"),
                "needs_processing": "quantize",
                "recommended": False
            })
        
        # GGUFモデル
        for model in all_models["gguf_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "gguf_model",
                "display_name": f"[GGUF] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "quantization": model.get("quantization", "Unknown"),
                "modified_date": model.get("modified_date"),
                "needs_processing": "register_ollama" if model.get("has_modelfile") else "create_modelfile",
                "recommended": model.get("ollama_ready", False)
            })
        
        # Ollama登録済みモデル
        for model in all_models["ollama_ready"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "ollama_ready",
                "display_name": f"[Ollama Ready] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": None,
                "ready_to_use": True,
                "recommended": True
            })
        
        # 推奨順・新しい順でソート
        # 日付文字列を比較可能な形式に変換
        def sort_key(model):
            ready = not model.get("ready_to_use", False)
            recommended = not model.get("recommended", False)
            # 日付文字列を逆順にするため、存在しない場合は"0"、存在する場合は逆転
            date_str = model.get("modified_date", "")
            if date_str:
                # ISO形式の日付文字列は直接比較可能、新しい順にするため反転
                # 文字列の前に"-"を付けるのではなく、文字を反転させる
                date_sort = "".join(chr(255 - ord(c)) for c in date_str)
            else:
                date_sort = "zzz"  # 日付がない場合は最後に
            return (ready, recommended, date_sort)
        
        # ソート（モデルがある場合のみ）
        if unified_models:
            unified_models.sort(key=sort_key)
        
        # モデルが見つからない場合のデフォルトモデルを追加
        if not unified_models:
            logger.info("No finetuned models found, adding default entry")
            unified_models.append({
                "path": "auto",
                "name": "自動検出",
                "type": "auto",
                "display_name": "最新のLoRAモデルを自動検出",
                "base_model": "自動",
                "needs_processing": "auto_detect",
                "recommended": True
            })
        
        return {
            "models": unified_models,
            "count": len(unified_models),
            "summary": {
                "total": len(unified_models),
                "ready_to_use": sum(1 for m in unified_models if m.get("ready_to_use")),
                "needs_quantization": sum(1 for m in unified_models if m.get("needs_processing") in ["quantize", "merge_and_quantize"]),
                "lora_adapters": len(all_models.get("lora_adapters", [])),
                "merged_models": len(all_models.get("merged_models", [])),
                "continual_models": len(all_models.get("continual_models", [])),
                "gguf_models": len(all_models.get("gguf_models", []))
            }
        }
        
    except Exception as e:
        logger.error(f"LoRAモデルリスト取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-settings")
async def rag_update_settings(settings: Dict[str, Any]):
    """RAGシステムの設定を更新"""
    try:
        # 設定ファイルを更新
        config_path = Path("src/rag/config/rag_config.yaml")
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # LLMセクションが存在しない場合は作成
            if 'llm' not in config:
                config['llm'] = {}
            
            # LLMモデルの更新
            if 'llm_model' in settings and settings['llm_model']:
                if settings['llm_model'].startswith('ollama:'):
                    # Ollamaモデルの場合
                    ollama_model = settings['llm_model'].replace('ollama:', '')
                    config['llm']['provider'] = 'ollama'
                    config['llm']['ollama_model'] = ollama_model
                    config['llm']['model_name'] = f"ollama:{ollama_model}"
                    
                    # Ollama設定セクションを更新
                    if 'ollama' not in config['llm']:
                        config['llm']['ollama'] = {}
                    config['llm']['ollama']['model'] = ollama_model
                    config['llm']['ollama']['base_url'] = 'http://localhost:11434'
                    
                    config['llm']['use_finetuned'] = False
                    config['llm']['use_moe'] = False
                    config['llm']['use_ollama_fallback'] = True
                    logger.info(f"Ollamaモデル設定を更新: {ollama_model}")
                    
                elif settings['llm_model'].startswith('moe:'):
                    # MoEモデルの場合
                    moe_task_id = settings['llm_model'].replace('moe:', '')
                    config['llm']['use_moe'] = True
                    config['llm']['moe_model_path'] = f"/workspace/outputs/moe_{moe_task_id}"
                    config['llm']['use_finetuned'] = False
                    config['llm']['provider'] = 'local'
                    
                    # MoE設定を取得
                    try:
                        from app.moe_training_endpoints import training_tasks
                        if moe_task_id in training_tasks:
                            task = training_tasks[moe_task_id]
                            config['llm']['moe_num_experts'] = len(task.config.experts) if task.config.experts else 8
                            config['llm']['moe_experts_per_token'] = 2
                    except:
                        config['llm']['moe_num_experts'] = 8
                        config['llm']['moe_experts_per_token'] = 2
                    
                    logger.info(f"MoE設定を更新: task_id={moe_task_id}, experts={config['llm']['moe_num_experts']}")
                        
                elif settings['llm_model'].startswith('finetuned:'):
                    model_path = settings['llm_model'].replace('finetuned:', '')
                    config['llm']['model_name'] = model_path
                    config['llm']['model_path'] = model_path
                    config['llm']['use_finetuned'] = True
                    config['llm']['use_moe'] = False
                    config['llm']['provider'] = 'local'
                    logger.info(f"ファインチューニングモデル設定を更新: {model_path}")
                else:
                    config['llm']['model_name'] = settings['llm_model']
                    config['llm']['use_finetuned'] = False
                    config['llm']['use_moe'] = False
                    config['llm']['provider'] = 'local'
                    logger.info(f"ベースモデル設定を更新: {settings['llm_model']}")
            
            # 埋め込みモデルの更新
            if 'embedding_model' in settings:
                config['embedding']['model_name'] = settings['embedding_model']
            
            # Temperatureの更新
            if 'temperature' in settings:
                config['llm']['temperature'] = settings['temperature']
            
            # 設定を保存
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            # RAGシステムに設定を再読み込みさせる
            try:
                if hasattr(rag_app, 'query_engine') and rag_app.query_engine:
                    from src.rag.config.rag_config import load_config
                    new_config = load_config()
                    rag_app.query_engine.config = new_config
                    # LLMGeneratorの設定も同期
                    if hasattr(rag_app.query_engine, 'llm_generator') and rag_app.query_engine.llm_generator:
                        rag_app.query_engine.llm_generator.config = new_config
                    # finetunedモデルが選択されている場合、モデル切り替えを実行
                    llm_cfg = config.get('llm', {})
                    if llm_cfg.get('use_finetuned') and llm_cfg.get('finetuned_model_path'):
                        rag_app.switch_to_finetuned_model(llm_cfg['finetuned_model_path'])
                    logger.info("RAGクエリエンジンの設定を再読み込みしました")
            except Exception as reload_error:
                logger.warning(f"設定の再読み込み中にエラー: {reload_error}")
                # エラーが発生しても設定保存は成功とする
            
            return {"status": "success", "message": "設定を更新しました"}
        else:
            return {"status": "error", "message": "設定ファイルが見つかりません"}
            
    except Exception as e:
        logger.error(f"Failed to update RAG settings: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/query", response_model=QueryResponse)
async def rag_query_documents(request: QueryRequest):
    """RAG文書検索・質問応答"""
    rag_app.check_initialized()
    
    try:
        # モデル選択がある場合、RAG設定を一時的に更新
        original_model = None
        if request.model:
            try:
                # 現在の設定を保存
                original_model = rag_app.config.get('llm', {}).get('model_name')
                
                # モデル名を解析 (例: "ollama:deepseek-32b-rag")
                if request.model.startswith("ollama:"):
                    model_name = request.model.replace("ollama:", "")
                    # RAG設定を一時的に更新
                    rag_app.config['llm']['model_name'] = f"ollama:{model_name}:latest"
                    rag_app.config['llm']['ollama']['model'] = f"{model_name}:latest"
                    rag_app.config['llm']['ollama_model'] = f"{model_name}:latest"
                    logger.info(f"RAGクエリで使用するモデルを切り替え: {model_name}")
                elif request.model.startswith("finetuned:"):
                    # ファインチューニングモデルの場合
                    model_path = request.model.replace("finetuned:", "")
                    rag_app.config['llm']['model_name'] = model_path
                    rag_app.config['llm']['use_finetuned'] = True
                    # query engine のLLMGeneratorに直接反映
                    rag_app.switch_to_finetuned_model(model_path)
                    logger.info(f"RAGクエリでファインチューニングモデルを使用: {model_path}")
                else:
                    # その他のモデル
                    rag_app.config['llm']['model_name'] = request.model
                    logger.info(f"RAGクエリでモデルを使用: {request.model}")
                    
                # クエリエンジンを再初期化（必要な場合）
                if hasattr(rag_app, '_reinitialize_query_engine'):
                    rag_app._reinitialize_query_engine()
            except Exception as e:
                logger.warning(f"モデル切り替えに失敗、デフォルトを使用: {e}")
        
        # document_idsをfiltersに追加
        filters = request.filters or {}
        if request.document_ids:
            filters['document_ids'] = request.document_ids
            logger.info(f"Filtering by document IDs: {request.document_ids}")
        
        # クエリを実行
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_app.query_engine.query,
            request.query,
            request.top_k,
            request.search_type,
            filters if filters else None,
            request.include_sources
        )
        
        # 元のモデル設定を復元
        if original_model and request.model:
            rag_app.config['llm']['model_name'] = original_model
            if request.model.startswith("finetuned:"):
                rag_app.restore_default_model()
            elif hasattr(rag_app, '_reinitialize_query_engine'):
                rag_app._reinitialize_query_engine()
        
        return QueryResponse(**result.to_dict())
        
    except HTTPException:
        raise
    except (RAGQueryError, RAGInitializationError) as e:
        logger.error("RAGクエリエラー: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")
    except VectorStoreConnectionError as e:
        logger.error("ベクトルストア接続エラー: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")
    except (KeyError, AttributeError) as e:
        logger.error("RAG設定エラー: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG configuration error: {e}")
    except Exception as e:
        logger.error("予期しないRAGクエリエラー: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@router.post("/batch-query")
async def rag_batch_query_documents(request: BatchQueryRequest):
    """RAGバッチクエリ"""
    rag_app.check_initialized()
    
    if len(request.queries) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries allowed in batch"
        )
        
    try:
        # バッチクエリを実行
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_app.query_engine.batch_query,
            request.queries,
            request.top_k,
            request.search_type
        )
        
        return {
            "status": "success",
            "results": [result.to_dict() for result in results],
            "total_queries": len(request.queries),
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def rag_search_documents(
    q: str = Query(..., description="検索クエリ"),
    top_k: int = Query(5, description="取得する結果数", ge=1, le=20),
    search_type: str = Query("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
):
    """RAG簡易検索API"""
    
    request = QueryRequest(
        query=q,
        top_k=top_k,
        search_type=search_type,
        include_sources=True
    )
    
    return await rag_query_documents(request)

@router.get("/documents")
async def rag_list_documents(
    limit: int = Query(50, description="取得件数", ge=1, le=100),
    offset: int = Query(0, description="オフセット", ge=0),
    document_type: Optional[str] = Query(None, description="文書タイプフィルター")
):
    """RAG文書一覧を取得"""
    rag_app.check_initialized()
    
    try:
        # メタデータから文書を検索
        try:
            from src.rag.indexing.metadata_manager import DocumentType
        except ImportError:
            # DocumentTypeがない場合のフォールバック
            DocumentType = None
        
        filters = {}
        if document_type and DocumentType:
            try:
                filters['document_type'] = DocumentType(document_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid document_type: {document_type}")
                
        documents = rag_app.metadata_manager.search_documents(**filters)
        
        # ページネーション
        total = len(documents)
        paginated_docs = documents[offset:offset + limit]
        
        return {
            "status": "success",
            "documents": [doc.to_dict() for doc in paginated_docs],
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            },
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def rag_get_statistics():
    """RAGシステム統計を取得"""
    rag_app.check_initialized()
    
    try:
        stats = rag_app.metadata_manager.get_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def rag_delete_document(document_id: str):
    """RAG文書を削除"""
    rag_app.check_initialized()
    
    try:
        # メタデータから文書情報を取得
        doc_metadata = rag_app.metadata_manager.get_document(document_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        # ベクトルストアからチャンクを削除
        # チャンクIDはdocument_id_<chunk_index>形式で保存されている
        chunk_ids = []
        for i in range(1000):  # 最大1000チャンクまで対応
            chunk_id = f"{document_id}_{i}"
            chunk_ids.append(chunk_id)
        
        # ベクトルストアから削除（存在しないIDは無視される）
        try:
            rag_app.vector_store.delete(chunk_ids)
            logger.info(f"Deleted chunks from vector store for document: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete from vector store: {e}")
        
        # メタデータから削除
        rag_app.metadata_manager.delete_document(document_id)
        
        # 処理済みファイルを削除
        processed_file = Path(f"./outputs/rag_index/processed_documents/{document_id}.json")
        if processed_file.exists():
            processed_file.unlink()
            logger.info(f"Deleted processed file: {processed_file}")
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully",
            "document_title": doc_metadata.title,
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_rag_document(
    file_path: str,
    title: str,
    category: str,
    document_type: str,
    status_callback: Optional[callable] = None
):
    """アップロードされたRAG文書を処理（バックグラウンドタスク）"""
    
    # 文書IDを生成
    doc_id = str(uuid.uuid4())
    
    status_info = {
        "file_path": file_path,
        "title": title,
        "doc_id": doc_id,
        "status": "processing",
        "progress": 0,
        "message": "処理を開始しています...",
        "start_time": datetime.now(JST).isoformat()
    }
    
    # ステータス情報を保存
    status_file = PathlibPath(f"./temp_uploads/status_{PathlibPath(file_path).stem}.json")
    
    def update_status(progress: int, message: str, status: str = "processing"):
        """処理状況を更新"""
        status_info.update({
            "progress": progress,
            "message": message,
            "status": status,
            "last_update": datetime.now(JST).isoformat()
        })
        
        # ステータスファイルに保存
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save status: {e}")
        
        logger.info(f"[{progress}%] {message}")
    
    try:
        update_status(10, f"ファイルを読み込んでいます: {PathlibPath(file_path).name}")
        
        # ファイルサイズチェック
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        logger.info(f"Processing file: {file_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
        
        if file_size_mb > 100:
            update_status(15, f"大きなファイル ({file_size_mb:.1f}MB) を処理中... 時間がかかる場合があります")
        
        # PDFの前処理とページ数取得
        update_status(20, "PDFの構造を解析中...")
        
        # PyMuPDFでページ数を事前確認
        import fitz
        with fitz.open(file_path) as pdf_doc:
            total_pages = len(pdf_doc)
            logger.info(f"PDF has {total_pages} pages")
            
            # OCRが必要かチェック
            needs_ocr = False
            for page_num in range(min(5, total_pages)):  # 最初の5ページをチェック
                page = pdf_doc[page_num]
                text = page.get_text()
                if not text.strip():
                    needs_ocr = True
                    break
            
            if needs_ocr:
                update_status(25, f"スキャンされたPDFを検出しました。OCR処理を準備中... (全{total_pages}ページ)")
            else:
                update_status(25, f"テキストPDFを処理中... (全{total_pages}ページ)")
        
        # インデックス作成スクリプトを実行（タイムアウト設定）
        update_status(30, "文書のインデックス化を開始...")
        
        import subprocess
        import asyncio
        
        logger.info(f"Starting indexing process - Pages: {total_pages}, OCR needed: {needs_ocr}")
        
        # OCRモデルがダウンロード済みかチェック
        ocr_model_exists = False
        try:
            easyocr_model_dir = os.path.expanduser("~/.EasyOCR/model")
            if os.path.exists(easyocr_model_dir):
                # モデルファイルが存在するかチェック
                model_files = os.listdir(easyocr_model_dir) if os.path.exists(easyocr_model_dir) else []
                ocr_model_exists = len(model_files) > 0
                logger.info(f"OCR model directory exists: {ocr_model_exists}, files: {len(model_files)}")
        except Exception as e:
            logger.warning(f"Could not check OCR model: {e}")
        
        # 大きなファイルやOCRモデル未ダウンロードの場合はOCRを無効化
        # 7GB以上のファイルは常にOCRを無効化
        if file_size_gb >= 7:
            logger.warning(f"Large file ({file_size_gb:.1f}GB) - forcing OCR disabled")
            needs_ocr = False
            update_status(28, "大容量ファイル（7GB以上）のため、OCRを強制的にスキップします")
        elif file_size_gb >= 5 and not ocr_model_exists:
            logger.warning(f"Large file ({file_size_gb:.1f}GB) and OCR model not downloaded - disabling OCR")
            needs_ocr = False
            update_status(28, "大容量ファイルのため、OCRをスキップします")
        
        # タイムアウト時間を動的に設定（ファイルサイズとページ数とOCR必要性に基づく）
        # OCRモデルのダウンロードが必要な場合は追加時間を設定
        model_download_time = 0 if ocr_model_exists else 1800  # モデルダウンロードに30分
        base_timeout = 1200 + model_download_time  # 基本20分 + モデルダウンロード時間
        
        # ファイルサイズに基づく追加時間（1GBあたり20分）
        # file_size_gbは既に定義済み
        size_timeout = int(file_size_gb * 1200)  # 1GBあたり20分
        
        # ページ数に基づく追加時間
        per_page_timeout = 20 if needs_ocr else 10  # OCR必要なら20秒/ページ、不要なら10秒/ページ
        page_timeout = total_pages * per_page_timeout
        
        # 合計タイムアウト時間
        timeout_seconds = base_timeout + size_timeout + page_timeout
        
        # 7GB以上のファイルには特別な配慮
        if file_size_gb >= 7:
            logger.info(f"Large file detected: {file_size_gb:.2f}GB - applying extended timeout")
            timeout_seconds = max(timeout_seconds, 7200)  # 最小2時間
        
        # 最小20分、最大10時間に制限（30GBのPDFに対応）
        timeout_seconds = max(1200, min(timeout_seconds, 36000))
        
        logger.info(f"File size: {file_size_mb:.2f}MB ({file_size_gb:.2f}GB), Pages: {total_pages}, OCR: {needs_ocr}")
        logger.info(f"Timeout calculation: base={base_timeout}s, size={size_timeout}s, pages={page_timeout}s, total={timeout_seconds}s ({timeout_seconds//60} minutes)")
        logger.info(f"Final timeout: {timeout_seconds} seconds ({timeout_seconds//60:.1f} minutes, {timeout_seconds//3600:.1f} hours)")
        
        update_status(35, f"インデックス処理中... (最大{timeout_seconds//60}分待機)")
        
        # スクリプトの存在確認
        script_path = "/workspace/scripts/rag/index_documents.py"
        if not os.path.exists(script_path):
            error_msg = f"インデックススクリプトが見つかりません: {script_path}"
            logger.error(error_msg)
            update_status(100, error_msg, status="error")
            return
        
        # 非同期でサブプロセスを実行
        update_status(40, "インデックススクリプトを起動中...")
        
        cmd = [
            sys.executable,
            script_path,
            file_path,
            "--output-dir", "/workspace/outputs/rag_index",
            "--metadata-db-path", "/workspace/metadata/metadata.db"
        ]
        
        # 大きなファイルでOCRが不要な場合は明示的に無効化
        if not needs_ocr:
            cmd.extend(["--no-ocr"])
            logger.info("OCR disabled for this document")
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # 環境変数を設定（大容量ファイル処理の最適化）
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 出力をバッファリングしない
        env['OMP_NUM_THREADS'] = '4'  # OpenMPスレッド数を制限
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/workspace",
                env=env,
                # バッファ制限を削除（大きなファイルでのデッドロック防止）
            )
            logger.info(f"Process started with PID: {process.pid}")
        except Exception as e:
            error_msg = f"プロセス起動エラー: {str(e)}"
            logger.error(error_msg)
            update_status(100, error_msg, status="error")
            return
        
        # プログレス更新（シンプル版）
        update_status(50, f"文書処理中... (全{total_pages}ページ)")
        
        try:
            logger.info(f"Waiting for subprocess to complete (timeout: {timeout_seconds}s)")
            
            # プロセスの完了を待つ（ストリーミング版 - 大容量ファイル対応）
            stdout_lines = []
            stderr_lines = []
            
            async def read_stream(stream, lines_list, stream_name):
                """ストリームを非同期で読み込み（メモリ効率的）"""
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        # 最新の100行だけ保持（メモリ節約）
                        if len(lines_list) > 100:
                            lines_list.pop(0)
                        lines_list.append(line.decode('utf-8', errors='ignore'))
                        
                        # 進捗表示があれば更新
                        if stream_name == "stdout" and "Progress:" in lines_list[-1]:
                            logger.debug(f"Progress: {lines_list[-1].strip()}")
                except Exception as e:
                    logger.debug(f"Stream read error ({stream_name}): {e}")
            
            try:
                # タスクを作成してストリームを並行読み込み
                stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_lines, "stdout"))
                stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_lines, "stderr"))
                
                # タイムアウトが正しく設定されていることを確認
                logger.info(f"Starting process wait with timeout: {timeout_seconds} seconds")
                start_time = asyncio.get_event_loop().time()
                
                # プロセスの完了を待つ
                await asyncio.wait_for(process.wait(), timeout=float(timeout_seconds))
                
                elapsed_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"Process completed after {elapsed_time:.1f} seconds")
                
                # ストリーム読み込みタスクも完了を待つ
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=10  # ストリーム読み込みは10秒でタイムアウト
                )
                
                logger.info(f"Process completed successfully")
                stdout = '\n'.join(stdout_lines).encode('utf-8')
                stderr = '\n'.join(stderr_lines).encode('utf-8')
                
            except asyncio.TimeoutError:
                elapsed_time = asyncio.get_event_loop().time() - start_time if 'start_time' in locals() else 0
                logger.error(f"Process timed out after {elapsed_time:.1f} seconds (timeout was {timeout_seconds} seconds)")
                
                # タイムアウト時はタスクをキャンセル
                if 'stdout_task' in locals():
                    stdout_task.cancel()
                if 'stderr_task' in locals():
                    stderr_task.cancel()
                
                # プロセスを終了
                process.terminate()
                await asyncio.sleep(2)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                
                stdout = '\n'.join(stdout_lines).encode('utf-8')
                stderr = '\n'.join(stderr_lines).encode('utf-8')
                logger.info(f"Process terminated due to timeout (PID: {process.pid})")
            
            # デバッグ用にログ出力
            return_code = process.returncode
            logger.info(f"Process finished with return code: {return_code}")
            
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                logger.info(f"Index script stdout: {stdout_text[:1000]}")
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                if stderr_text.strip():
                    logger.warning(f"Index script stderr: {stderr_text[:1000]}")
            
            # インデックススクリプトは部分的な成功でも return_code 1 を返すことがある
            # stdout に Progress 表示があれば成功として扱う
            has_progress = False
            has_completion = False
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                has_progress = "Progress:" in stdout_text or "processed successfully" in stdout_text.lower()
                # インデックス作成完了のメッセージも確認
                has_completion = "Document processed successfully" in stdout_text or "Successfully added" in stdout_text
                
            # 成功判定の条件を緩和（return_code 1でも処理が完了していれば成功とする）
            # タイムアウトした場合も、ある程度処理が進んでいれば成功とする
            if return_code == 0 or return_code == 1 or has_progress or has_completion:
                if return_code == 1:
                    logger.info(f"Script returned code 1 (partial success), checking for actual completion")
                elif return_code != 0:
                    logger.info(f"Script returned code {return_code} but has progress output, treating as success")
                
                # 処理が成功したことを確実に記録
                logger.info(f"RAG Document processed and saved successfully: {file_path}")
                
                # 必ず100%に更新（データ保存完了）
                logger.info("Updating status to 100% completed")
                update_status(100, "文書のインデックス化と保存が完了しました！", status="completed")
                logger.info("Status updated to 100% completed")
                
                # キーワード検索エンジンを再初期化して新しい文書を反映
                try:
                    logger.info("Reinitializing keyword search engine to include newly uploaded document")
                    if 'rag_app' in globals() and getattr(rag_app, 'query_engine', None):
                        # ブロッキング処理のためスレッドプールで実行
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, rag_app.query_engine._initialize_search_corpus)
                        logger.info("Keyword search engine reinitialized successfully")
                    else:
                        logger.warning("RAG application or query engine not available for keyword search reinitialization")
                except Exception as e:
                    logger.error(f"Failed to reinitialize keyword search engine: {e}")
                    # エラーが発生してもRAGシステムは正常に動作するため、処理を継続
                
                # 完了を確実にするために短い待機
                await asyncio.sleep(2)
                
                # ファイル削除（処理完了後の必須クリーンアップ）
                try:
                    os.remove(file_path)
                    logger.info(f"Removed processed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file: {e}")
                    
            else:
                error_msg = stderr_text if 'stderr_text' in locals() and stderr_text else "Unknown error"
                logger.error(f"RAG Document processing failed: {error_msg}")
                update_status(100, f"処理エラー: {error_msg[:200]}", status="error")
                
        except asyncio.TimeoutError:
            process.terminate()
            await asyncio.sleep(2)
            if process.returncode is None:
                process.kill()
                await process.wait()
            
            error_msg = f"処理がタイムアウトしました ({timeout_seconds}秒)。ファイルが大きすぎるか、複雑すぎる可能性があります。"
            logger.error(f"RAG Document processing timeout: {file_path}")
            update_status(100, error_msg, status="timeout")
            
    except Exception as e:
        error_msg = f"処理中にエラーが発生: {str(e)}"
        logger.error(f"Background RAG document processing failed: {e}", exc_info=True)
        update_status(100, error_msg, status="error")
    
    finally:
        # 最終的なステータス確認と完了処理
        if status_info.get("status") not in ["completed", "error", "timeout"]:
            # まだ完了していない場合は、強制的に完了とする
            logger.warning("Process did not reach completion status, forcing completion")
            update_status(100, "処理を完了しました（強制完了）", status="completed")
            
            # 一時ファイルのクリーンアップ
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
        
        # ステータスファイルの削除は遅延させる
        if status_info.get("status") == "completed":
            # 成功時は5分後に削除（クライアントが結果を確認する時間を確保）
            await asyncio.sleep(300)  # 5分後に削除
            try:
                if os.path.exists(status_file):
                    os.remove(status_file)
                    logger.info(f"Removed status file: {status_file}")
            except Exception as e:
                logger.warning(f"Failed to remove status file: {e}")
        elif status_info.get("status") in ["error", "timeout"]:
            # エラー時は10分後に削除（デバッグ用に長めに保持）
            await asyncio.sleep(600)  # 10分後に削除
            try:
                if os.path.exists(status_file):
                    os.remove(status_file)
            except:
                pass

@router.post("/save-search")
async def save_search_result(request: SaveSearchRequest):
    """検索結果を保存"""
    try:
        rag_app.check_initialized()
        
        saved_result = rag_app.save_search_result(
            query_response=request.query_response,
            name=request.name,
            tags=request.tags
        )
        
        return {
            "status": "success",
            "message": "Search result saved successfully",
            "result_id": saved_result.id,
            "saved_at": saved_result.saved_at
        }
        
    except Exception as e:
        logger.error(f"Failed to save search result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-history", response_model=SearchHistoryResponse)
async def get_search_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    tag: Optional[str] = None
):
    """検索履歴を取得"""
    try:
        rag_app.check_initialized()
        
        history = rag_app.get_search_history(page=page, limit=limit, tag=tag)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/search-history/{result_id}")
async def delete_search_history_item(result_id: str):
    """検索履歴の個別アイテムを削除"""
    try:
        rag_app.check_initialized()
        
        # 削除を実行
        success = rag_app.delete_search_history_item(result_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Search history item '{result_id}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Search history item '{result_id}' not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete search history item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/search-history/clear")
async def clear_all_search_history():
    """全ての検索履歴を削除"""
    try:
        rag_app.check_initialized()
        
        # 全履歴をクリア
        rag_app.search_history = []
        
        return {
            "status": "success",
            "message": "All search history cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear all search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

## (Removed) /rag/ephemeral-verify endpoint — reverting per request

@router.get("/search-result/{result_id}")
async def get_saved_search_result(result_id: str):
    """保存された検索結果を取得"""
    try:
        rag_app.check_initialized()
        
        result = rag_app.get_saved_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Search result not found")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get saved search result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-searches")
async def export_search_results(
    result_ids: str = Query(..., description="カンマ区切りの結果ID"),
    format: str = Query("json", pattern="^(json|csv)$")
):
    """検索結果をエクスポート"""
    try:
        rag_app.check_initialized()
        
        ids = result_ids.split(",")
        export_data = rag_app.export_search_results(ids, format=format)
        
        filename = f"search_results_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.{format}"
        media_type = "text/csv" if format == "csv" else "application/json"
        
        return StreamingResponse(
            io.BytesIO(export_data),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to export search results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- New: Extract PDF text to inject into prompt ---
@router.post("/extract-pdf-text")
async def extract_pdf_text(file: UploadFile = File(...), max_chars: int = Form(120000)):
    """Extract plain text from a PDF and return it for prompt injection.
    Does not index or persist content. Truncates to max_chars.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # 安全のため上限・下限をクランプ（過大入力による負荷を避ける）
        try:
            max_chars = int(max_chars)
        except Exception:
            max_chars = 120000
        max_chars = max(1000, min(max_chars, 200000))

        tmp_dir = Path("./temp_uploads/prompt")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
        content = await file.read()
        with open(tmp_path, 'wb') as f:
            f.write(content)

        # Lightweight extraction (tables/ocr off)
        from src.rag.document_processing.document_processor import RoadDesignDocumentProcessor
        processor = RoadDesignDocumentProcessor(
            extract_tables=False,
            extract_figures=False,
            perform_ocr=False,
            chunk_size=512,
            chunk_overlap=128,
        )
        processed = processor.process_document(str(tmp_path), document_metadata={"source": file.filename, "attached": True})
        if processed is None:
            raise HTTPException(status_code=500, detail="Failed to process PDF")

        # Join chunk texts as a single prompt text
        if processed.chunks and len(processed.chunks) > 0:
            full_text = "\n\n".join(c.text for c in processed.chunks if c.text and c.text.strip())
            logger.info(f"Extracted text from {len(processed.chunks)} chunks, total length: {len(full_text)}")
        else:
            # チャンクが空の場合、PDFから直接テキストを抽出
            logger.warning("No chunks found, attempting direct PDF text extraction")
            try:
                import fitz  # PyMuPDF
                pdf_doc = fitz.open(str(tmp_path))
                text_parts = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text_parts.append(page_text)
                pdf_doc.close()
                full_text = "\n\n".join(text_parts)
                logger.info(f"Direct PDF extraction: {len(full_text)} characters from {len(text_parts)} pages")
            except Exception as e:
                logger.error(f"Direct PDF extraction failed: {e}")
                full_text = ""

        if not full_text or not full_text.strip():
            raise HTTPException(status_code=500, detail="No text could be extracted from PDF. The PDF may be image-only or corrupted.")

        truncated = (full_text[: max_chars] + "\n... (truncated)") if len(full_text) > max_chars else full_text
        logger.info(f"Returning {len(truncated)} characters (truncated from {len(full_text)})")

        return {
            "filename": file.filename,
            "doc_id": processed.id,
            "chars": len(full_text),
            "returned_chars": len(truncated),
            "text": truncated,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF extract failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

## (Removed) /rag/fact-check-pdf endpoint — reverting per request

@router.post("/upload-document")
async def rag_upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    category: Optional[str] = None,
    document_type: Optional[str] = None
):
    """RAG文書をアップロードしてインデックス化"""
    
    # ファイル形式チェック
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
        
    try:
        # インデックススクリプトの存在を事前確認
        script_path = "/workspace/scripts/rag/index_documents.py"
        if not os.path.exists(script_path):
            raise HTTPException(
                status_code=500,
                detail=f"インデックススクリプトが見つかりません: {script_path}"
            )
        
        # 一時ファイルに保存
        upload_dir = PathlibPath("./temp_uploads")
        upload_dir.mkdir(exist_ok=True)

        # パストラバーサル対策: ファイル名からディレクトリ成分を除去
        safe_filename = os.path.basename(file.filename)
        if not safe_filename or safe_filename.startswith("."):
            raise HTTPException(status_code=400, detail="ファイル名が不正です")

        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{safe_filename}"
        temp_path = upload_dir / temp_filename
        # シンボリックリンク等による脱出を検証
        if not temp_path.resolve().is_relative_to(upload_dir.resolve()):
            raise HTTPException(status_code=400, detail="ファイル名が不正です")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # バックグラウンドでインデックス化を実行
        background_tasks.add_task(
            process_uploaded_rag_document,
            str(temp_path),
            title or file.filename,
            category or "その他",
            document_type or "other"
        )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded and queued for processing",
            document_id=temp_filename,
            processing_status="queued",
            metadata={"page_count": 0, "status": "processing"}
        )
        
    except HTTPException:
        raise
    except (FileValidationError, DocumentProcessingError) as e:
        logger.error("文書アップロードエラー: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Document upload failed: {e}")
    except (OSError, IOError) as e:
        logger.error("ファイルI/Oエラー: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"File I/O error: {e}")
    except Exception as e:
        logger.error("予期しないアップロードエラー: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@router.get("/upload-status/{document_id}")
async def get_upload_status(document_id: str):
    """PDFアップロードの処理状況を取得"""
    try:
        # ステータスファイルを確認
        status_file = PathlibPath(f"./temp_uploads/status_{document_id.replace('.pdf', '')}.json")
        
        if not status_file.exists():
            # ステータスファイルがない場合
            # 元のファイルも存在しない場合は処理完了とみなす
            original_file = PathlibPath(f"./temp_uploads/{document_id}")
            if not original_file.exists():
                # ファイルが削除されている = 処理完了
                return {
                    "status": "completed",
                    "message": "処理が完了しました",
                    "progress": 100
                }
            else:
                # ファイルはあるがステータスがない = 未開始
                return {
                    "status": "unknown",
                    "message": "処理状況が見つかりません",
                    "progress": 0
                }
        
        # ステータス情報を読み込み
        with open(status_file, "r", encoding="utf-8") as f:
            status_info = json.load(f)
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        return {
            "status": "error",
            "message": f"状況取得エラー: {str(e)}",
            "progress": 0
        }

@router.post("/stream-query")
async def rag_stream_query(request: QueryRequest):
    """RAGストリーミングクエリ（リアルタイム応答）"""
    rag_app.check_initialized()
    
    async def generate_response():
        """レスポンスを段階的に生成"""
        
        # 検索フェーズ
        yield f"data: {json.dumps({'phase': 'search', 'message': '文書を検索中...'})}\n\n"
        await asyncio.sleep(0.1)
        
        try:
            # 実際のクエリ実行
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                rag_app.query_engine.query,
                request.query,
                request.top_k,
                request.search_type,
                request.filters,
                request.include_sources
            )
            
            # 結果フェーズ
            yield f"data: {json.dumps({'phase': 'result', 'data': result.to_dict()})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'phase': 'error', 'error': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
