"""Model management API router: convert, list, delete, stream, generate."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM

from ..dependencies import logger, model_cache, training_tasks
from ..model_utils import load_tokenizer, create_quantization_config
from .proofreading import (
    PROOFREADING_AVAILABLE,
    is_proofreading_request,
    _truncate_chars,
    _tokenizer_input_max_len,
)
import app.dependencies as _deps

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/api", tags=["models"])


# ---------------------------------------------------------------------------
# Lazy Ollama helper
# ---------------------------------------------------------------------------

def _get_ollama():
    """Return (OllamaIntegration_instance, True) or (None, False)."""
    if not _deps.OLLAMA_AVAILABLE:
        return None, False
    try:
        scripts_convert_path = Path(__file__).parent.parent.parent / "scripts" / "convert"
        if str(scripts_convert_path) not in sys.path:
            sys.path.insert(0, str(scripts_convert_path))
        from ollama_integration import OllamaIntegration
        return OllamaIntegration(), True
    except ImportError:
        return None, False


# ---------------------------------------------------------------------------
# Convert endpoints
# ---------------------------------------------------------------------------

@router.post("/convert-to-ollama")
async def convert_finetuned_to_ollama(request: dict):
    """ファインチューニング済みモデルをOllama形式に変換"""
    try:
        model_path = request.get("model_path")
        model_name = request.get("model_name", "road-engineering-expert")

        if not model_path:
            return {"success": False, "error": "model_pathが指定されていません"}

        logger.info(f"ファインチューニング済みモデルのOllama変換開始: {model_path}")

        script_path = Path("convert_finetuned_to_ollama.py")
        if not script_path.exists():
            return {"success": False, "error": "変換スクリプトが見つかりません"}

        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            logger.info("ファインチューニング済みモデルのOllama変換が完了しました")
            return {
                "success": True,
                "model_name": model_name,
                "message": "ファインチューニング済みモデルがOllamaで使用可能になりました",
                "usage": f"ollama run {model_name}",
            }
        else:
            logger.error(f"変換エラー: {result.stderr}")
            return {"success": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("変換がタイムアウトしました")
        return {"success": False, "error": "Conversion timeout"}
    except Exception as e:
        logger.error(f"変換エラー: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/apply-lora-to-ollama")
async def apply_lora_to_ollama(request: dict, background_tasks: BackgroundTasks):
    """LoRAアダプターをGGUFベースモデルに適用してOllamaに登録"""
    try:
        base_model_url = request.get("base_model_url")
        base_model_name = request.get("base_model_name", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf")
        lora_adapter_path = request.get("lora_adapter_path")
        output_model_name = request.get("output_model_name", "deepseek-32b-finetuned")
        use_improved_version = request.get("use_improved_version", True)

        task_id = str(uuid.uuid4())

        async def run_conversion():
            try:
                dynamic_script_path = "/workspace/scripts/apply_lora_gpt_neox_dynamic.py"
                auto_script_path = "/workspace/scripts/apply_lora_to_gguf_auto.py"

                if Path(dynamic_script_path).exists() and "gpt-neox" in base_model_name.lower():
                    script_path = dynamic_script_path
                    logger.info("動的適用版LoRAスクリプトを使用（GPT-NeoX ワークフローB）")
                elif Path(auto_script_path).exists():
                    script_path = auto_script_path
                    logger.info("自動判定版LoRA適用スクリプトを使用")
                elif use_improved_version:
                    script_path = "/workspace/scripts/apply_lora_to_gguf_improved.py"
                else:
                    script_path = "/workspace/scripts/apply_lora_to_gguf.py"

                cmd = ["python", script_path]

                if "dynamic" in script_path:
                    cmd.extend([
                        "--base-gguf", f"/workspace/models/{base_model_name}",
                        "--lora-adapter", lora_adapter_path if lora_adapter_path else "/workspace/outputs/lora_latest",
                        "--output-dir", f"/workspace/outputs/workflow_b_{output_model_name}",
                        "--ollama-create", output_model_name,
                    ])
                else:
                    if base_model_url:
                        cmd.extend(["--base-model-url", base_model_url])
                    cmd.extend(["--base-model-name", base_model_name, "--output-name", output_model_name])
                    if lora_adapter_path:
                        cmd.extend(["--lora-adapter", lora_adapter_path])

                logger.info(f"LoRA to Ollama変換開始: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    if task_id in training_tasks:
                        training_tasks[task_id]["status"] = "completed"
                        training_tasks[task_id]["message"] = f"Model {output_model_name} created successfully"
                        training_tasks[task_id]["model_name"] = output_model_name
                        logger.info(f"LoRA to Ollama変換成功: {output_model_name}")
                else:
                    if task_id in training_tasks:
                        training_tasks[task_id]["status"] = "error"
                        training_tasks[task_id]["message"] = result.stderr
                        logger.error(f"LoRA to Ollama変換失敗: {result.stderr}")

            except Exception as e:
                if task_id in training_tasks:
                    training_tasks[task_id]["status"] = "error"
                    training_tasks[task_id]["message"] = str(e)
                logger.error(f"LoRA to Ollama変換エラー: {str(e)}")

        training_tasks[task_id] = {
            "status": "running",
            "message": "Converting LoRA adapter to Ollama format...",
            "progress": 0,
            "task_id": task_id,
        }

        background_tasks.add_task(run_conversion)

        return {"status": "started", "task_id": task_id, "message": "LoRA to Ollama conversion started"}

    except Exception as e:
        logger.error(f"LoRA to Ollama conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert-to-ollama-wsl")
async def convert_finetuned_to_ollama_wsl(request: dict):
    """WSL環境用：ファインチューニング済みモデルをOllama形式に変換"""
    try:
        model_path = request.get("model_path")
        model_name = request.get("model_name", "road-engineering-expert")

        if not model_path:
            return {"success": False, "error": "model_pathが指定されていません"}

        logger.info(f"WSL環境でファインチューニング済みモデルのOllama変換開始: {model_path}")

        script_path = Path("setup_wsl_ollama.py")
        if not script_path.exists():
            return {"success": False, "error": "WSL変換スクリプトが見つかりません"}

        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            logger.info("WSL環境でのファインチューニング済みモデルのOllama変換が完了しました")
            return {
                "success": True,
                "model_name": model_name,
                "message": "WSL環境でファインチューニング済みモデルがOllamaで使用可能になりました",
                "usage": f"ollama run {model_name}",
            }
        else:
            logger.error(f"WSL変換エラー: {result.stderr}")
            return {"success": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("WSL変換がタイムアウトしました")
        return {"success": False, "error": "Conversion timeout"}
    except Exception as e:
        logger.error(f"WSL変換エラー: {str(e)}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# List / query endpoints
# ---------------------------------------------------------------------------

@router.get("/finetuned-lora-models")
async def get_finetuned_lora_models():
    """ファインチューニング済みのLoRAアダプターモデルを取得"""
    try:
        models = []
        outputs_dir = Path("/workspace/outputs")

        if outputs_dir.exists():
            # safetensors
            for path in outputs_dir.rglob("adapter_model.safetensors"):
                model_dir = path.parent
                config_path = model_dir / "adapter_config.json"

                model_info = {
                    "path": str(model_dir),
                    "name": model_dir.name,
                    "type": "lora",
                    "format": "safetensors",
                }

                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            model_info["base_model"] = config.get("base_model_name_or_path", "unknown")
                            model_info["r"] = config.get("r", "unknown")
                            model_info["alpha"] = config.get("lora_alpha", "unknown")
                    except Exception:
                        pass

                try:
                    jst = timezone(timedelta(hours=9), name="JST")
                    created_dt = datetime.fromtimestamp(model_dir.stat().st_mtime, tz=jst)
                    model_info["created"] = created_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    model_info["created"] = datetime.fromtimestamp(
                        model_dir.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")

                models.append(model_info)

            # bin format
            for path in outputs_dir.rglob("adapter_model.bin"):
                model_dir = path.parent
                if any(m["path"] == str(model_dir) for m in models):
                    continue
                config_path = model_dir / "adapter_config.json"

                model_info = {
                    "path": str(model_dir),
                    "name": model_dir.name,
                    "type": "lora",
                    "format": "bin",
                }

                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            model_info["base_model"] = config.get("base_model_name_or_path", "unknown")
                            model_info["r"] = config.get("r", "unknown")
                            model_info["alpha"] = config.get("lora_alpha", "unknown")
                    except Exception:
                        pass

                try:
                    jst = timezone(timedelta(hours=9), name="JST")
                    created_dt = datetime.fromtimestamp(model_dir.stat().st_mtime, tz=jst)
                    model_info["created"] = created_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    model_info["created"] = datetime.fromtimestamp(
                        model_dir.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")

                models.append(model_info)

        models.sort(key=lambda x: x["created"], reverse=True)
        return {"success": True, "models": models, "count": len(models)}

    except Exception as e:
        logger.error(f"Error getting finetuned LoRA models: {e}")
        return {"success": False, "error": str(e), "models": []}


@router.get("/available-models")
async def get_available_models():
    """利用可能なファインチューニング済みモデルとOllamaモデルを取得"""
    try:
        models: dict = {"finetuned_models": [], "ollama_models": []}

        # ファインチューニング済みモデルの検索
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for model_dir in outputs_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                has_model_files = (
                    (model_dir / "pytorch_model.bin").exists()
                    or (model_dir / "adapter_model.safetensors").exists()
                    or (model_dir / "adapter_config.json").exists()
                    or (model_dir / "config.json").exists()
                )
                if not has_model_files:
                    continue

                config_path = model_dir / "config.json"
                training_info_path = model_dir / "training_info.json"

                model_info: dict = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "finetuned",
                    "size": "Unknown",
                    "created": "Unknown",
                }

                if config_path.exists():
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            model_info["base_model"] = config.get("_name_or_path", "Unknown")
                            model_info["model_type"] = config.get("model_type", "Unknown")
                    except Exception:
                        pass

                if training_info_path.exists():
                    try:
                        with open(training_info_path, "r", encoding="utf-8") as f:
                            training_info = json.load(f)
                            model_info["training_method"] = training_info.get("training_method", "unknown")
                            created_str = training_info.get("created_at") or training_info.get("timestamp")
                            formatted_created = None
                            try:
                                if isinstance(created_str, str):
                                    if "T" in created_str:
                                        iso = created_str.replace("Z", "+00:00")
                                        dt = datetime.fromisoformat(iso)
                                        formatted_created = dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST")
                                    elif "_" in created_str and len(created_str) >= 15:
                                        try:
                                            dt2 = datetime.strptime(created_str[:15], "%Y%m%d_%H%M%S").replace(tzinfo=JST)
                                            formatted_created = dt2.strftime("%Y-%m-%d %H:%M:%S JST")
                                        except Exception:
                                            formatted_created = None
                            except Exception:
                                formatted_created = None
                            if not formatted_created:
                                formatted_created = datetime.fromtimestamp(
                                    model_dir.stat().st_mtime, tz=JST
                                ).strftime("%Y-%m-%d %H:%M:%S JST")
                            model_info["created"] = formatted_created
                            model_info["created_at"] = formatted_created
                            model_info["base_model"] = training_info.get(
                                "base_model", model_info.get("base_model", "Unknown")
                            )
                    except Exception:
                        pass

                # モデルタイプの判定
                if "continual_task" in model_dir.name.lower():
                    model_info["training_method"] = "continual"
                    model_info["type"] = "継続学習 (EWC)"
                    model_info["size"] = "~500MB+"
                elif "qlora" in model_dir.name.lower() or "4bit" in model_dir.name.lower():
                    model_info["training_method"] = "qlora"
                    model_info["type"] = "QLoRA (4bit)"
                    model_info["size"] = "~1.0MB"
                elif "lora" in model_dir.name.lower():
                    model_info["training_method"] = "lora"
                    model_info["type"] = "LoRA"
                    model_info["size"] = "~1.6MB"
                elif "フルファインチューニング" in model_dir.name:
                    model_info["training_method"] = "full"
                    model_info["type"] = "フルファインチューニング"
                    model_info["size"] = "~500MB+"
                elif "openagentrl" in model_dir.name.lower() or "_rl_" in model_dir.name.lower():
                    model_info["training_method"] = "rl"
                    model_info["type"] = "強化学習 (RL)"
                    model_info["size"] = "~15GB"

                models["finetuned_models"].append(model_info)
                logger.info(f"ファインチューニング済みモデルを検出: {model_dir.name}")

        # Ollamaモデルの検索
        ollama, available = _get_ollama()
        if available and ollama is not None:
            try:
                ollama_models = ollama.list_models()
                logger.debug(f"Ollamaモデル取得結果: {ollama_models}")

                if ollama_models.get("success", False):
                    for model in ollama_models.get("models", []):
                        modified_raw = model.get("modified", "Unknown")
                        modified_jst = modified_raw
                        if isinstance(modified_raw, str):
                            try:
                                iso = modified_raw.replace("Z", "+00:00")
                                dtm = datetime.fromisoformat(iso)
                                modified_jst = dtm.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST")
                            except Exception:
                                pass
                        models["ollama_models"].append({
                            "name": model.get("name", "Unknown"),
                            "type": "ollama",
                            "size": model.get("size", "Unknown"),
                            "modified": modified_jst,
                        })
                else:
                    logger.warning(f"Ollamaモデル取得失敗: {ollama_models.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Ollamaモデル取得エラー: {e}")
                import traceback as _tb
                logger.error(f"詳細エラー: {_tb.format_exc()}")
                models["ollama_models"] = []

        # RAG設定ファイルから利用可能なモデルを追加
        try:
            rag_config_path = Path("src/rag/config/rag_config.yaml")
            if rag_config_path.exists():
                with open(rag_config_path, "r", encoding="utf-8") as f:
                    rag_config = yaml.safe_load(f)
                cfg_models = rag_config.get("llm", {}).get("available_models", [])
                for model_name in cfg_models:
                    if model_name.startswith("finetuned:"):
                        # finetuned: プレフィックス付きモデルはファインチューニング済みモデルとして追加
                        ft_path = model_name.replace("finetuned:", "")
                        ft_dir_name = Path(ft_path).name
                        if not any(m["name"] == ft_dir_name for m in models["finetuned_models"]):
                            ft_info = {
                                "name": ft_dir_name,
                                "path": ft_path,
                                "type": "finetuned",
                                "size": "Unknown",
                                "created": "From config",
                            }
                            # config.json があればモデル情報を読み取る
                            ft_config_path = Path(ft_path) / "config.json"
                            if ft_config_path.exists():
                                try:
                                    with open(ft_config_path, "r", encoding="utf-8") as cf:
                                        ft_config = json.load(cf)
                                        ft_info["base_model"] = ft_config.get("_name_or_path", "Unknown")
                                        ft_info["model_type"] = ft_config.get("model_type", "Unknown")
                                        archs = ft_config.get("architectures", [])
                                        if any("qwen" in a.lower() for a in archs):
                                            ft_info["type"] = "強化学習 (RL)"
                                except Exception:
                                    pass
                            if "openagentrl" in ft_dir_name.lower() or "_rl_" in ft_dir_name.lower():
                                ft_info["type"] = "強化学習 (RL)"
                                ft_info["size"] = "~15GB"
                            models["finetuned_models"].append(ft_info)
                            logger.info(f"RAG設定からファインチューニングモデルを追加: {ft_dir_name}")
                    elif not any(m["name"] == model_name for m in models["ollama_models"]):
                        models["ollama_models"].append({
                            "name": model_name,
                            "type": "ollama",
                            "size": "Configured",
                            "modified": "From config",
                        })
                        logger.info(f"RAG設定からモデルを追加: {model_name}")
        except Exception as e:
            logger.warning(f"RAG設定ファイル読み込みエラー: {e}")

        return models

    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {str(e)}")
        return {"finetuned_models": [], "ollama_models": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Delete endpoints
# ---------------------------------------------------------------------------

@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """ファインチューニング済みモデルを削除"""
    import shutil

    try:
        if ".." in model_name or "/" in model_name or "\\" in model_name or "%2F" in model_name or "%2f" in model_name:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid model name"})

        model_path = Path("outputs") / model_name
        if not model_path.exists():
            return JSONResponse(status_code=404, content={"success": False, "error": f"Model '{model_name}' not found"})
        if not model_path.is_dir():
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid model path"})

        outputs_dir = Path("outputs").resolve()
        model_path_resolved = model_path.resolve()
        if not str(model_path_resolved).startswith(str(outputs_dir)):
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid model location"})

        logger.info(f"Deleting model: {model_name}")
        shutil.rmtree(model_path)
        logger.info(f"Model '{model_name}' deleted successfully")
        return {"success": True, "message": f"Model '{model_name}' deleted successfully"}

    except PermissionError:
        logger.error(f"Permission denied when deleting model: {model_name}")
        return JSONResponse(status_code=403, content={"success": False, "error": "Permission denied"})
    except Exception as e:
        logger.error(f"Error deleting model '{model_name}': {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@router.delete("/ollama/models/{model_name:path}")
async def delete_ollama_model(model_name: str):
    """Ollamaモデルを削除"""
    try:
        if ".." in model_name or model_name.startswith("/"):
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid model name"})

        logger.info(f"Deleting Ollama model: {model_name}")
        result = subprocess.run(["ollama", "rm", model_name], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            logger.info(f"Ollama model '{model_name}' deleted successfully")
            return {"success": True, "message": f"Ollama model '{model_name}' deleted successfully"}
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Failed to delete Ollama model '{model_name}': {error_msg}")
            return JSONResponse(status_code=400, content={"success": False, "error": error_msg})

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout when deleting Ollama model: {model_name}")
        return JSONResponse(status_code=504, content={"success": False, "error": "Operation timed out"})
    except FileNotFoundError:
        logger.error("Ollama command not found")
        return JSONResponse(status_code=503, content={"success": False, "error": "Ollama is not installed or not in PATH"})
    except Exception as e:
        logger.error(f"Error deleting Ollama model '{model_name}': {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-stream")
async def generate_text_stream(request: dict):
    """ストリーミング対応のテキスト生成"""
    import asyncio

    async def generate_stream():
        try:
            model_name = request.get("model_name")
            model_type = request.get("model_type")
            prompt = request.get("prompt")
            max_length = request.get("max_length", 2048)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.9)

            if not model_name or not model_type or not prompt:
                yield f"data: {json.dumps({'error': 'model_name, model_type, promptが必要です'})}\n\n"
                return

            # 校正モード検出: 校正リクエストの場合は専用モジュールに委譲
            proofread_mode = is_proofreading_request(prompt)
            if proofread_mode and PROOFREADING_AVAILABLE:
                try:
                    from src.proofreading import ProofreadingService
                    service = ProofreadingService()
                    report = await service.proofread_document(prompt)
                    result_text = json.dumps(report, ensure_ascii=False, indent=2)
                    yield f"data: {json.dumps({'text': result_text, 'done': True})}\n\n"
                    return
                except Exception as e:
                    logger.error(f"校正サービスエラー: {e}")

            effective_prompt = prompt
            char_limit = None

            logger.info(f"ストリーミング生成開始: {model_type}/{model_name}")

            if model_type == "ollama":
                ollama, available = _get_ollama()
                if not available or ollama is None:
                    yield f"data: {json.dumps({'error': 'Ollamaが利用できません'})}\n\n"
                    return

                import requests
                ollama_params = {
                    "model": model_name,
                    "prompt": effective_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_length,
                        # 長文入力を優先（サーバ側でも4096固定だが、ここは明示）
                        "num_ctx": 8192 if proofread_mode else 4096,
                    },
                }

                try:
                    response = requests.post(
                        f"{ollama.base_url}/api/generate", json=ollama_params, stream=True, timeout=300,
                    )
                    if response.status_code == 200:
                        emitted_chars = 0
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode("utf-8"))
                                if "response" in data:
                                    chunk = data["response"]
                                    if char_limit is not None:
                                        remaining = char_limit - emitted_chars
                                        if remaining <= 0:
                                            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                                            break
                                        if len(chunk) > remaining:
                                            chunk = chunk[:remaining]
                                    emitted_chars += len(chunk)
                                    if chunk:
                                        yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                                if data.get("done", False):
                                    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                                    break
                    else:
                        yield f"data: {json.dumps({'error': f'Ollama API エラー: {response.status_code}'})}\n\n"
                except Exception as e:
                    logger.error(f"Ollamaストリーミングエラー: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            elif model_type == "finetuned":
                model_path = f"outputs/{model_name}"
                try:
                    if model_path not in model_cache:
                        logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")
                        max_memory = {}
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                max_memory[i] = "18GB"
                            max_memory["cpu"] = "30GB"

                        tokenizer = load_tokenizer(model_path)
                        quantization_config = create_quantization_config(model_path, "lora", force_4bit=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            max_memory=max_memory,
                            trust_remote_code=True,
                        )
                        model_cache[model_path] = {"tokenizer": tokenizer, "model": model}

                    cached_model = model_cache[model_path]
                    tokenizer = cached_model["tokenizer"]
                    model = cached_model["model"]

                    # 長文入力を切り詰めすぎない（tokenizerの上限に合わせる）
                    input_max_len = _tokenizer_input_max_len(
                        tokenizer, 4096 if proofread_mode else 2048
                    )
                    inputs = tokenizer(
                        effective_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=input_max_len
                    )
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    model.eval()
                    device = inputs["input_ids"].device
                    logger.info(f"ストリーミング生成デバイス: {device}")

                    generated_tokens = []
                    current_text = ""
                    token_buffer = []
                    emitted_chars = 0

                    with torch.no_grad():
                        for _ in range(max_length):
                            outputs = model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask"),
                                max_new_tokens=1,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=temperature > 0.0,
                                temperature=temperature if temperature > 0.0 else 1.0,
                                top_p=top_p if temperature > 0.0 else 1.0,
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=3,
                            )

                            new_token = outputs[0][-1].unsqueeze(0).to(device)
                            generated_tokens.append(new_token)
                            token_buffer.append(new_token)

                            if len(token_buffer) >= 1:
                                try:
                                    buffer_tokens = torch.cat(token_buffer, dim=0)
                                    decoded_text = tokenizer.decode(buffer_tokens, skip_special_tokens=True)
                                    if len(current_text) < len(decoded_text):
                                        new_text = decoded_text[len(current_text):]
                                        current_text = decoded_text
                                        if new_text and new_text.strip():
                                            invalid_chars = ["", "\ufffd", "\u0000", "\u0001", "\u0002", "\u0003"]
                                            has_invalid = any(char in new_text for char in invalid_chars)
                                            if not has_invalid:
                                                chunk = new_text
                                                if char_limit is not None:
                                                    remaining = char_limit - emitted_chars
                                                    if remaining <= 0:
                                                        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                                                        break
                                                    if len(chunk) > remaining:
                                                        chunk = chunk[:remaining]
                                                emitted_chars += len(chunk)
                                                if chunk:
                                                    yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                                    token_buffer = []
                                except Exception as decode_error:
                                    logger.error(f"デコードエラー: {decode_error}")
                                    for token in token_buffer:
                                        try:
                                            single_text = tokenizer.decode(token, skip_special_tokens=True)
                                            if single_text and single_text.strip():
                                                invalid_chars = ["", "\ufffd", "\u0000", "\u0001", "\u0002", "\u0003"]
                                                has_invalid = any(char in single_text for char in invalid_chars)
                                                if not has_invalid:
                                                    yield f"data: {json.dumps({'text': single_text, 'done': False})}\n\n"
                                        except Exception as e:
                                            logger.error(f"個別デコードエラー: {e}")
                                    token_buffer = []

                            inputs["input_ids"] = torch.cat([inputs["input_ids"], new_token.unsqueeze(0)], dim=1)
                            if "attention_mask" in inputs:
                                ones = torch.ones(1, 1, dtype=torch.long, device=device)
                                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], ones], dim=1)

                            if new_token.item() == tokenizer.eos_token_id:
                                break
                            if char_limit is not None and emitted_chars >= char_limit:
                                break

                    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

                except Exception as e:
                    logger.error(f"ファインチューニング済みモデルストリーミングエラー: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'サポートされていないモデルタイプです'})}\n\n"

        except Exception as e:
            logger.error(f"ストリーミング生成エラー: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.post("/generate-with-model-selection")
async def generate_with_model_selection(request: dict):
    """モデル選択機能付きテキスト生成"""
    try:
        model_name = request.get("model_name")
        model_type = request.get("model_type")
        prompt = request.get("prompt")
        max_length = request.get("max_length", 2048)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)

        if not model_name or not model_type or not prompt:
            return {"success": False, "error": "model_name, model_type, promptが必要です"}

        # 校正モード検出: 校正リクエストの場合は専用モジュールに委譲
        proofread_mode = is_proofreading_request(prompt)
        if proofread_mode and PROOFREADING_AVAILABLE:
            try:
                from src.proofreading import ProofreadingService
                service = ProofreadingService()
                report = await service.proofread_document(prompt)
                return {
                    "success": True,
                    "generated_text": json.dumps(report, ensure_ascii=False, indent=2),
                    "model_name": model_name,
                    "model_type": model_type,
                    "method": "proofreading_service",
                    "proofreading_report": report
                }
            except Exception as e:
                logger.error(f"校正サービスエラー: {e}")

        effective_prompt = prompt
        effective_max_tokens = int(max_length)

        logger.info(f"モデル選択生成: {model_type}/{model_name}")

        if model_type == "ollama":
            ollama, available = _get_ollama()
            if not available or ollama is None:
                return {"success": False, "error": "Ollamaが利用できません"}

            result = ollama.generate_text(
                model_name=model_name,
                prompt=effective_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                # 長文の読み込み量（コンテキスト）を増やす
                num_ctx=8192 if proofread_mode else 4096,
            )
            if result["success"]:
                generated_text = result["generated_text"]
                if proofread_mode:
                    generated_text = _truncate_chars(generated_text, limit=5200)
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "model_type": "ollama",
                    "method": "ollama_api",
                }
            else:
                return {"success": False, "error": result.get("error", "Ollama生成エラー")}

        elif model_type == "finetuned":
            model_path = f"outputs/{model_name}"

            # メモリ不足の場合はOllamaにフォールバック
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 30 and _deps.OLLAMA_AVAILABLE:
                    ollama_model_name = "llama3.2:3b"
                    logger.info(f"メモリ不足のため、Ollamaモデル {ollama_model_name} を使用します")
                    ollama, available = _get_ollama()
                    if available and ollama is not None:
                        result = ollama.generate_text(
                            model_name=ollama_model_name, prompt=effective_prompt, temperature=temperature,
                            top_p=top_p, max_tokens=effective_max_tokens,
                            num_ctx=8192 if proofread_mode else 4096,
                        )
                        if result["success"]:
                            gen_txt = result["generated_text"]
                            if proofread_mode:
                                gen_txt = _truncate_chars(gen_txt, limit=5200)
                            return {
                                "success": True,
                                "generated_text": gen_txt,
                                "model_name": f"{model_name} (Ollama fallback: {ollama_model_name})",
                                "model_type": "finetuned_ollama_fallback",
                                "method": "ollama_fallback",
                            }

            try:
                if model_path not in model_cache:
                    logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")
                    max_memory = {}
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            max_memory[i] = "18GB"
                        max_memory["cpu"] = "30GB"

                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        max_memory=max_memory,
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        offload_folder="./offload",
                        offload_state_dict=True,
                    )
                    tokenizer = load_tokenizer(model_path)
                    model_cache[model_path] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "base_model_name": model_path,
                        "training_method": "full",
                    }
                    logger.info(f"モデル読み込み完了: {model_path}")

                cached_model = model_cache[model_path]
                model = cached_model["model"]
                tokenizer = cached_model["tokenizer"]

                # 長文入力を切り詰めすぎない（tokenizerの上限に合わせる）
                input_max_len = _tokenizer_input_max_len(
                    tokenizer, 4096 if proofread_mode else 2048
                )
                inputs = tokenizer(
                    effective_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=input_max_len
                )
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=int(effective_max_tokens),
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=(not proofread_mode) and (temperature is not None and float(temperature) > 0.0),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_len = inputs.input_ids.shape[1]
                generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                if proofread_mode:
                    generated_text = _truncate_chars(generated_text, limit=5200)

                logger.info(f"ファインチューニング済みモデルでの生成完了: {len(generated_text)}文字")
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "model_type": "finetuned",
                    "method": "direct_inference",
                }

            except Exception as model_error:
                logger.error(f"ファインチューニング済みモデル生成エラー: {str(model_error)}")
                return {"success": False, "error": f"モデル生成エラー: {str(model_error)}"}

        else:
            return {"success": False, "error": f"未知のモデルタイプ: {model_type}"}

    except Exception as e:
        logger.error(f"モデル選択生成エラー: {str(e)}")
        return {"success": False, "error": str(e)}
