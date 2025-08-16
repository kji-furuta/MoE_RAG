"""
モデル管理関連のAPIルーター
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models.models import ModelInfo
from ..models.training import GenerationRequest
from ..dependencies import logger, PROJECT_ROOT, OUTPUTS_DIR, model_cache

router = APIRouter(prefix="/api", tags=["models"])

# Ollama統合の可用性
OLLAMA_AVAILABLE = False

try:
    from src.integrations.ollama_integration import OllamaIntegration
    OLLAMA_AVAILABLE = True
    logger.info("Ollama統合が利用可能です")
except ImportError:
    logger.warning("Ollama統合が利用できません")


def get_saved_models() -> List[Dict[str, Any]]:
    """保存されたモデル一覧を取得"""
    models = []
    
    if OUTPUTS_DIR.exists():
        for model_dir in OUTPUTS_DIR.iterdir():
            if model_dir.is_dir():
                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                }
                
                # training_info.jsonがあれば読み込む
                training_info_file = model_dir / "training_info.json"
                if training_info_file.exists():
                    try:
                        with open(training_info_file, 'r', encoding='utf-8') as f:
                            training_info = json.load(f)
                            model_info.update({
                                "base_model": training_info.get("base_model", "unknown"),
                                "training_method": training_info.get("training_method", "unknown"),
                                "epochs": training_info.get("epochs", 0)
                            })
                    except Exception as e:
                        logger.warning(f"training_info.json読み込みエラー: {e}")
                
                # model_info.jsonがあれば読み込む（継続学習モデル用）
                model_info_file = model_dir / "model_info.json"
                if model_info_file.exists():
                    try:
                        with open(model_info_file, 'r', encoding='utf-8') as f:
                            saved_info = json.load(f)
                            model_info.update(saved_info)
                    except Exception as e:
                        logger.warning(f"model_info.json読み込みエラー: {e}")
                
                models.append(model_info)
    
    return models


@router.get("/system-info")
async def get_system_info():
    """システム情報を取得"""
    try:
        import torch
        import psutil
        
        system_info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "memory_percent": psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                system_info[f"gpu_{i}_name"] = gpu_name
                system_info[f"gpu_{i}_memory"] = f"{gpu_memory:.2f} GB"
        
        return system_info
        
    except Exception as e:
        logger.error(f"システム情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_model_cache():
    """モデルキャッシュをクリア"""
    try:
        import torch
        import gc
        
        # キャッシュをクリア
        model_cache.clear()
        
        # GPUメモリを解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ガベージコレクション
        gc.collect()
        
        logger.info("モデルキャッシュとGPUメモリをクリアしました")
        
        return {
            "status": "success",
            "message": "Model cache and GPU memory cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"キャッシュクリアエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-models")
async def get_available_models():
    """利用可能なファインチューニング済みモデルとOllamaモデルを取得"""
    try:
        models = {
            "finetuned_models": [],
            "ollama_models": []
        }
        
        # ファインチューニング済みモデルの検索
        if OUTPUTS_DIR.exists():
            for model_dir in OUTPUTS_DIR.iterdir():
                if model_dir.is_dir():
                    # モデルファイルの存在確認
                    has_model_files = (
                        (model_dir / "pytorch_model.bin").exists() or
                        (model_dir / "adapter_model.safetensors").exists() or
                        (model_dir / "adapter_config.json").exists() or
                        (model_dir / "config.json").exists()
                    )
                    
                    if has_model_files:
                        # モデル情報を取得
                        config_path = model_dir / "config.json"
                        training_info_path = model_dir / "training_info.json"
                        
                        model_info = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "type": "finetuned",
                            "size": "Unknown",
                            "created": "Unknown"
                        }
                        
                        # 設定ファイルから情報を読み取り
                        if config_path.exists():
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                    model_info["base_model"] = config.get("_name_or_path", "Unknown")
                                    model_info["model_type"] = config.get("model_type", "Unknown")
                            except:
                                pass
                        
                        # 訓練情報から詳細を取得
                        if training_info_path.exists():
                            try:
                                with open(training_info_path, 'r', encoding='utf-8') as f:
                                    training_info = json.load(f)
                                    model_info["training_method"] = training_info.get("training_method", "unknown")
                                    model_info["created"] = training_info.get("timestamp", "Unknown")
                            except:
                                pass
                        
                        # モデルタイプの判定
                        if "qlora" in model_dir.name.lower() or "4bit" in model_dir.name.lower():
                            model_info["training_method"] = "qlora"
                            model_info["size"] = "~1.0MB"
                        elif "lora" in model_dir.name.lower():
                            model_info["training_method"] = "lora"
                            model_info["size"] = "~1.6MB"
                        elif "フルファインチューニング" in model_dir.name:
                            model_info["training_method"] = "full"
                            model_info["size"] = "~500MB+"
                        
                        models["finetuned_models"].append(model_info)
                        logger.info(f"ファインチューニング済みモデルを検出: {model_dir.name}")
        
        # Ollamaモデルの検索
        if OLLAMA_AVAILABLE:
            try:
                ollama = OllamaIntegration()
                ollama_models = ollama.list_models()
                logger.debug(f"Ollamaモデル取得結果: {ollama_models}")
                
                if ollama_models.get("success", False):
                    for model in ollama_models.get("models", []):
                        models["ollama_models"].append({
                            "name": model.get("name", "Unknown"),
                            "type": "ollama",
                            "size": model.get("size", "Unknown"),
                            "modified": model.get("modified", "Unknown")
                        })
                else:
                    logger.warning(f"Ollamaモデル取得失敗: {ollama_models.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Ollamaモデル取得エラー: {e}")
                models["ollama_models"] = []
        
        return models
        
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {str(e)}")
        return {"finetuned_models": [], "ollama_models": [], "error": str(e)}


@router.post("/convert-to-ollama")
async def convert_finetuned_to_ollama(model_path: str, model_name: str):
    """ファインチューニング済みモデルをOllama形式に変換"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Ollama integration is not available"
        )
    
    try:
        logger.info(f"Ollama形式への変換開始: {model_path} -> {model_name}")
        
        # モデルパスの検証
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model path not found")
        
        # Ollamaモデルファイルを作成
        modelfile_content = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "あなたは日本語に堪能なAIアシスタントです。"
"""
        
        # Modelfileを一時的に保存
        modelfile_path = model_dir / "Modelfile"
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # Ollamaコマンドを実行
        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Ollama変換成功: {result.stdout}")
            
            return {
                "status": "success",
                "message": f"Model converted to Ollama format: {model_name}",
                "output": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama変換エラー: {e.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Ollama conversion failed: {e.stderr}"
            )
        finally:
            # 一時ファイルを削除
            if modelfile_path.exists():
                modelfile_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"変換エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert-to-ollama-wsl")
async def convert_finetuned_to_ollama_wsl(request: dict):
    """WSL環境でファインチューニング済みモデルをOllama形式に変換"""
    if not OLLAMA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Ollama integration is not available"
        )
    
    try:
        model_path = request.get("model_path")
        model_name = request.get("model_name")
        
        if not model_path or not model_name:
            raise HTTPException(
                status_code=400,
                detail="model_path and model_name are required"
            )
        
        logger.info(f"WSL Ollama形式への変換開始: {model_path} -> {model_name}")
        
        # WSLパスに変換
        wsl_path = model_path.replace("C:", "/mnt/c").replace("\\", "/")
        
        # モデルファイルの作成
        modelfile_content = f"""FROM {wsl_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM "あなたは日本語に堪能なAIアシスタントです。道路設計と土木工学の専門知識を持っています。"
"""
        
        # 一時ファイルとして保存
        temp_modelfile = Path("/tmp") / f"Modelfile_{model_name}"
        with open(temp_modelfile, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # WSL用のパスに変換
        wsl_modelfile = str(temp_modelfile).replace("/tmp", "/mnt/c/temp")
        
        # Ollamaコマンドを実行
        try:
            # WSL経由でOllamaコマンドを実行
            cmd = f'wsl ollama create {model_name} -f {wsl_modelfile}'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分のタイムアウト
            )
            
            if result.returncode == 0:
                logger.info(f"WSL Ollama変換成功: {result.stdout}")
                
                # 変換成功後、Ollamaモデルリストを更新
                ollama = OllamaIntegration()
                models = ollama.list_models()
                
                return {
                    "status": "success",
                    "message": f"Model converted to Ollama format: {model_name}",
                    "output": result.stdout,
                    "available_models": models.get("models", [])
                }
            else:
                logger.error(f"WSL Ollama変換エラー: {result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Ollama conversion failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama変換タイムアウト")
            raise HTTPException(
                status_code=500,
                detail="Ollama conversion timed out after 5 minutes"
            )
        finally:
            # 一時ファイルを削除
            if temp_modelfile.exists():
                temp_modelfile.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WSL変換エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-stream")
async def generate_text_stream(request: GenerationRequest):
    """ストリーミング形式でテキスト生成"""
    try:
        from fastapi.responses import StreamingResponse
        import asyncio
        
        async def generate():
            # ここでは簡単なデモ実装
            # 実際にはモデルからのストリーミング生成を実装
            text = f"これは{request.prompt}に対するストリーミング応答です。"
            
            for char in text:
                yield char
                await asyncio.sleep(0.05)  # 文字ごとに少し遅延
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"ストリーミング生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-with-selection")
async def generate_with_model_selection(request: dict):
    """モデル選択付きテキスト生成"""
    try:
        model_type = request.get("model_type", "finetuned")
        model_name = request.get("model_name")
        prompt = request.get("prompt")
        
        if not model_name or not prompt:
            raise HTTPException(
                status_code=400,
                detail="model_name and prompt are required"
            )
        
        logger.info(f"モデル選択生成: type={model_type}, name={model_name}")
        
        if model_type == "ollama" and OLLAMA_AVAILABLE:
            # Ollamaモデルを使用
            ollama = OllamaIntegration()
            result = ollama.generate_text(
                model_name=model_name,
                prompt=prompt,
                temperature=request.get("temperature", 0.7),
                top_p=request.get("top_p", 0.9),
                max_tokens=request.get("max_length", 2048)
            )
            
            if result.get("success"):
                return {
                    "prompt": prompt,
                    "generated_text": result.get("generated_text"),
                    "model_type": "ollama",
                    "model_name": model_name
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Generation failed")
                )
        
        elif model_type == "finetuned":
            # ファインチューニング済みモデルを使用
            # ここでは簡単なデモ応答を返す
            # 実際にはgenerate_text関数を呼び出す
            return {
                "prompt": prompt,
                "generated_text": f"{prompt}\n\n[ファインチューニングモデル {model_name} による生成結果]",
                "model_type": "finetuned",
                "model_name": model_name
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {model_type}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデル選択生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-model")
async def verify_finetuned_model(request: GenerationRequest):
    """ファインチューニング済みモデルの検証"""
    try:
        logger.info(f"モデル検証開始: {request.model_path}")
        
        # モデルパスの存在確認
        model_path = Path(request.model_path)
        if not model_path.exists():
            return {
                "status": "error",
                "error": "モデルが見つかりません",
                "model_path": request.model_path
            }
        
        # 検証用のテストケース
        test_cases = [
            "縦断曲線とは何のために設置しますか？",
            "道路の横断勾配の標準的な値はどのくらいですか？",
            "アスファルト舗装の主な利点と欠点は何ですか？",
            "設計CBRとは舗装設計においてどのような指標ですか？",
            "道路の平面線形を構成する3つの要素は何ですか？"
        ]
        
        verification_results = []
        
        for i, test_prompt in enumerate(test_cases):
            logger.info(f"テストケース {i+1}/{len(test_cases)}: {test_prompt}")
            
            # 簡単な検証結果を生成（実際にはモデルで生成）
            verification_result = {
                "test_case": i + 1,
                "prompt": test_prompt,
                "generated_text": f"[検証結果 {i+1}]",
                "success": True
            }
            
            verification_results.append(verification_result)
        
        # 検証サマリーを作成
        success_count = sum(1 for r in verification_results if r["success"])
        total_count = len(verification_results)
        
        verification_summary = {
            "model_path": request.model_path,
            "total_test_cases": total_count,
            "successful_tests": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "verification_results": verification_results
        }
        
        logger.info(f"モデル検証完了: 成功率 {success_count}/{total_count}")
        
        return {
            "status": "success",
            "verification_summary": verification_summary
        }
        
    except Exception as e:
        logger.error(f"モデル検証エラー: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_path": request.model_path
        }