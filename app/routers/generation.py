"""Text generation and model verification endpoints."""

from __future__ import annotations

import gc
import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from fastapi import APIRouter
from transformers import AutoModelForCausalLM

from ..dependencies import logger, model_cache
from ..model_utils import load_tokenizer, create_quantization_config, get_device_map
from ..training.models import GenerationRequest
from ..exceptions import ModelLoadError, ModelMemoryError, GenerationError, OllamaError
from .proofreading import (
    PROOFREADING_AVAILABLE,
    is_proofreading_request,
    _truncate_chars,
    _tokenizer_input_max_len,
)
import app.dependencies as _deps

JST = timezone(timedelta(hours=9))

router = APIRouter(prefix="/api", tags=["generation"])


def _get_ollama():
    """Lazy-load OllamaIntegration, returns (instance, True) or (None, False)."""
    if not getattr(_deps, "OLLAMA_AVAILABLE", False):
        return None, False
    try:
        import sys
        scripts_convert_path = Path(__file__).parent.parent.parent / "scripts" / "convert"
        if str(scripts_convert_path) not in sys.path:
            sys.path.insert(0, str(scripts_convert_path))
        from ollama_integration import OllamaIntegration
        return OllamaIntegration(), True
    except Exception:
        return None, False


@router.post("/generate")
async def generate_text(request: GenerationRequest):
    """実際のファインチューニング済みモデルを使用したテキスト生成"""
    try:
        logger.info(f"テキスト生成開始: モデル={request.model_path}, プロンプト={request.prompt[:50]}...")

        # 校正モード検出: 校正リクエストの場合は専用モジュールに委譲
        proofread_mode = is_proofreading_request(request.prompt)
        if proofread_mode and PROOFREADING_AVAILABLE:
            try:
                from src.proofreading import ProofreadingService
                service = ProofreadingService()
                report = await service.proofread_document(request.prompt)
                return {
                    "prompt": request.prompt,
                    "generated_text": json.dumps(report, ensure_ascii=False, indent=2),
                    "model_path": request.model_path,
                    "method": "proofreading_service",
                    "proofreading_report": report
                }
            except Exception as e:
                logger.error(f"校正サービスエラー: {e}")
                # フォールバック: 従来のテキスト生成に進む

        effective_prompt = request.prompt

        model_path = Path(request.model_path)

        # モデルパスの存在確認
        if not model_path.exists() or not model_path.is_dir():
            logger.warning(f"モデルパスが存在しません: {model_path}")
            return {
                "prompt": request.prompt,
                "generated_text": request.prompt + " [エラー: モデルパスが見つかりません]",
                "model_path": request.model_path,
                "error": "モデルパスが存在しません"
            }

        # キャッシュキー
        cache_key = str(model_path)

        # メモリ不足を事前にチェック
        if torch.cuda.is_available():
            # 現在の空きメモリを確認
            free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"現在のGPU空きメモリ: {free_memory:.2f} GB")

            # 32Bモデルは最低でも10GBの空きメモリが必要
            if free_memory < 10 and getattr(_deps, "OLLAMA_AVAILABLE", False):
                logger.info("メモリ不足のため、直接Ollamaにフォールバックします")
                try:
                    ollama, ok = _get_ollama()
                    if ok:
                        result = ollama.generate_text(
                            model_name="llama3.2:3b",
                            prompt=effective_prompt,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_length
                        )

                        if result.get("success", False):
                            generated_text = result.get("generated_text", "")
                            if proofread_mode:
                                generated_text = _truncate_chars(generated_text, limit=5200)
                            return {
                                "prompt": request.prompt,
                                "generated_text": generated_text,
                                "model_path": request.model_path,
                                "method": "ollama",
                                "note": "GPUメモリ不足のため、Ollamaモデルを使用しました"
                            }
                except Exception as e:
                    logger.error(f"Ollamaフォールバック失敗: {e}")

        # モデルがキャッシュにない場合は読み込み
        if cache_key not in model_cache:
            # メモリ不足を防ぐため、既存のキャッシュをクリア
            if len(model_cache) > 0:
                logger.info("メモリ節約のため既存のモデルキャッシュをクリア")
                for key in list(model_cache.keys()):
                    if key != cache_key:  # 現在のモデル以外をクリア
                        del model_cache[key]
                torch.cuda.empty_cache()
                gc.collect()

            logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")

            try:
                # トレーニング情報を読み込み
                training_info_path = model_path / "training_info.json"
                base_model_name = "distilgpt2"  # デフォルト
                training_method = "lora"

                if training_info_path.exists():
                    with open(training_info_path, 'r', encoding='utf-8') as f:
                        training_info = json.load(f)
                        base_model_name = training_info.get("base_model", "distilgpt2")
                        training_method = training_info.get("training_method", "lora")
                        logger.info(f"ベースモデル: {base_model_name}, メソッド: {training_method}")

                # トークナイザーの読み込み
                try:
                    tokenizer = load_tokenizer(str(model_path))
                    logger.info("ファインチューニング済みトークナイザーを読み込み")
                except Exception as e:
                    logger.warning(f"ファインチューニング済みトークナイザーの読み込みに失敗: {e}")
                    # ベースモデルのトークナイザーを使用
                    tokenizer = load_tokenizer(base_model_name)
                    logger.info(f"ベースモデルのトークナイザーを使用: {base_model_name}")

                # モデルの読み込み
                if training_method in ["lora", "qlora"]:
                    # LoRA/QLoRAモデルの場合
                    from peft import PeftModel

                    # ベースモデルを読み込み
                    logger.info(f"ベースモデルを読み込み中: {base_model_name}")

                    # 大きなモデルの場合は量子化を使用
                    quantization_config = create_quantization_config(base_model_name, "lora")
                    device_map = get_device_map(base_model_name)

                    model_kwargs = {
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }

                    if quantization_config:
                        model_kwargs["quantization_config"] = quantization_config
                    if device_map:
                        model_kwargs["device_map"] = device_map

                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        **model_kwargs
                    )

                    # LoRAアダプターを読み込み
                    logger.info(f"LoRAアダプターを読み込み中: {model_path}")
                    model = PeftModel.from_pretrained(base_model, str(model_path))
                    logger.info("LoRAモデルをロード完了。GPUへ転送します。")
                    model.to("cuda")
                    logger.info("モデルをGPUへ転送しました。")

                else:
                    # フルファインチューニングの場合
                    if torch.cuda.is_available():
                        # GPUメモリをクリア
                        torch.cuda.empty_cache()

                        # オフロードフォルダを作成
                        offload_dir = Path("offload")
                        offload_dir.mkdir(exist_ok=True)
                        logger.info(f"オフロードディレクトリを作成: {offload_dir}")

                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        logger.info(f"利用可能なGPUメモリ: {gpu_memory:.2f} GB")

                        # 32Bモデルの推論時メモリ効率化
                        from transformers import BitsAndBytesConfig

                        # 推論時は常に4bit量子化を使用（メモリ効率重視）
                        logger.info("推論時メモリ効率化: 4bit量子化を適用")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_enable_fp32_cpu_offload=True
                        )

                        # デバイスマップを最適化
                        free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                        safe_memory = max(1, int(free_memory_gb * 0.8))  # 80%を使用

                        max_memory = {
                            0: f"{safe_memory}GB",
                            "cpu": "32GB"
                        }

                        # メモリ不足対策の強化
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            logger.info("フルファインチューニングモデルを量子化付きで読み込み中...")
                            model = AutoModelForCausalLM.from_pretrained(
                                str(model_path),
                                quantization_config=quantization_config,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                max_memory=max_memory,
                                trust_remote_code=True,
                                offload_folder=str(offload_dir)
                            )
                            logger.info("フルファインチューニングモデルの読み込み完了（4bit量子化）")

                        except Exception as e:
                            logger.warning(f"量子化読み込み失敗: {str(e)}")
                            logger.info("Ollamaフォールバックを試行...")

                            # Ollamaフォールバックを試行
                            if getattr(_deps, "OLLAMA_AVAILABLE", False):
                                try:
                                    logger.info("Ollamaフォールバックを試行中...")
                                    ollama_integration, ok = _get_ollama()

                                    if ok:
                                        # 利用可能なOllamaモデルを確認
                                        available_models = ollama_integration.list_models()
                                        logger.info(f"利用可能なOllamaモデル: {available_models}")

                                        # 利用可能なOllamaモデルから選択
                                        ollama_model_name = "llama3.2:3b"  # 直接指定
                                        logger.info(f"Ollamaモデル {ollama_model_name} を使用します")

                                        # Ollamaでテキスト生成
                                        result = ollama_integration.generate_text(
                                            model_name=ollama_model_name,
                                            prompt=effective_prompt,
                                            temperature=request.temperature,
                                            top_p=request.top_p,
                                            max_tokens=request.max_length
                                        )

                                        if result.get("success", False):
                                            logger.info("Ollamaフォールバック成功")
                                            generated_text = result.get("generated_text", "Ollama生成エラー")
                                            if proofread_mode:
                                                generated_text = _truncate_chars(generated_text, limit=5200)
                                            return {
                                                "prompt": request.prompt,
                                                "generated_text": generated_text,
                                                "model_path": request.model_path,
                                                "fallback": "ollama",
                                                "method": "ollama",
                                                "note": "GPUメモリ不足のため、Ollamaモデルで生成しました"
                                            }
                                        else:
                                            logger.warning(f"Ollama生成失敗: {result.get('error', 'Unknown error')}")
                                except Exception as ollama_error:
                                    logger.error(f"Ollamaフォールバック失敗: {str(ollama_error)}")
                                    logger.error(f"Ollamaエラー詳細: {traceback.format_exc()}")

                            # 最終手段: CPUモード
                            logger.info("最終手段: CPUモードで読み込み中...")
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    str(model_path),
                                    torch_dtype=torch.float32,
                                    device_map=None,
                                    low_cpu_mem_usage=True,
                                    trust_remote_code=True
                                )
                                logger.info("CPUモードでの読み込み成功")
                            except Exception as final_error:
                                logger.error(f"全ての読み込み方法が失敗: {str(final_error)}")
                                return {
                                    "prompt": request.prompt,
                                    "generated_text": request.prompt + " [エラー: モデル読み込み失敗 - GPUメモリ不足]",
                                    "model_path": request.model_path,
                                    "error": f"モデル読み込み失敗: {str(final_error)}"
                                }
                    else:
                        # CPUモードでの実行
                        logger.info("CPUモードでモデルを読み込みます（GPUが利用できない場合）")
                        model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            torch_dtype=torch.float32,
                            device_map=None,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        logger.info("フルファインチューニングモデルの読み込み完了（CPU）")

                # キャッシュに保存
                model_cache[cache_key] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "base_model_name": base_model_name,
                    "training_method": training_method
                }
                logger.info("モデルキャッシュに保存完了")

            except Exception as model_error:
                logger.error(f"モデル読み込みエラー: {str(model_error)}")
                logger.error(traceback.format_exc())

                # GPUメモリ不足の場合の推奨事項を追加
                error_message = str(model_error)
                if "CUDA out of memory" in error_message:
                    recommendation = """
                    GPUメモリ不足のため、以下の対策を試してください：
                    1. より小さなモデル（7Bや14B）を使用する
                    2. 他のアプリケーションを終了してGPUメモリを解放する
                    3. CPUモードで実行する（速度は遅くなります）
                    """
                    error_message += recommendation

                return {
                    "prompt": request.prompt,
                    "generated_text": request.prompt + f" [エラー: モデル読み込み失敗 - {error_message}]",
                    "model_path": request.model_path,
                    "error": error_message
                }

        # ファインチューニング済みモデルの検証用テキスト生成
        logger.info("ファインチューニング済みモデルでテキスト生成を実行します")

        # Ollamaが利用可能で、メモリ不足の場合のフォールバック
        if getattr(_deps, "OLLAMA_AVAILABLE", False):
            # GPUメモリを確認
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                    logger.info(f"GPUメモリ: 合計 {gpu_memory:.1f}GB, 空き {free_memory:.1f}GB")

                    # 空きメモリが5GB未満の場合はOllamaを使用
                    if free_memory < 5:
                        logger.info(f"GPUメモリ不足（空き{free_memory:.1f}GB）のため、Ollamaを使用します")
                        ollama, ok = _get_ollama()

                        if ok:
                            # 利用可能なOllamaモデルを確認
                            available_models = ollama.list_models()
                            logger.info(f"利用可能なOllamaモデル: {available_models}")

                            # 使用するOllamaモデルを選択
                            ollama_model_name = None
                            if available_models.get("models"):
                                for model_info in available_models["models"]:
                                    model_name_str = model_info.get("name", "")
                                    if "llama3.2:3b" in model_name_str:
                                        ollama_model_name = model_name_str
                                        break
                                if not ollama_model_name and available_models["models"]:
                                    ollama_model_name = available_models["models"][0].get("name")

                            if ollama_model_name:
                                logger.info(f"Ollamaモデル {ollama_model_name} を使用します")

                                # Ollamaでテキスト生成
                                result = ollama.generate_text(
                                    model_name=ollama_model_name,
                                    prompt=effective_prompt,
                                    temperature=request.temperature,
                                    top_p=request.top_p,
                                    max_tokens=request.max_length
                                )

                                if result.get("success", False):
                                    logger.info("Ollamaでの生成が成功しました")
                                    generated_text = result.get("generated_text", "Ollama生成エラー")
                                    if proofread_mode:
                                        generated_text = _truncate_chars(generated_text, limit=5200)
                                    return {
                                        "prompt": request.prompt,
                                        "generated_text": generated_text,
                                        "model_path": request.model_path,
                                        "method": "ollama",
                                        "note": "GPUメモリ不足のため、Ollamaモデルで生成しました",
                                        "verification_info": {
                                            "model_path": request.model_path,
                                            "base_model": "ollama-converted",
                                            "training_method": "full",
                                            "prompt": request.prompt,
                                            "generation_params": {
                                                "max_length": request.max_length,
                                                "temperature": request.temperature,
                                                "top_p": request.top_p
                                            }
                                        }
                                    }
                                else:
                                    logger.warning(f"Ollama生成失敗: {result.get('error', 'Unknown error')}")
                                    # フォールバック: 通常の方法を試行
            except Exception as ollama_error:
                logger.error(f"Ollama統合エラー: {str(ollama_error)}")
                logger.error(f"Ollamaエラー詳細: {traceback.format_exc()}")
                    # フォールバック: 通常の方法を試行

        # 通常の方法（Transformers）を使用
        # まずキャッシュからモデルを取得
        cached_model = model_cache.get(cache_key, {})

        # モデル情報の記録
        model_info = {
            "model_path": request.model_path,
            "base_model": cached_model.get("base_model_name", "unknown"),
            "training_method": cached_model.get("training_method", "unknown"),
            "prompt": request.prompt,
            "generation_params": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }

        logger.info(f"モデル情報: {model_info}")
        tokenizer = cached_model["tokenizer"]
        model = cached_model["model"]

        # テキスト生成
        logger.info("テキスト生成を実行中...")

        # プロンプトのトークナイズ（長文を切り詰めすぎない）
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

        # GPUに移動
        if torch.cuda.is_available() and hasattr(model, 'device'):
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except:
                # デバイス移動に失敗した場合はそのまま続行
                pass

        # ファインチューニング済みモデルの検証用テキスト生成
        model.eval()
        with torch.no_grad():
            try:
                # 生成パラメータの設定
                generation_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    # UIのmax_lengthは「生成する新規トークン数」として扱う
                    'max_new_tokens': int(request.max_length),
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'repetition_penalty': 1.2,  # 繰り返しペナルティを追加
                    'no_repeat_ngram_size': 3,  # 3-gramの繰り返しを防ぐ
                }

                # サンプリングを使用する場合のみtemperatureとtop_pを設定
                if request.temperature > 0.0:
                    generation_kwargs['do_sample'] = True
                    generation_kwargs['temperature'] = request.temperature
                    generation_kwargs['top_p'] = request.top_p
                else:
                    generation_kwargs['do_sample'] = False

                logger.info(f"生成パラメータ: {generation_kwargs}")
                logger.info("model.generate()を実行中...")

                outputs = model.generate(**generation_kwargs)

                logger.info(f"生成完了: 出力トークン数={outputs.shape}")

                # トークンに関する情報を記録
                generated_ids = outputs[0]  # 生成されたIDを記録
                input_length = len(inputs['input_ids'][0])  # 入力トークン数を記録

                # 入力部分を除いた生成結果のみを取り出す
                input_len = inputs["input_ids"].shape[1]
                generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                logger.info(f"デコード完了: テキスト長={len(generated_text)}")

                # 校正モードの場合は5000文字程度に寄せる（上限を超えたら切り詰め）
                if proofread_mode:
                    generated_text = _truncate_chars(generated_text, limit=5200)

                logger.info(f"テキスト生成完了: {len(generated_text)}文字")

                # 生成結果をファイルに保存
                try:
                    project_root = Path(os.getcwd())
                    outputs_dir = project_root / "outputs"
                    outputs_dir.mkdir(exist_ok=True)

                    # タイムスタンプ付きファイル名
                    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
                    model_name = Path(request.model_path).name
                    output_filename = f"generated_text_{model_name}_{timestamp}.json"
                    output_path = outputs_dir / output_filename

                    # 生成結果を保存
                    generation_result = {
                        "timestamp": timestamp,
                        "model_path": request.model_path,
                        "base_model": cached_model.get("base_model_name", "unknown"),
                        "training_method": cached_model.get("training_method", "unknown"),
                        "prompt": request.prompt,
                        "generated_text": generated_text,
                        "parameters": {
                            "max_length": request.max_length,
                            "temperature": request.temperature,
                            "top_p": request.top_p
                        }
                    }

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(generation_result, f, indent=2, ensure_ascii=False)

                    logger.info(f"生成結果を保存: {output_path}")

                except Exception as save_error:
                    logger.warning(f"生成結果の保存に失敗: {save_error}")

                # 検証結果の詳細情報を記録
                verification_info = {
                    "model_path": request.model_path,
                    "base_model": cached_model.get("base_model_name", "unknown"),
                    "training_method": cached_model.get("training_method", "unknown"),
                    "prompt": request.prompt,
                    "generated_text": generated_text,
                    "generation_params": {
                        "max_length": request.max_length,
                        "temperature": request.temperature,
                        "top_p": request.top_p
                    },
                    "model_info": {
                        "total_tokens": len(generated_ids),
                        "input_tokens": input_length,
                        "generated_tokens": len(generated_ids) - input_length
                    }
                }

                # 検証結果をログに記録
                logger.info(f"ファインチューニング済みモデル検証結果: {verification_info}")

                return {
                    "prompt": request.prompt,
                    "generated_text": generated_text,
                    "model_path": request.model_path,
                    "base_model": cached_model.get("base_model_name", "unknown"),
                    "training_method": cached_model.get("training_method", "unknown"),
                    "verification_info": verification_info
                }

            except Exception as gen_error:
                logger.error(f"テキスト生成エラー: {str(gen_error)}")
                logger.error(traceback.format_exc())
                return {
                    "prompt": request.prompt,
                    "generated_text": request.prompt + f" [エラー: 生成失敗 - {str(gen_error)}]",
                    "model_path": request.model_path,
                    "error": str(gen_error)
                }

    except (ModelLoadError, ModelMemoryError) as e:
        logger.error("モデルエラー: %s", e, exc_info=True)
        return {
            "prompt": request.prompt,
            "generated_text": request.prompt + f" [モデルエラー: {e}]",
            "model_path": request.model_path,
            "error": str(e)
        }
    except (GenerationError, OllamaError) as e:
        logger.error("生成エラー: %s", e, exc_info=True)
        return {
            "prompt": request.prompt,
            "generated_text": request.prompt + f" [生成エラー: {e}]",
            "model_path": request.model_path,
            "error": str(e)
        }
    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU OOM: %s", e, exc_info=True)
        torch.cuda.empty_cache()
        return {
            "prompt": request.prompt,
            "generated_text": request.prompt + " [エラー: GPUメモリ不足]",
            "model_path": request.model_path,
            "error": f"GPU out of memory: {e}"
        }
    except Exception as e:
        logger.error("予期しない生成エラー: %s", e, exc_info=True)
        return {
            "prompt": request.prompt,
            "generated_text": request.prompt + f" [エラー: {e}]",
            "model_path": request.model_path,
            "error": str(e)
        }


@router.post("/verify-model")
async def verify_finetuned_model(request: GenerationRequest):
    """ファインチューニング済みモデルの検証専用エンドポイント"""
    try:
        logger.info(f"ファインチューニング済みモデル検証開始: {request.model_path}")

        # モデル情報の取得
        model_path = Path(request.model_path)
        if not model_path.exists():
            return {
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

            # テキスト生成リクエストを作成
            gen_request = GenerationRequest(
                model_path=request.model_path,
                prompt=test_prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )

            # テキスト生成を実行
            result = await generate_text(gen_request)

            # 検証結果を記録
            verification_result = {
                "test_case": i + 1,
                "prompt": test_prompt,
                "generated_text": result.get("generated_text", ""),
                "verification_info": result.get("verification_info", {}),
                "success": "error" not in result
            }

            verification_results.append(verification_result)

            # 進捗をログに記録
            logger.info(f"テストケース {i+1} 完了: {'成功' if verification_result['success'] else '失敗'}")

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

        logger.info(f"ファインチューニング済みモデル検証完了: 成功率 {success_count}/{total_count}")

        return {
            "status": "success",
            "verification_summary": verification_summary
        }

    except Exception as e:
        logger.error(f"モデル検証エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "model_path": request.model_path
        }
