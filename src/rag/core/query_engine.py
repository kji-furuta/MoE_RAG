"""
統合クエリエンジン
検索・生成システムを統合したメインエンジン
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logger = logging.getLogger(__name__)

# RAGコンポーネントのインポート
from ..indexing.vector_store import QdrantVectorStore
from ..indexing.embedding_model import EmbeddingModelFactory
from ..indexing.metadata_manager import MetadataManager
from ..retrieval.hybrid_search import HybridSearchEngine, SearchQuery
from ..retrieval.reranker import HybridReranker
from ..core.citation_engine import CitationQueryEngine, GeneratedResponse
from ..config.rag_config import RAGConfig, load_config
from ..utils.exceptions import (
    RAGException,
    SearchError,
    GenerationError,
    VectorStoreError,
    ModelLoadError,
    ValidationError,
    QueryTimeoutError,
    LLMMemoryError,
    VectorStoreConnectionError
)


def _contains_hangul(text: str) -> bool:
    """テキストに韓国語（ハングル）が含まれるかを判定"""
    return any(
        '\uac00' <= ch <= '\ud7af'  # ハングル音節
        or '\u1100' <= ch <= '\u11ff'  # ハングル字母
        or '\u3130' <= ch <= '\u318f'  # ハングル互換字母
        for ch in text
    )


def _dedupe_repeated_paragraphs(text: str) -> str:
    """生成テキスト中の重複段落を除去する

    量子化モデルでは同一段落を延々と繰り返すループが発生することがあるため、
    正規化して同一とみなせる段落の2回目以降を削除する。
    """
    paragraphs = re.split(r'\n{2,}', text)
    seen = set()
    kept = []
    removed = 0
    for para in paragraphs:
        # 空白を除去して正規化したものを比較キーにする
        key = re.sub(r'\s+', '', para)
        if not key:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(para.strip())
    if removed:
        logger.warning(f"重複段落を{removed}件除去しました")
    return '\n\n'.join(kept)


def _ollama_generate_japanese(ollama_client, model_name: str, prompt: str,
                              max_tokens: int = 2048) -> Dict[str, Any]:
    """Ollamaで日本語回答を生成する共通ヘルパー

    パラメータ調整の経緯:
    - repeat_penalty 1.3 + frequency/presence 併用 → 日本語トークンが抑制され
      韓国語・中国語への言語ドリフトが発生（強すぎ）
    - repeat_penalty 1.1 のみ（検出窓64トークン） → 段落単位のループを検出できず
      同一段落の無限繰り返しが発生（弱すぎ）
    - 現在: repeat_penalty 1.15 + repeat_last_n 512 で段落サイズまで検出窓を拡大し、
      さらに後処理で重複段落を除去する
    ハングルが混入した場合は低温度で1回だけ再生成する。
    """
    result = ollama_client.generate_text(
        model_name=model_name,
        prompt=prompt,
        temperature=0.4,
        top_p=0.9,
        max_tokens=max_tokens,
        repetition_penalty=1.15,
        repeat_last_n=512,
    )

    generated = result.get("generated_text", "") if result.get("success", False) else ""

    if generated and _contains_hangul(generated):
        logger.warning("生成結果に韓国語（ハングル）が混入したため、低温度で再生成します")
        retry_prompt = (
            prompt
            + "\n\n【厳守】韓国語（ハングル）・中国語は一切使用せず、日本語のみで回答してください。"
        )
        retry = ollama_client.generate_text(
            model_name=model_name,
            prompt=retry_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            repeat_last_n=512,
        )
        if retry.get("success", False) and not _contains_hangul(retry.get("generated_text", "")):
            result = retry
            generated = retry.get("generated_text", "")
        else:
            logger.warning("再生成でもハングルが除去できなかったため、初回の生成結果を返します")

    # 段落単位のループを後処理で除去
    if generated:
        result["generated_text"] = _dedupe_repeated_paragraphs(generated)

    return result


@dataclass
class QueryResult:
    """クエリ結果"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'query': self.query,
            'answer': self.answer,
            'citations': self.citations,
            'sources': self.sources,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


class LLMGenerator:
    """LLM生成器（継続学習モデル対応版）"""
    
    def __init__(self, config: RAGConfig, load_model: bool = True):
        """
        Args:
            config: RAG設定
            load_model: Whether to load the model immediately
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ollama_fallback = False  # デフォルトではフォールバックを無効
        self.ollama = None
        
        # 継続学習モデルマネージャーの初期化
        self.continual_manager = None
        self.use_continual = False
        self.current_continual_task = None
        
        # 継続学習設定のチェック
        if hasattr(config, 'continual_learning') and config.continual_learning.enabled:
            try:
                from .continual_model_manager import ContinualModelManager
                self.continual_manager = ContinualModelManager(
                    base_path=Path(config.continual_learning.model_base_path)
                )
                self.use_continual = True
                logger.info(f"Continual learning enabled with {len(self.continual_manager.get_available_tasks())} tasks")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"Continual learning module not available: {e}")
                self.use_continual = False
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to initialize continual learning manager: {e}")
                self.use_continual = False
        
        # 動的LoRA適用モードのチェック
        self.use_dynamic_lora = False
        self.dynamic_lora_engine = None
        if hasattr(config.llm, 'use_dynamic_lora') and config.llm.use_dynamic_lora:
            try:
                from .dynamic_lora_llm import DynamicLoRAQueryEngine
                self.dynamic_lora_engine = DynamicLoRAQueryEngine(config.llm.__dict__)
                self.use_dynamic_lora = True
                logger.info("Dynamic LoRA application enabled for GPT-NeoX-20B")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"Dynamic LoRA module not available: {e}")
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to initialize dynamic LoRA: {e}")
                
        # 設定に基づいてOllamaモードを初期化
        if not self.use_dynamic_lora:
            if hasattr(config.llm, 'provider') and config.llm.provider == 'ollama':
                self._enable_ollama_fallback()
            elif hasattr(config.llm, 'use_ollama_fallback') and config.llm.use_ollama_fallback:
                self._enable_ollama_fallback()
        
        # 明示的にファインチューニングモデルが指定された場合のみローカルモデルを試行
        if load_model and hasattr(config.llm, 'use_finetuned') and config.llm.use_finetuned:
            # メモリチェック後にモデルロードを判断
            if self._check_memory_for_model():
                self._load_model()
            else:
                logger.info("メモリ不足のため、Ollamaフォールバックを使用します")
        
    def _check_memory_for_model(self) -> bool:
        """モデルロードのための十分なメモリがあるかチェック"""
        if not torch.cuda.is_available():
            return False

        # 全GPUのメモリをチェック
        gpu_count = torch.cuda.device_count()
        max_free_memory = 0
        total_free_memory = 0

        for i in range(gpu_count):
            free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
            max_free_memory = max(max_free_memory, free_memory)
            total_free_memory += free_memory

        # モデルサイズに基づいてメモリ要件を判定
        model_name = ''
        if hasattr(self.config.llm, 'model_name'):
            model_name = str(self.config.llm.model_name).lower()

        # モデルサイズを推定（config.jsonから判定も試みる）
        required_memory = 8  # デフォルト: 7B/8Bモデル用（4bit量子化前提）
        if any(s in model_name for s in ['32b', '22b']):
            required_memory = 20
        elif any(s in model_name for s in ['13b', '14b']):
            required_memory = 12
        else:
            # config.jsonからモデルサイズを推定
            model_dir = Path(model_name) if model_name else None
            if model_dir and model_dir.exists():
                config_file = model_dir / 'config.json'
                if config_file.exists():
                    try:
                        import json
                        with open(config_file, 'r') as f:
                            mc = json.load(f)
                        num_layers = mc.get('num_hidden_layers', 0)
                        hidden_size = mc.get('hidden_size', 0)
                        # 大まかなサイズ推定
                        if num_layers >= 60 or hidden_size >= 6144:
                            required_memory = 20  # 22B+
                        elif num_layers >= 40 or hidden_size >= 5120:
                            required_memory = 12  # 13B+
                        else:
                            required_memory = 8   # 7B以下
                    except Exception:
                        pass

        # device_map='auto'により複数GPUを使用可能なため、合計メモリで判定
        return total_free_memory >= required_memory
    
    def _load_model(self):
        """モデルを読み込み（メモリ最適化）"""
        
        # GPUメモリチェックとOllamaフォールバック
        if torch.cuda.is_available():
            # 全GPUのメモリをチェック
            gpu_count = torch.cuda.device_count()
            total_free_memory = 0
            max_free_memory = 0
            best_gpu = 0
            
            for i in range(gpu_count):
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
                total_free_memory += free_mem
                if free_mem > max_free_memory:
                    max_free_memory = free_mem
                    best_gpu = i
                logger.info(f"GPU {i}: 空きメモリ {free_mem:.2f} GB")
            
            logger.info(f"合計GPU空きメモリ: {total_free_memory:.2f} GB (最大単一GPU: {max_free_memory:.2f} GB on GPU {best_gpu})")
            
            # 32Bモデルには最低20GB必要（単一GPUで）
            required_memory = 20  # GB
            if max_free_memory < required_memory:
                logger.warning(f"GPUメモリ不足: 最大単一GPU {max_free_memory:.2f}GB / 必要 {required_memory}GB以上")
                logger.warning("ファインチューニング済みモデル（32B）を読み込むにはメモリが不足しています。")
                logger.warning("Ollamaフォールバックを有効化します。")
                self._enable_ollama_fallback()
                return
            
            # 最適なGPUを設定
            torch.cuda.set_device(best_gpu)
            self.device = torch.device(f'cuda:{best_gpu}')
            logger.info(f"GPU {best_gpu} を使用してモデルをロードします")
        
        llm_config = self.config.llm
        
        # モデルパスの決定（設定ファイルの選択を優先）
        model_path = None
        
        # 1. 設定ファイルのmodel_nameを優先使用
        if hasattr(llm_config, 'model_name') and llm_config.model_name:
            model_path = llm_config.model_name
            logger.info(f"Using configured model: {model_path}")
        
        # 2. model_pathが設定されている場合
        elif hasattr(llm_config, 'model_path') and llm_config.model_path:
            model_path = llm_config.model_path
            logger.info(f"Using configured model path: {model_path}")
        
        # 3. ファインチューニング済みモデルを確認
        elif llm_config.use_finetuned and hasattr(llm_config, 'finetuned_model_path') and os.path.exists(llm_config.finetuned_model_path):
            model_path = llm_config.finetuned_model_path
            logger.info(f"Using fine-tuned model: {model_path}")
        
        # 4. フォールバック: ベースモデル
        else:
            model_path = llm_config.base_model
            logger.info(f"Using base model: {model_path}")
        
        # パスが相対パスの場合は絶対パスに変換
        if not model_path.startswith('/') and '/' in model_path and not model_path.startswith('http'):
            # プロジェクトルートからの相対パスとみなす
            project_root = Path(__file__).parent.parent.parent.parent
            absolute_path = project_root / model_path
            if absolute_path.exists():
                model_path = str(absolute_path)
                logger.info(f"Resolved to absolute path: {model_path}")
            else:
                logger.warning(f"Model path does not exist: {absolute_path}")
        
        logger.info(f"Final model path: {model_path}")
            
        try:
            # メモリクリア
            torch.cuda.empty_cache()
            
            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # メモリ最適化されたモデル読み込み
            model_kwargs = self._get_optimized_model_kwargs(llm_config)
            
            # GPUメモリ不足対策
            if torch.cuda.is_available():
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as gpu_error:
                    if isinstance(gpu_error, torch.cuda.OutOfMemoryError) or "CUDA" in str(gpu_error) or "GPU" in str(gpu_error):
                        logger.warning(f"GPU読み込み失敗: {gpu_error}")
                        logger.info("CPUモードで再試行します")
                        torch.cuda.empty_cache()
                        # CPUモードで再試行
                        model_kwargs['device_map'] = None
                        model_kwargs['torch_dtype'] = torch.float32
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                    else:
                        raise
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM loading model: {e}")
            torch.cuda.empty_cache()
            logger.warning("GPUメモリ不足。Ollamaフォールバックを有効化します")
            self._enable_ollama_fallback()
        except (OSError, ValueError) as e:
            logger.error(f"Failed to load model (IO/config): {e}")
            logger.warning("モデルロード失敗。Ollamaフォールバックを有効化します")
            self._enable_ollama_fallback()
        except Exception as e:
            logger.error(f"Unexpected model load error: {e}", exc_info=True)
            logger.warning("モデルロード失敗。Ollamaフォールバックを有効化します")
            self._enable_ollama_fallback()
    
    def _get_optimized_model_kwargs(self, llm_config) -> Dict[str, Any]:
        """メモリ最適化されたモデルロードパラメータを取得"""
        
        # 基本設定
        model_kwargs = {
            'torch_dtype': torch.float16,  # メモリ効率を優先
            'low_cpu_mem_usage': True,
            'trust_remote_code': True
        }
        
        if torch.cuda.is_available():
            # 現在のデバイスのメモリ情報を取得（すでに最適なGPUが選択されている）
            current_device = torch.cuda.current_device()
            free_memory = torch.cuda.mem_get_info(current_device)[0] / (1024**3)
            logger.info(f"GPU {current_device} 空きメモリ: {free_memory:.2f} GB")
            
            if free_memory < 8:  # 8GB未満の場合
                # 4bit量子化を適用
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_enable_fp32_cpu_offload=True  # CPUオフロードを有効化
                    )
                    logger.info("4bit量子化を使用します")
                except ImportError:
                    logger.warning("BitsAndBytesが利用できません。通常のfp16を使用します")
            
            # メモリ配分を最適化
            safe_memory = max(1, int(free_memory * 0.7))  # 70%を使用
            
            # offload_dirを設定（Qwen2ForCausalLM以外の場合のみ）
            import tempfile
            import os
            offload_dir = tempfile.mkdtemp(prefix="model_offload_")
            logger.info(f"オフロードディレクトリ: {offload_dir}")
            
            # モデルタイプに応じてoffload_dirを条件付きで追加
            # 複数GPUに対応したメモリ配分
            max_memory_dict = {}
            for i in range(torch.cuda.device_count()):
                gpu_free = torch.cuda.mem_get_info(i)[0] / (1024**3)
                gpu_safe = max(1, int(gpu_free * 0.7))  # 各GPUの70%を使用
                max_memory_dict[i] = f"{gpu_safe}GB"
            max_memory_dict['cpu'] = '32GB'  # CPUメモリ
            
            model_kwargs.update({
                'device_map': 'auto',
                'max_memory': max_memory_dict
            })
            
            # Qwen2ForCausalLM以外のモデルの場合のみoffload_dirを追加
            try:
                is_qwen = False
                # モデル名をチェックしてQwen2ForCausalLMかどうかを判定
                if hasattr(llm_config, 'model_name'):
                    model_name = str(llm_config.model_name).lower()
                elif hasattr(llm_config, 'base_model'):
                    model_name = str(llm_config.base_model).lower()
                else:
                    model_name = ''

                if 'qwen' in model_name:
                    is_qwen = True
                else:
                    # モデル名にqwenが含まれない場合、config.jsonで判定
                    model_dir = Path(model_name) if model_name else None
                    if model_dir and model_dir.exists():
                        config_file = model_dir / 'config.json'
                        if config_file.exists():
                            import json
                            with open(config_file, 'r') as f:
                                model_config = json.load(f)
                            arch = model_config.get('architectures', [])
                            model_type = model_config.get('model_type', '')
                            if any('qwen' in a.lower() for a in arch) or 'qwen' in model_type.lower():
                                is_qwen = True
                                logger.info(f"config.jsonからQwenモデルを検出: {arch}")

                if not is_qwen:
                    model_kwargs.update({
                        'offload_folder': offload_dir,  # オフロードディレクトリを追加
                        'offload_state_dict': True   # 状態辞書のオフロードを有効化
                    })
                    logger.info("offload_folderを有効化しました")
                else:
                    logger.info("Qwen2ForCausalLMのため、offload_folderを無効化しました")
            except (AttributeError, TypeError) as e:
                logger.warning(f"モデルタイプ判定エラー: {e}。offload_folderを無効化します")
        else:
            # CPUモード
            logger.info("CPUモードで実行します")
            model_kwargs.update({
                'torch_dtype': torch.float32,
                'device_map': None
            })
            
        return model_kwargs
    
    def _enable_ollama_fallback(self):
        """メモリ不足時のOllamaフォールバックを有効化"""
        
        self.use_ollama_fallback = True
        self.model = None
        self.tokenizer = None
        logger.info("Ollamaフォールバックモードを有効化しました")
        
        # Ollama統合をインポート
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "convert"))
            from ollama_integration import OllamaIntegration
            self.ollama = OllamaIntegration()
            logger.info("Ollama統合が利用可能です")
        except ImportError as e:
            logger.error(f"Ollama統合のインポートに失敗: {e}")
            self.ollama = None
            
    def generate(self, 
                prompt: str, 
                context: str,
                max_new_tokens: Optional[int] = None,
                query_text: Optional[str] = None) -> str:
        """テキストを生成（継続学習モデル対応）"""
        
        # 継続学習モデルの選択チェック
        if self.use_continual and self.continual_manager and query_text:
            should_use, task_name = self.continual_manager.should_use_continual_model(query_text)
            if should_use and task_name:
                logger.info(f"Using continual learning model for task: {task_name}")
                model, tokenizer = self.continual_manager.load_model_for_task(task_name, str(self.device))
                if model and tokenizer:
                    # 継続学習モデルを一時的に使用
                    original_model = self.model
                    original_tokenizer = self.tokenizer
                    self.model = model
                    self.tokenizer = tokenizer
                    self.current_continual_task = task_name
                    try:
                        # 継続学習モデルで生成
                        result = self._generate_with_model(prompt, context, max_new_tokens)
                        # 元のモデルに戻す（メモリ節約のため）
                        self.model = original_model
                        self.tokenizer = original_tokenizer
                        return result
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"GPU OOM with continual model: {e}")
                        torch.cuda.empty_cache()
                        self.model = original_model
                        self.tokenizer = original_tokenizer
                    except (RuntimeError, ValueError) as e:
                        logger.error(f"Failed to generate with continual model: {e}")
                        self.model = original_model
                        self.tokenizer = original_tokenizer
        
        # 通常のモデル選択ロジック
        # Ollamaプロバイダーが設定されている場合は直接Ollamaを使用
        if hasattr(self.config.llm, 'provider') and self.config.llm.provider == 'ollama':
            if not self.use_ollama_fallback:
                self._enable_ollama_fallback()
            return self._ollama_generation(prompt, context)
        
        # モデルが未ロードの場合、オンデマンドでロード
        if not self.model and not self.use_ollama_fallback:
            logger.info("Model not loaded, attempting on-demand loading...")
            try:
                self._load_model()
            except (torch.cuda.OutOfMemoryError, OSError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to load model on-demand: {e}")
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                self._enable_ollama_fallback()
        
        if not self.model or not self.tokenizer or self.use_ollama_fallback:
            return self._ollama_generation(prompt, context)
            
        llm_config = self.config.llm
        max_tokens = max_new_tokens or llm_config.max_new_tokens
        
        # プロンプトを構築
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            # トークナイズ
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - max_tokens,
                padding=True
            )
            
            # モデルがロードされているデバイスに送る
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to(self.device)
            
            # 生成実行（タイムアウト設定を追加）
            logger.info(f"Starting generation with max_tokens={max_tokens}")
            
            # 生成パラメータを調整（大規模モデル用の最適化）
            generation_kwargs = {
                'max_new_tokens': min(max_tokens, 512),  # 最大512トークンに制限
                'temperature': llm_config.temperature,
                'top_p': llm_config.top_p,
                'repetition_penalty': llm_config.repetition_penalty,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True  # 早期停止を有効化
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            logger.info(f"Generation completed, output shape: {outputs.shape}")
                
            # デコード
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            logger.info(f"Generated text length: {len(generated_text)}")
            
            return generated_text.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during generation: {e}")
            torch.cuda.empty_cache()
            return self._ollama_generation(prompt, context)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Generation failed: {e}")
            return self._ollama_generation(prompt, context)

    def _generate_with_model(self, prompt: str, context: str, max_new_tokens: Optional[int] = None) -> str:
        """モデルを使用してテキストを生成（継続学習・通常モデル共通）"""
        llm_config = self.config.llm
        max_tokens = max_new_tokens or llm_config.max_new_tokens
        
        # プロンプトを構築
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            # トークナイズ
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - max_tokens,
                padding=True
            )
            
            # モデルがロードされているデバイスに送る
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to(self.device)
            
            # 生成実行
            logger.info(f"Generating with {'continual' if self.current_continual_task else 'standard'} model, max_tokens={max_tokens}")
            
            # 生成パラメータを調整
            generation_kwargs = {
                'max_new_tokens': min(max_tokens, 512),
                'temperature': llm_config.temperature,
                'top_p': llm_config.top_p,
                'do_sample': llm_config.temperature > 0,
                'repetition_penalty': 1.15,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # デコード
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            if self.current_continual_task:
                logger.info(f"Generated {len(generated_text)} chars using continual model: {self.current_continual_task}")
            
            return generated_text.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM in _generate_with_model: {e}")
            torch.cuda.empty_cache()
            return self._fallback_generation(prompt, context)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Generation error: {e}")
            return self._fallback_generation(prompt, context)

    def _ollama_generation(self, prompt: str, context: str) -> str:
        """メモリ不足時のOllamaフォールバック生成"""
        
        # メモリ不足の警告メッセージを追加
        memory_warning = ""
        if torch.cuda.is_available():
            # 全GPUのメモリをチェック
            gpu_count = torch.cuda.device_count()
            max_free_memory = 0
            total_free_memory = 0
            
            for i in range(gpu_count):
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
                total_free_memory += free_mem
                max_free_memory = max(max_free_memory, free_mem)
            
            if max_free_memory < 20:  # 32Bモデルには最低20GB必要
                memory_warning = (
                    f"\n\n【システム通知】GPUメモリ不足のため、ファインチューニング済みモデルを読み込めません。\n"
                    f"最大単一GPU空きメモリ: {max_free_memory:.2f}GB / 必要メモリ: 約20GB以上\n"
                    f"合計GPU空きメモリ: {total_free_memory:.2f}GB (GPU数: {gpu_count})\n"
                    f"代替モデル（Ollama）で回答を生成しています。\n"
                )
        
        if not self.ollama:
            return f"エラー: ファインチューニング済みモデルが利用できません。{memory_warning}\nクエリ: {prompt}"
        
        try:
            # コンテキストとプロンプトを組み合わせ
            full_prompt = self._build_prompt(prompt, context)
            
            # 設定からOllamaモデル名を取得（優先順位で試行）
            ollama_model = 'llama3.2:3b'  # デフォルトを一般的なモデルに変更
            
            # 複数の設定箇所から取得を試みる
            if hasattr(self.config.llm, 'ollama') and hasattr(self.config.llm.ollama, 'model'):
                ollama_model = self.config.llm.ollama.model
            elif hasattr(self.config.llm, 'ollama_model'):
                ollama_model = self.config.llm.ollama_model
            elif hasattr(self.config.llm, 'model_name') and self.config.llm.model_name.startswith('ollama:'):
                ollama_model = self.config.llm.model_name[7:]
            
            logger.info(f"Ollamaモデル使用: {ollama_model}")
            
            # Ollamaで生成（日本語固定・ハングル混入時はリトライ）
            result = _ollama_generate_japanese(self.ollama, ollama_model, full_prompt)
            
            if result.get("success", False):
                generated_text = result.get("generated_text", "")
                # 中国語簡体字を日本語に変換（ポストプロセッシング）
                generated_text = self._convert_chinese_to_japanese(generated_text)
                logger.info("Ollamaでの生成が成功しました")
                return memory_warning + generated_text if memory_warning else generated_text
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Ollama生成エラー: {error_msg}")
                return f"エラー: Ollama生成に失敗しました - {error_msg}{memory_warning}"
                
        except ConnectionError as e:
            logger.error(f"Ollama接続エラー: {e}")
            return f"エラー: Ollamaサーバーに接続できません - {str(e)}{memory_warning}"
        except (TimeoutError, OSError) as e:
            logger.error(f"Ollamaタイムアウト/IOエラー: {e}")
            return f"エラー: Ollama生成がタイムアウトしました - {str(e)}{memory_warning}"
        except Exception as e:
            logger.error(f"Ollamaフォールバックエラー: {e}", exc_info=True)
            return f"エラー: 生成に失敗しました - {str(e)}{memory_warning}"

    def _build_prompt(self, query: str, context: str) -> str:
        """プロンプトを構築"""

        prompt_template = """あなたは道路設計の専門家です。以下の参考資料に基づいて、質問に正確に回答してください。

**重要: 回答は必ず日本語のみを使用してください。中国語の簡体字や繁体字、韓国語（ハングル）は使用しないでください。**

重要な指示:
1. 数値や基準値は必ず参考資料から正確に引用すること
2. 該当する条文番号や表番号を明記すること
3. 複数の基準がある場合は、すべて列挙すること
4. 不明な場合は推測せず「参考資料に該当する情報が見つかりません」と回答すること
5. 回答は簡潔で実践的にすること
6. **日本語のみを使用**し、中国語の簡体字（例: 车、学、国）や韓国語（ハングル）は使わないでください

参考資料:
{context}

質問: {query}

回答:"""

        return prompt_template.format(context=context, query=query)
        
    def _fallback_generation(self, query: str, context: str) -> str:
        """フォールバック生成（モデルが利用できない場合）"""
        
        logger.warning("Using fallback generation")
        
        # 簡易的な回答生成
        if context:
            lines = context.split('\n')
            relevant_lines = [line for line in lines if line.strip() and not line.startswith('[')]
            
            if relevant_lines:
                return f"参考資料によると：\n\n{relevant_lines[0][:300]}..."
                
        return "申し訳ございませんが、現在回答を生成できません。参考資料をご確認ください。"


class RoadDesignQueryEngine:
    """道路設計特化型クエリエンジン"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 metadata_db_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス
            vector_store_path: ベクトルストアのパス
            metadata_db_path: メタデータDBのパス
        """
        
        # 設定を読み込み
        self.config = load_config(config_path)
        
        # パスの設定
        self.vector_store_path = vector_store_path or self.config.vector_store.path
        self.metadata_db_path = metadata_db_path or "./metadata/metadata.db"
        
        # コンポーネントの初期化
        self.embedding_model = None
        self.vector_store = None
        self.hybrid_search = None
        self.reranker = None
        self.llm_generator = None
        self.citation_engine = None
        self.metadata_manager = None
        
        # MoE統合コンポーネント
        self.moe_rag_system = None
        self.use_moe = False
        
        self.is_initialized = False

    def _map_model_name_to_type(self, model_name: str) -> str:
        """
        設定ファイルのモデル名をEmbeddingModelFactoryのmodel_typeに変換

        Args:
            model_name: 設定ファイルのモデル名（フルパスまたは短縮名）

        Returns:
            EmbeddingModelFactoryで使用できるmodel_type（短縮名）
        """
        # モデル名のマッピング辞書
        model_mapping = {
            "intfloat/multilingual-e5-large": "multilingual-e5-large",
            "intfloat/multilingual-e5-large-instruct": "multilingual-e5-large-instruct",
            "intfloat/multilingual-e5-base": "multilingual-e5-base",
            "intfloat/multilingual-e5-small": "multilingual-e5-small",
            "sentence-transformers/multilingual-e5-large": "multilingual-e5-large",
            "sonoisa/sentence-bert-base-ja-mean-tokens-v2": "sentence-bert-ja",
            "cl-nagoya/sup-simcse-ja-large": "sup-simcse-ja",
        }

        # マッピングに存在する場合は変換
        if model_name in model_mapping:
            return model_mapping[model_name]

        # すでに短縮名の場合はそのまま返す
        if model_name in ["multilingual-e5-large", "multilingual-e5-large-instruct", "multilingual-e5-base",
                          "multilingual-e5-small", "sentence-bert-ja", "sup-simcse-ja"]:
            return model_name

        # 未知のモデル名の場合は警告してデフォルトを返す
        logger.warning(f"Unknown model name: {model_name}, using default: multilingual-e5-large")
        return "multilingual-e5-large"

    def initialize(self):
        """エンジンを初期化（メモリ最適化）"""
        
        logger.info("Initializing RoadDesignQueryEngine...")
        
        # GPUメモリチェック
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"GPU空きメモリ: {free_memory:.2f} GB")
            
            # メモリ不足の場合は軽量モードで初期化
            if free_memory < 6:
                logger.warning("メモリ不足のため、軽量モードで初期化します")
                self._initialize_lightweight_mode()
                return
        
        try:
            # 1. 埋め込みモデル
            logger.info("Loading embedding model...")
            embedding_config = self.config.embedding

            # 設定ファイルのモデル名を短縮名に変換
            model_type = self._map_model_name_to_type(embedding_config.model_name)
            logger.info(f"Using embedding model: {model_type} (from config: {embedding_config.model_name})")

            self.embedding_model = EmbeddingModelFactory.create(
                model_type=model_type,
                device=embedding_config.device
            )

            # 2. ベクトルストア
            logger.info("Loading vector store...")
            embedding_dim = EmbeddingModelFactory.get_embedding_dim(model_type)
            
            # URLが設定されている場合はサーバーモードを使用
            if hasattr(self.config.vector_store, 'url') and self.config.vector_store.url:
                self.vector_store = QdrantVectorStore(
                    collection_name=self.config.vector_store.collection_name,
                    embedding_dim=embedding_dim,
                    url=self.config.vector_store.url,
                    prefer_grpc=self.config.vector_store.prefer_grpc
                )
                logger.info(f"Using Qdrant server at {self.config.vector_store.url}")
            else:
                self.vector_store = QdrantVectorStore(
                    collection_name=self.config.vector_store.collection_name,
                    embedding_dim=embedding_dim,
                    path=self.vector_store_path
                )
                logger.info(f"Using local Qdrant at {self.vector_store_path}")
            
            # 3. メタデータマネージャー
            logger.info("Loading metadata manager...")
            self.metadata_manager = MetadataManager(db_path=self.metadata_db_path)
            
            # 4. ハイブリッド検索エンジン
            logger.info("Initializing hybrid search...")
            retrieval_config = self.config.retrieval
            # Keywordエンジンの詳細設定（存在しない場合はデフォルト）
            ke = getattr(retrieval_config, 'keyword_engine', None)
            ke_backend = getattr(ke, 'backend', 'tfidf') if ke else 'tfidf'
            ke_max_features = getattr(ke, 'max_features', 30000) if ke else 30000
            ke_ngram = (
                getattr(ke, 'ngram_min', 2) if ke else 2,
                getattr(ke, 'ngram_max', 4) if ke else 4,
            )
            ke_min_df = getattr(ke, 'min_df', 2) if ke else 2
            ke_rebuild_threshold = getattr(ke, 'rebuild_threshold', 200) if ke else 200

            self.hybrid_search = HybridSearchEngine(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                vector_weight=retrieval_config.vector_weight,
                keyword_weight=retrieval_config.keyword_weight,
                keyword_backend=ke_backend,
                keyword_max_features=ke_max_features,
                keyword_ngram_range=ke_ngram,
                keyword_min_df=ke_min_df,
                keyword_rebuild_threshold=ke_rebuild_threshold
            )
            
            # コーパス情報が必要な場合は別途初期化
            self._initialize_search_corpus()
            
            # 5. リランカー
            if retrieval_config.reranking_enabled:
                logger.info("Initializing reranker...")
                self.reranker = HybridReranker()
            
            # 6. LLM生成器
            logger.info("Loading LLM generator...")
            self.llm_generator = LLMGenerator(self.config, load_model=False)
            
            # 7. 引用エンジン
            logger.info("Initializing citation engine...")
            self.citation_engine = CitationQueryEngine(
                hybrid_search_engine=self.hybrid_search,
                reranker=self.reranker,
                llm_generator=self.llm_generator
            )
            
            # 8. MoE統合チェック
            if hasattr(self.config.llm, 'use_moe') and self.config.llm.use_moe:
                self._initialize_moe_integration()
            
            self.is_initialized = True
            logger.info("RoadDesignQueryEngine initialization completed")
            
        except VectorStoreConnectionError as e:
            logger.error("ベクトルストア接続失敗: %s", e, exc_info=True)
            logger.warning("標準初期化失敗。軽量モードでリトライします")
            self._initialize_lightweight_mode()
        except (ModelLoadError, LLMMemoryError) as e:
            logger.error("モデルロード失敗: %s", e, exc_info=True)
            logger.warning("モデルロード失敗。軽量モードでリトライします")
            self._initialize_lightweight_mode()
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during initialization: %s", e)
            torch.cuda.empty_cache()
            logger.warning("GPUメモリ不足。軽量モードでリトライします")
            self._initialize_lightweight_mode()
        except (ImportError, ModuleNotFoundError) as e:
            logger.error("依存モジュールが見つかりません: %s", e, exc_info=True)
            self._initialize_lightweight_mode()
        except Exception as e:
            logger.error("予期しない初期化エラー: %s", e, exc_info=True)
            logger.warning("標準初期化失敗。軽量モードでリトライします")
            self._initialize_lightweight_mode()

    def _initialize_lightweight_mode(self):
        """メモリ不足時の軽量モード初期化"""
        
        logger.info("軽量モードで初期化中...")
        
        try:
            # 1. 基本コンポーネントのみ初期化
            logger.info("Loading lightweight embedding model...")
            # GPUメモリ状況を安全に判定
            device_choice = "cpu"
            try:
                if torch.cuda.is_available():
                    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                    device_choice = "cuda" if free_gb >= 4 else "cpu"
            except Exception:
                device_choice = "cpu"

            # 軽量埋め込みモデルをロード
            self.embedding_model = EmbeddingModelFactory.create(
                model_type="multilingual-e5-small",  # 軽量モデルに寄せる
                device=device_choice
            )
            
            # 2. ベクターストア（基本機能のみ）
            embedding_dim = 384  # MiniLMの次元数
            self.vector_store = QdrantVectorStore(
                collection_name=self.config.vector_store.collection_name,
                embedding_dim=embedding_dim,
                path=self.vector_store_path
            )
            
            # 3. メタデータマネージャー
            self.metadata_manager = MetadataManager(db_path=self.metadata_db_path)
            
            # 4. ハイブリッド検索エンジン（軽量モード）
            from ..retrieval.hybrid_search import HybridSearchEngine
            retrieval_config = self.config.retrieval
            ke = getattr(retrieval_config, 'keyword_engine', None)
            ke_backend = getattr(ke, 'backend', 'tfidf') if ke else 'tfidf'
            ke_max_features = getattr(ke, 'max_features', 30000) if ke else 30000
            ke_ngram = (
                getattr(ke, 'ngram_min', 2) if ke else 2,
                getattr(ke, 'ngram_max', 4) if ke else 4,
            )
            ke_min_df = getattr(ke, 'min_df', 2) if ke else 2
            ke_rebuild_threshold = getattr(ke, 'rebuild_threshold', 200) if ke else 200

            self.hybrid_search = HybridSearchEngine(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                vector_weight=retrieval_config.vector_weight if retrieval_config else 0.7,
                keyword_weight=retrieval_config.keyword_weight if retrieval_config else 0.3,
                keyword_backend=ke_backend,
                keyword_max_features=ke_max_features,
                keyword_ngram_range=ke_ngram,
                keyword_min_df=ke_min_df,
                keyword_rebuild_threshold=ke_rebuild_threshold
            )
            
            # コーパスを初期化
            self._initialize_search_corpus()
            
            # 5. OllamaベースのLLM生成器
            self.llm_generator = LLMGenerator(self.config, load_model=False)
            # Ollama統合を有効化（フォールバックではなくメインモードとして）
            if hasattr(self.config.llm, 'provider') and self.config.llm.provider == 'ollama':
                self.llm_generator._enable_ollama_fallback()  # Ollamaをメインモードとして使用
            
            # 6. シンプルな引用エンジン
            self.citation_engine = CitationQueryEngine(
                hybrid_search_engine=self.hybrid_search,
                llm_generator=self.llm_generator,
                metadata_manager=self.metadata_manager
            )
            
            self.is_initialized = True
            logger.info("軽量モードでの初期化が完了しました")
            
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"軽量モード依存モジュール不足: {e}")
            self._initialize_minimal_mode()
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"軽量モード初期化も失敗: {e}")
            self._initialize_minimal_mode()
    
    def _initialize_minimal_mode(self):
        """最低限の機能で初期化（Ollamaのみ）"""
        
        logger.warning("最低限モードで初期化中...")
        
        try:
            # Ollamaのみで動作するシンプルなモード
            self.llm_generator = LLMGenerator(self.config, load_model=False)
            # 設定に基づいてOllamaを有効化
            if hasattr(self.config.llm, 'provider') and self.config.llm.provider == 'ollama':
                self.llm_generator._enable_ollama_fallback()
            
            # ダミーのメタデータマネージャー
            self.metadata_manager = MetadataManager(db_path=":memory:")  # インメモリデータベース
            
            # ダミーのハイブリッド検索（基本的な機能のみ）
            self.hybrid_search = None  # 最低限モードでは無効
            
            self.is_initialized = True
            logger.info("最低限モードでの初期化が完了しました")
            
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"最低限モード初期化も失敗: {e}")
            raise RuntimeError("すべての初期化が失敗しました") from e
            
    def _initialize_search_corpus(self):
        """検索用コーパスを初期化"""
        
        try:
            # ベクトルストアから実際の文書内容を取得
            logger.info("Fetching documents from vector store for keyword search initialization...")
            
            # Qdrantから全文書を取得（スクロール検索）
            corpus_texts = []
            corpus_ids = []
            
            try:
                # ベクトルストアのコレクション情報を取得
                collection_info = self.vector_store.get_collection_info()
                total_vectors = collection_info.get('vectors_count', 0)
                total_points = collection_info.get('points_count', 0)
                
                # vectors_countが0でもpoints_countがあれば処理を続行
                if total_vectors > 0 or total_points > 0:
                    actual_count = total_vectors if total_vectors > 0 else total_points
                    logger.info(f"Found {actual_count} documents in collection (vectors: {total_vectors}, points: {total_points})")
                    
                    # スクロール検索で全文書を取得（最大1000件）
                    limit = min(1000, actual_count)
                    offset = None
                    batch_size = 100
                    
                    while len(corpus_texts) < limit:
                        # Qdrantのスクロール検索
                        try:
                            scroll_result = self.vector_store.client.scroll(
                                collection_name=self.vector_store.collection_name,
                                limit=batch_size,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False
                            )
                            
                            if not scroll_result or not scroll_result[0]:
                                break
                                
                            points, next_offset = scroll_result
                            
                            for point in points:
                                if point.payload and 'text' in point.payload:
                                    corpus_texts.append(point.payload['text'])
                                    # original_idまたはdoc_idを使用
                                    doc_id = point.payload.get('original_id', point.payload.get('doc_id', str(point.id)))
                                    corpus_ids.append(doc_id)
                            
                            if next_offset is None or len(corpus_texts) >= limit:
                                break
                            offset = next_offset
                        except Exception as scroll_error:
                            logger.warning(f"Scroll search error: {scroll_error}")
                            # 代替手段：検索を使用
                            try:
                                dummy_embedding = np.zeros(self.embedding_model.embedding_dim)
                                search_results = self.vector_store.search(
                                    query_embedding=dummy_embedding,
                                    top_k=min(100, limit),
                                    score_threshold=0.0
                                )
                                for result in search_results:
                                    corpus_texts.append(result.text)
                                    corpus_ids.append(result.metadata.get('original_id', result.metadata.get('doc_id', result.id)))
                                logger.info(f"Fallback search retrieved {len(search_results)} documents")
                            except Exception as search_error:
                                logger.error(f"Fallback search also failed: {search_error}")
                            break
                    
                    logger.info(f"Retrieved {len(corpus_texts)} documents for keyword search")
                else:
                    logger.warning("No documents count from collection info, trying direct scroll...")
                    # コレクション情報が取得できなくても、直接スクロール検索を試みる
                    try:
                        offset = None
                        batch_size = 100
                        limit = 1000
                        
                        while len(corpus_texts) < limit:
                            scroll_result = self.vector_store.client.scroll(
                                collection_name=self.vector_store.collection_name,
                                limit=batch_size,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False
                            )
                            
                            if not scroll_result or not scroll_result[0]:
                                break
                                
                            points, next_offset = scroll_result
                            
                            for point in points:
                                if point.payload and 'text' in point.payload:
                                    corpus_texts.append(point.payload['text'])
                                    doc_id = point.payload.get('original_id', point.payload.get('doc_id', str(point.id)))
                                    corpus_ids.append(doc_id)
                            
                            if next_offset is None or len(corpus_texts) >= limit:
                                break
                            offset = next_offset
                            
                        if corpus_texts:
                            logger.info(f"Successfully retrieved {len(corpus_texts)} documents via direct scroll")
                    except Exception as direct_scroll_error:
                        logger.warning(f"Direct scroll also failed: {direct_scroll_error}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch from vector store: {e}")
                # フォールバック: メタデータから取得
                documents = self.metadata_manager.search_documents()
                if documents:
                    corpus_texts = [f"Document: {doc.title}" for doc in documents[:100]]
                    corpus_ids = [doc.id for doc in documents[:100]]
            
            # キーワード検索エンジンを初期化
            if corpus_texts:
                self.hybrid_search.initialize(corpus_texts, corpus_ids)
                logger.info(f"Initialized keyword search with {len(corpus_texts)} documents")
            else:
                logger.warning("No documents found, initializing with empty corpus")
                # 空のコーパスで初期化（ベクトル検索のみ有効）
                self.hybrid_search.initialize([], [])
                
        except Exception as e:
            logger.warning(f"Failed to initialize search corpus: {e}")
            # エラー時も空のコーパスで初期化
            try:
                self.hybrid_search.initialize([], [])
                logger.info("Fallback: Initialized with empty corpus")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize empty corpus: {fallback_e}")
            
    def _initialize_moe_integration(self):
        """MoE統合を初期化"""
        try:
            logger.info("Initializing MoE-RAG integration...")
            
            # UnifiedMoERAGSystemをインポート
            from ...moe_rag_integration.unified_moe_rag_system import UnifiedMoERAGSystem
            from ...moe.moe_architecture import MoEConfig
            
            # MoE設定を作成
            moe_config = MoEConfig(
                num_experts=self.config.llm.moe_num_experts,
                num_experts_per_tok=self.config.llm.moe_experts_per_token,
                hidden_size=768,
                domain_specific_routing=True
            )
            
            # 統合システムを初期化
            self.moe_rag_system = UnifiedMoERAGSystem(
                rag_config_path=None,  # 既存の設定を使用
                moe_config=moe_config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 既存のRAGコンポーネントを設定
            if self.vector_store:
                self.moe_rag_system.vector_store = self.vector_store
            if self.hybrid_search:
                self.moe_rag_system.hybrid_searcher = self.hybrid_search
            
            self.use_moe = True
            logger.info("MoE-RAG integration initialized successfully")
            
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"MoE module not available: {e}")
            self.use_moe = False
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to initialize MoE integration: {e}")
            self.use_moe = False
    
    def query(self, 
             query_text: str,
             top_k: int = 5,
             search_type: str = "hybrid",
             filters: Optional[Dict[str, Any]] = None,
             include_sources: bool = True) -> QueryResult:
        """クエリを実行"""
        
        if not self.is_initialized:
            raise RuntimeError("QueryEngine must be initialized before use")
            
        import time
        start_time = time.time()
        
        logger.info(f"Processing query: {query_text}")
        
        # MoEモードが有効な場合
        if self.use_moe and self.moe_rag_system:
            return self._query_with_moe(query_text, top_k, search_type, filters, include_sources)
        
        try:
            # 引用エンジンがない場合のみシンプルフォールバック
            if not self.citation_engine and not self.hybrid_search:
                processing_time = time.time() - start_time
                return self._simple_ollama_query(query_text, top_k, processing_time)
            
            # 検索クエリを構築
            search_query = SearchQuery(
                text=query_text,
                search_type=search_type,
                filters=filters
            )
            
            # Ollamaフォールバック時のハイブリッド検索対応
            if self.llm_generator and self.llm_generator.use_ollama_fallback:
                response = self._hybrid_search_with_ollama(
                    query_text=query_text,
                    top_k=top_k,
                    search_type=search_type,
                    filters=filters
                )
            else:
                # 標準の引用エンジンでクエリを実行
                response = self.citation_engine.query(
                    query_text=query_text,
                    top_k=top_k,
                    include_sources=include_sources,
                    filters=filters
                )
            
            processing_time = time.time() - start_time
            
            if not response or not hasattr(response, 'source_chunks'):
                # レスポンスが無い場合はOllamaフォールバック
                return self._simple_ollama_query(query_text, top_k, processing_time)
            
            # 結果を変換
            sources = []
            for chunk in response.source_chunks:
                try:
                    if hasattr(chunk, 'original_result'):
                        # RerankedResultの場合
                        source_data = chunk.original_result.__dict__.copy()
                        # scoreプロパティまたはfinal_scoreを使用
                        if hasattr(chunk, 'score'):
                            source_data['score'] = chunk.score
                        elif hasattr(chunk, 'final_score'):
                            source_data['score'] = chunk.final_score
                        else:
                            source_data['score'] = 0.0
                    elif hasattr(chunk, '__dict__'):
                        # HybridSearchResultまたは他のオブジェクトの場合
                        # dataclassの場合、__dict__を直接使用
                        source_data = {}
                        
                        # 基本的な属性をコピー
                        if hasattr(chunk, 'id'):
                            source_data['id'] = chunk.id
                        if hasattr(chunk, 'text'):
                            source_data['text'] = chunk.text
                        if hasattr(chunk, 'metadata'):
                            source_data['metadata'] = chunk.metadata
                        
                        # スコア関連の属性を明示的に取得
                        source_data['vector_score'] = getattr(chunk, 'vector_score', 0.0)
                        source_data['keyword_score'] = getattr(chunk, 'keyword_score', 0.0)
                        source_data['hybrid_score'] = getattr(chunk, 'hybrid_score', 0.0)
                        source_data['tech_boost'] = getattr(chunk, 'tech_boost', 0.0)
                        source_data['score'] = getattr(chunk, 'hybrid_score', getattr(chunk, 'score', 0.0))
                        
                        # その他の属性
                        if hasattr(chunk, 'rank'):
                            source_data['rank'] = chunk.rank
                    else:
                        # フォールバック
                        source_data = {'text': str(chunk), 'score': 0.0}
                except Exception as e:
                    logger.warning(f"Error processing source chunk: {e}")
                    source_data = {'text': str(chunk), 'score': 0.0}
                
                # titleがない場合は作成
                if 'title' not in source_data:
                    if 'metadata' in source_data and isinstance(source_data['metadata'], dict):
                        source_data['title'] = source_data['metadata'].get('title', 'Untitled Document')
                    else:
                        source_data['title'] = 'Untitled Document'
                
                sources.append(source_data)
            
            # citationsの処理を修正
            citations = []
            if hasattr(response, 'citations'):
                for cite in response.citations:
                    if isinstance(cite, dict):
                        citations.append(cite)
                    elif hasattr(cite, '__dict__'):
                        citations.append(cite.__dict__)
                    else:
                        citations.append({'text': str(cite)})
            
            # metadataの処理
            metadata = {}
            if hasattr(response, 'generation_metadata'):
                metadata = response.generation_metadata
            elif hasattr(response, 'metadata'):
                metadata = response.metadata
            
            result = QueryResult(
                query=query_text,
                answer=response.answer,
                citations=citations,
                sources=sources,
                confidence_score=response.confidence_score,
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.info(f"Query completed in {processing_time:.2f}s, confidence: {response.confidence_score:.3f}")
            return result
            
        except (SearchError, GenerationError, VectorStoreError) as e:
            logger.error(f"Query failed with known error: {e}")
            raise
        except Exception as e:
            logger.error(f"Query failed with unexpected error: {e}")
            
            # エラー時はOllamaフォールバックを試行
            processing_time = time.time() - start_time
            return self._simple_ollama_query(query_text, top_k, processing_time, error=str(e))
    
    def _query_with_moe(self, 
                        query_text: str,
                        top_k: int,
                        search_type: str,
                        filters: Optional[Dict[str, Any]],
                        include_sources: bool) -> QueryResult:
        """MoE統合システムでクエリを実行"""
        import time
        import asyncio
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query with MoE-RAG: {query_text}")
            
            # MoE統合クエリを実行（同期的に実行）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            moe_result = loop.run_until_complete(
                self.moe_rag_system.query(
                    query=query_text,
                    top_k=top_k,
                    use_reranking=self.config.retrieval.reranking_enabled
                )
            )
            
            processing_time = time.time() - start_time
            
            # MoE結果をQueryResultに変換
            sources = []
            for doc in moe_result.retrieved_documents:
                source_data = {
                    'text': doc.get('content', ''),
                    'score': doc.get('expert_relevance_score', 0.0),
                    'title': doc.get('metadata', {}).get('source', 'Unknown'),
                    'expert': doc.get('expert', 'General'),
                    'metadata': doc.get('metadata', {})
                }
                sources.append(source_data)
            
            # 引用情報を生成
            citations = []
            for expert in moe_result.selected_experts:
                citations.append({
                    'expert': expert,
                    'score': moe_result.expert_scores.get(expert, 0.0)
                })
            
            # メタデータに MoE情報を追加
            metadata = moe_result.metadata.copy()
            metadata['moe_experts'] = moe_result.selected_experts
            metadata['moe_strategy'] = moe_result.fusion_strategy
            metadata['moe_confidence'] = moe_result.confidence
            
            result = QueryResult(
                query=query_text,
                answer=moe_result.answer,
                citations=citations,
                sources=sources,
                confidence_score=moe_result.confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.info(f"MoE query completed in {processing_time:.2f}s, experts: {', '.join(moe_result.selected_experts)}")
            return result
            
        except Exception as e:
            logger.error(f"MoE query failed: {e}")
            # フォールバック
            processing_time = time.time() - start_time
            return self._simple_ollama_query(query_text, top_k, processing_time, error=str(e))
    
    def _simple_ollama_query(self, query_text: str, top_k: int, processing_time: float, error: str = None) -> QueryResult:
        """シンプルなOllamaクエリ（メモリ不足時のフォールバック）"""
        
        try:
            if self.llm_generator and self.llm_generator.ollama:
                # 拡張されたプロンプトで詳細な回答を生成
                enhanced_prompt = self._build_enhanced_rag_prompt(query_text, "")
                
                # 設定からOllamaモデル名を取得（優先順位で試行）
                ollama_model = 'llama3.2:3b'  # デフォルトを一般的なモデルに変更
                
                # 複数の設定箇所から取得を試みる
                if hasattr(self.config.llm, 'ollama') and hasattr(self.config.llm.ollama, 'model'):
                    ollama_model = self.config.llm.ollama.model
                elif hasattr(self.config.llm, 'ollama_model'):
                    ollama_model = self.config.llm.ollama_model
                elif hasattr(self.config.llm, 'model_name') and self.config.llm.model_name.startswith('ollama:'):
                    ollama_model = self.config.llm.model_name[7:]
                
                logger.info(f"Ollamaモデル使用: {ollama_model}")
                
                result = _ollama_generate_japanese(
                    self.llm_generator.ollama, ollama_model, enhanced_prompt
                )
                
                if result.get("success", False):
                    answer = result.get("generated_text", "")
                    # 中国語簡体字を日本語に変換（ポストプロセッシング）
                    answer = self._convert_chinese_to_japanese(answer)
                    confidence = 0.6  # Ollamaフォールバックの信頼度
                else:
                    answer = f"エラー: Ollama生成に失敗 - {result.get('error', 'Unknown error')}"
                    confidence = 0.0
            else:
                answer = f"エラー: 生成モデルが利用できません。クエリ: {query_text}"
                confidence = 0.0
                
            # エラー情報を追加
            metadata = {'fallback': 'ollama', 'mode': 'simple'}
            if error:
                metadata['original_error'] = error
                answer = f"[Ollamaフォールバック] {answer}"
                
            return QueryResult(
                query=query_text,
                answer=answer,
                citations=[],
                sources=[],
                confidence_score=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Ollamaフォールバックも失敗: {e}")
            return QueryResult(
                query=query_text,
                answer=f"申し訳ございませんが、すべての処理手段が失敗しました。エラー: {str(e)}",
                citations=[],
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={'error': str(e), 'fallback_failed': True}
            )
    
    def _hybrid_search_with_ollama(self, query_text: str, top_k: int, search_type: str, filters: Dict[str, Any] = None):
        """ハイブリッド検索とOllama生成を組み合わせたクエリ処理"""
        
        try:
            logger.info(f"Ollamaフォールバックモードでハイブリッド検索を実行: {query_text}")
            
            # 1. ハイブリッド検索で関連文書を取得
            search_results = []
            context_texts = []
            
            if self.hybrid_search:
                try:
                    # 検索クエリを構築
                    search_query = SearchQuery(
                        text=query_text,
                        search_type=search_type,
                        filters=filters
                    )
                    
                    # ハイブリッド検索を実行
                    search_results = self.hybrid_search.search(
                        query=search_query,
                        top_k=top_k
                    )
                    
                    # コンテキストテキストを構築 - ファイル名を明確に表示
                    for result in search_results:
                        # ファイル名を優先的に取得
                        source_name = (
                            result.metadata.get('filename', '').replace('.pdf', '') or
                            result.metadata.get('source', '').replace('.pdf', '') or
                            result.metadata.get('title', '') or
                            '不明'
                        )
                        context_texts.append(f"[出典: {source_name}]\n{result.text}")
                    
                    logger.info(f"ハイブリッド検索で{len(search_results)}件の関連文書を取得")
                    
                except Exception as search_error:
                    logger.error(f"ハイブリッド検索エラー: {search_error}")
                    search_results = []
                    context_texts = []
            
            # 2. Ollamaでコンテキスト付き回答を生成
            context = "\n\n".join(context_texts) if context_texts else ""
            
            if self.llm_generator and self.llm_generator.ollama:
                # 拡張されたプロンプトで詳細な回答を生成
                enhanced_prompt = self._build_enhanced_rag_prompt(query_text, context)
                
                # 設定からOllamaモデル名を取得（優先順位で試行）
                ollama_model = 'llama3.2:3b'  # デフォルトを一般的なモデルに変更
                
                # 複数の設定箇所から取得を試みる
                if hasattr(self.config.llm, 'ollama') and hasattr(self.config.llm.ollama, 'model'):
                    ollama_model = self.config.llm.ollama.model
                elif hasattr(self.config.llm, 'ollama_model'):
                    ollama_model = self.config.llm.ollama_model
                elif hasattr(self.config.llm, 'model_name') and self.config.llm.model_name.startswith('ollama:'):
                    ollama_model = self.config.llm.model_name[7:]
                    
                logger.info(f"Ollamaモデル使用: {ollama_model}")
                
                result = _ollama_generate_japanese(
                    self.llm_generator.ollama, ollama_model, enhanced_prompt
                )
                
                if result.get("success", False):
                    answer = result.get("generated_text", "")
                    # 中国語簡体字を日本語に変換（ポストプロセッシング）
                    answer = self._convert_chinese_to_japanese(answer)
                    confidence = 0.8 if context_texts else 0.6  # コンテキストがある場合は高い信頼度
                else:
                    answer = f"エラー: Ollama生成に失敗 - {result.get('error', 'Unknown error')}"
                    confidence = 0.0
            else:
                answer = "エラー: Ollama生成モデルが利用できません"
                confidence = 0.0
            
            # 3. 結果を構築して返す
            # ダミーのレスポンスオブジェクトを作成
            class DummyResponse:
                def __init__(self, answer, search_results, confidence):
                    self.answer = answer
                    self.source_chunks = search_results  # 検索結果をソースとして使用
                    self.confidence_score = confidence
                    self.citations = self._build_citations(search_results)
                    self.metadata = {
                        'fallback': 'ollama',
                        'mode': 'hybrid_search_with_ollama',
                        'source_count': len(search_results)
                    }
                
                def _build_citations(self, search_results):
                    citations = []
                    for i, result in enumerate(search_results, 1):
                        citations.append({
                            'id': i,
                            'text': result.text[:200] + "..." if len(result.text) > 200 else result.text,
                            'source': result.metadata.get('title', f'文書{i}'),
                            'score': getattr(result, 'hybrid_score', getattr(result, 'score', 0.0)),
                            'vector_score': getattr(result, 'vector_score', 0.0),
                            'keyword_score': getattr(result, 'keyword_score', 0.0),
                            'hybrid_score': getattr(result, 'hybrid_score', 0.0),
                            'tech_boost': getattr(result, 'tech_boost', 0.0)
                        })
                    return citations
            
            return DummyResponse(answer, search_results, confidence)
            
        except Exception as e:
            logger.error(f"ハイブリッド検索+Ollamaエラー: {e}")
            # エラー時はシンプルモードにフォールバック
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.answer = f"エラー: {error_msg}"
                    self.source_chunks = []
                    self.confidence_score = 0.0
                    self.citations = []
                    self.metadata = {'error': error_msg}
            
            return ErrorResponse(str(e))
    
    def _build_enhanced_rag_prompt(self, query: str, context: str) -> str:
        """拡張されたRAGプロンプトを構築（3000文字程度の詳細な回答用）"""
        
        if context:
            prompt = f"""# 道路設計の専門家としての回答

あなたは経験豊富な道路設計の専門家です。以下の参考資料を基に、質問に対して**詳細で実用的な回答**を提供してください。

**重要: 回答は必ず日本語のみを使用してください。中国語の簡体字や繁体字、韓国語（ハングル）は使用しないでください。**

## 参考資料
{context}

## 質問
{query}

## 回答の指示
1. **具体的で詳細な説明**を提供してください
2. **数値や基準値**は参考資料から正確に引用してください
3. **実務での注意点やポイント**を含めてください
4. **関連する法規や基準**があれば言及してください
5. **1500-5000文字程度**の充実した回答をお願いします
6. 参考資料から情報を引用した場合は、**[出典: ファイル名]** の形式で具体的な出典ファイル名を明記してください（例: [出典: 20251201_道路舗装主材料]）
7. 回答の根拠は、日本の法令、基準、指針、要領、マニュアルの情報を活用することとし、中国の法令、基準、指針、要領、マニュアルの情報は使わないでください。
7. **日本語のみを使用**し、中国語の簡体字（例: 车、时、间）や韓国語（ハングル）は使わないでください。


## 回答"""
        else:
            prompt = f"""# 道路設計の専門家としての回答

あなたは経験豊富な日本の道路設計の専門家です。以下の質問に対して、一般的な知識を基に**詳細で実用的な回答**を提供してください。

**重要: 回答は必ず日本語のみを使用してください。中国語の簡体字や繁体字、韓国語（ハングル）は使用しないでください。**

## 質問
{query}

## 回答の指示
1. **具体的で詳細な説明**を提供してください
2. **実務での注意点やポイント**を含めてください
3. **関連する法規や基準**があれば言及してください
4. **1500-5000文字程度**の充実した回答をお願いします
5. 参考資料がないため、一般的な道路設計の知識を活用してください
6. **日本語のみを使用**し、中国語の簡体字（例: 车、时、间）や韓国語（ハングル）は使わないでください

## 回答"""
        
        return prompt
            
    def batch_query(self, 
                   queries: List[str],
                   **kwargs) -> List[QueryResult]:
        """バッチクエリを実行"""
        
        results = []
        total_queries = len(queries)
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing batch query {i}/{total_queries}")
            
            try:
                result = self.query(query, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch query {i} failed: {e}")
                # エラー結果を追加
                error_result = QueryResult(
                    query=query,
                    answer=f"エラー: {str(e)}",
                    citations=[],
                    sources=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    metadata={'error': str(e), 'batch_index': i}
                )
                results.append(error_result)
                
        return results
        
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        
        info = {
            'is_initialized': self.is_initialized,
            'config': {
                'system_name': self.config.system_name,
                'version': self.config.version,
                'language': self.config.language
            }
        }
        
        if self.is_initialized:
            try:
                # ベクトルストア情報
                if self.vector_store:
                    info['vector_store'] = self.vector_store.get_collection_info()
                    
                # メタデータ統計
                if self.metadata_manager:
                    info['metadata_stats'] = self.metadata_manager.get_statistics()
                    
                # モデル情報
                info['models'] = {
                    'embedding_model': getattr(self.embedding_model, 'model_name', 'Unknown'),
                    'llm_available': self.llm_generator.model is not None,
                    'reranker_enabled': self.reranker is not None
                }
                
            except Exception as e:
                info['error'] = f"Failed to get system info: {e}"
                
        return info
        
    def reload_config(self, config_path: Optional[str] = None):
        """設定を再読み込み"""

        logger.info("Reloading configuration...")
        self.config = load_config(config_path)

        # 必要に応じてコンポーネントを再初期化
        if self.is_initialized:
            logger.info("Reinitializing components with new config...")
            self.initialize()

    def _convert_chinese_to_japanese(self, text: str) -> str:
        """中国語の簡体字を日本語の漢字に変換（ポストプロセッシング）"""

        # よく混入する中国語簡体字 → 日本語漢字の変換マップ
        chinese_to_japanese = {
            # 交通・道路関連
            '车': '車',
            '辆': '両',
            '驾': '駕',
            '驶': '駛',
            '轮': '輪',
            '铁': '鉄',
            '路': '路',  # 同じ
            '桥': '橋',
            '隧': '隧',  # 同じ
            '灯': '灯',  # 同じ（簡体字と同じ）

            # 一般的な簡体字
            '学': '学',  # 同じ（新字体）
            '国': '国',  # 同じ（新字体）
            '时': '時',
            '间': '間',
            '实': '実',
            '际': '際',
            '经': '経',
            '验': '験',
            '应': '応',
            '该': '該',
            '证': '証',
            '标': '標',
            '规': '規',
            '设': '設',
            '计': '計',
            '备': '備',
            '记': '記',
            '认': '認',
            '为': '為',
            '号': '号',  # 同じ
            '条': '条',  # 同じ
            '项': '項',
            '务': '務',
            '业': '業',
            '区': '区',  # 同じ（新字体）
            '产': '産',
            '质': '質',
            '检': '検',
            '查': '査',
            '题': '題',
            '问': '問',
            '观': '観',
            '环': '環',
            '现': '現',
            '发': '発',
            '变': '変',
            '达': '達',
            '过': '過',
            '还': '還',
            '进': '進',
            '远': '遠',
            '连': '連',
            '运': '運',
            '迁': '遷',
            '适': '適',
            '选': '選',
            '择': '択',
            '处': '処',
            '级': '級',
            '纪': '紀',
            '约': '約',
            '组': '組',
            '织': '織',
            '维': '維',
            '综': '総',
            '线': '線',
            '练': '練',
            '继': '継',
            '续': '続',
            '统': '統',
            '绩': '績',
            '缘': '縁',
            '编': '編',
            '县': '県',
            '听': '聴',
            '职': '職',
            '联': '聯',
            '声': '声',  # 同じ
            '壳': '殻',
            '贝': '貝',
            '负': '負',
            '财': '財',
            '货': '貨',
            '贸': '貿',
            '费': '費',
            '贴': '貼',
            '贯': '貫',
            '责': '責',
            '败': '敗',
            '账': '帳',
            '货': '貨',
            '质': '質',
            '购': '購',
            '贩': '販',
            '贷': '貸',
            '资': '資',
            '赋': '賦',
            '赖': '頼',
            '赞': '賛',
            '赛': '際',
            '赢': '勝',
            '航走性': '走行性',
            '舶上': '路上',
            '舤': '路',
            '阿斯法尔ト': 'アスファルト',
            '航走性': '走行性'
        }

        # 文字列を一文字ずつ変換
        result = []
        for char in text:
            if char in chinese_to_japanese:
                result.append(chinese_to_japanese[char])
            else:
                result.append(char)

        return ''.join(result)


# グローバルエンジンインスタンス
_global_engine: Optional[RoadDesignQueryEngine] = None


def get_query_engine(config_path: Optional[str] = None) -> RoadDesignQueryEngine:
    """グローバルクエリエンジンを取得"""
    
    global _global_engine
    
    if _global_engine is None:
        _global_engine = RoadDesignQueryEngine(config_path)
        _global_engine.initialize()
        
    return _global_engine


def set_query_engine(engine: RoadDesignQueryEngine):
    """グローバルクエリエンジンを設定"""
    
    global _global_engine
    _global_engine = engine


# 便利な関数
def query_road_design(query_text: str, **kwargs) -> QueryResult:
    """道路設計クエリ（便利関数）"""
    
    engine = get_query_engine()
    return engine.query(query_text, **kwargs)


def batch_query_road_design(queries: List[str], **kwargs) -> List[QueryResult]:
    """道路設計バッチクエリ（便利関数）"""
    
    engine = get_query_engine()
    return engine.batch_query(queries, **kwargs)

# ----------------------------------------------------------------------------
# Compatibility alias
# NOTE: Historically some modules referred to a class named `QueryEngine`.
#       The canonical implementation in this codebase is `RoadDesignQueryEngine`.
#       To avoid confusion with legacy code and stale imports, we expose a
#       backwards-compatible alias here. This does not change behavior.
# ----------------------------------------------------------------------------
QueryEngine = RoadDesignQueryEngine
