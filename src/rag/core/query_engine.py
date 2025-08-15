"""
統合クエリエンジン
検索・生成システムを統合したメインエンジン
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

# RAGコンポーネントのインポート
from ..indexing.vector_store import QdrantVectorStore
from ..indexing.embedding_model import EmbeddingModelFactory
from ..indexing.metadata_manager import MetadataManager
from ..retrieval.hybrid_search import HybridSearchEngine, SearchQuery
from ..retrieval.reranker import HybridReranker
from ..core.citation_engine import CitationQueryEngine, GeneratedResponse
from ..config.rag_config import RAGConfig, load_config


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
        self.use_ollama_fallback = True  # デフォルトでOllamaを使用
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
            except Exception as e:
                logger.warning(f"Failed to initialize continual learning manager: {e}")
                self.use_continual = False
        
        # Ollamaモードを優先的に初期化
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
        
        for i in range(gpu_count):
            free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
            max_free_memory = max(max_free_memory, free_memory)
        
        # 22Bモデルには最低30GBが必要
        required_memory = 30
        return max_free_memory >= required_memory
    
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
                except Exception as gpu_error:
                    if "GPU" in str(gpu_error) or "CUDA" in str(gpu_error):
                        logger.warning(f"GPU読み込み失敗: {gpu_error}")
                        logger.info("CPUモードで再試行します")
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
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # モデルロード失敗時はOllamaフォールバックを試行
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
                # モデル名をチェックしてQwen2ForCausalLMかどうかを判定
                if hasattr(llm_config, 'model_name'):
                    model_name = str(llm_config.model_name).lower()
                elif hasattr(llm_config, 'base_model'):
                    model_name = str(llm_config.base_model).lower()
                else:
                    model_name = ''
                    
                if 'qwen' not in model_name:
                    model_kwargs.update({
                        'offload_folder': offload_dir,  # オフロードディレクトリを追加
                        'offload_state_dict': True   # 状態辞書のオフロードを有効化
                    })
                    logger.info("offload_folderを有効化しました")
                else:
                    logger.info("Qwen2ForCausalLMのため、offload_folderを無効化しました")
            except Exception as e:
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
                    except Exception as e:
                        logger.error(f"Failed to generate with continual model: {e}")
                        # エラー時は元のモデルに戻す
                        self.model = original_model
                        self.tokenizer = original_tokenizer
        
        # 通常のモデル選択ロジック
        # モデルが未ロードの場合、オンデマンドでロード
        if not self.model and not self.use_ollama_fallback:
            logger.info("Model not loaded, attempting on-demand loading...")
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load model on-demand: {e}")
                # ロード失敗時はOllamaフォールバックに切り替え
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
            
        except Exception as e:
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
            
        except Exception as e:
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
            
            # Ollamaで生成
            result = self.ollama.generate_text(
                model_name="llama3.2:3b",
                prompt=full_prompt,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024
            )
            
            if result.get("success", False):
                generated_text = result.get("generated_text", "")
                logger.info("Ollamaでの生成が成功しました")
                return memory_warning + generated_text if memory_warning else generated_text
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Ollama生成エラー: {error_msg}")
                return f"エラー: Ollama生成に失敗しました - {error_msg}{memory_warning}"
                
        except Exception as e:
            logger.error(f"Ollamaフォールバックエラー: {e}")
            return f"エラー: 生成に失敗しました - {str(e)}{memory_warning}"
            
    def _build_prompt(self, query: str, context: str) -> str:
        """プロンプトを構築"""
        
        prompt_template = """あなたは道路設計の専門家です。以下の参考資料に基づいて、質問に正確に回答してください。

重要な指示:
1. 数値や基準値は必ず参考資料から正確に引用すること
2. 該当する条文番号や表番号を明記すること
3. 複数の基準がある場合は、すべて列挙すること
4. 不明な場合は推測せず「参考資料に該当する情報が見つかりません」と回答すること
5. 回答は簡潔で実践的にすること

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
        
        self.is_initialized = False
        
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
            self.embedding_model = EmbeddingModelFactory.create(
                model_type="multilingual-e5-large",  # 設定から取得する場合は修正
                device=embedding_config.device
            )
            
            # 2. ベクトルストア
            logger.info("Loading vector store...")
            embedding_dim = EmbeddingModelFactory.get_embedding_dim("multilingual-e5-large")
            
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
            self.hybrid_search = HybridSearchEngine(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                vector_weight=retrieval_config.vector_weight,
                keyword_weight=retrieval_config.keyword_weight
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
            
            self.is_initialized = True
            logger.info("RoadDesignQueryEngine initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize RoadDesignQueryEngine: {e}")
            # 初期化失敗時は軽量モードでリトライ
            logger.warning("標準初期化失敗。軽量モードでリトライします")
            self._initialize_lightweight_mode()
    
    def _initialize_lightweight_mode(self):
        """メモリ不足時の軽量モード初期化"""
        
        logger.info("軽量モードで初期化中...")
        
        try:
            # 1. 基本コンポーネントのみ初期化
            logger.info("Loading lightweight embedding model...")
            self.embedding_model = EmbeddingModelFactory.create_model(
                model_type="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 軽量モデル
                device="cpu" if torch.cuda.mem_get_info()[0] / (1024**3) < 4 else "cuda"
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
            self.hybrid_search = HybridSearchEngine(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                vector_weight=0.7,  # ベクター検索主体
                keyword_weight=0.3   # キーワード検索も併用
            )
            
            # コーパスを初期化
            self._initialize_search_corpus()
            
            # 5. OllamaベースのLLM生成器
            self.llm_generator = LLMGenerator(self.config, load_model=False)
            self.llm_generator._enable_ollama_fallback()  # 強制的にOllamaモード
            
            # 6. シンプルな引用エンジン
            self.citation_engine = CitationQueryEngine(
                hybrid_search_engine=self.hybrid_search,
                llm_generator=self.llm_generator,
                metadata_manager=self.metadata_manager
            )
            
            self.is_initialized = True
            logger.info("軽量モードでの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"軽量モード初期化も失敗: {e}")
            # 最低限の機能で初期化
            self._initialize_minimal_mode()
    
    def _initialize_minimal_mode(self):
        """最低限の機能で初期化（Ollamaのみ）"""
        
        logger.warning("最低限モードで初期化中...")
        
        try:
            # Ollamaのみで動作するシンプルなモード
            self.llm_generator = LLMGenerator(self.config, load_model=False)
            self.llm_generator._enable_ollama_fallback()
            
            # ダミーのメタデータマネージャー
            self.metadata_manager = MetadataManager(db_path=":memory:")  # インメモリデータベース
            
            # ダミーのハイブリッド検索（基本的な機能のみ）
            self.hybrid_search = None  # 最低限モードでは無効
            
            self.is_initialized = True
            logger.info("最低限モードでの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"最低限モード初期化も失敗: {e}")
            raise RuntimeError("すべての初期化が失敗しました")
            
    def _initialize_search_corpus(self):
        """検索用コーパスを初期化"""
        
        try:
            # メタデータから文書情報を取得
            documents = self.metadata_manager.search_documents()
            
            if documents:
                # 簡易実装: 実際にはベクトルストアから情報を取得
                corpus_texts = [f"Document: {doc.title}" for doc in documents[:100]]
                corpus_ids = [doc.id for doc in documents[:100]]
                
                self.hybrid_search.initialize(corpus_texts, corpus_ids)
                logger.info(f"Initialized search corpus with {len(corpus_texts)} documents")
            else:
                logger.warning("No documents found in metadata database, initializing with empty corpus")
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
        
        try:
            # 引用エンジンがない場合のみシンプルフォールバック
            if not self.citation_engine and not self.hybrid_search:
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
                        # HybridSearchResultの場合
                        source_data = chunk.__dict__.copy()
                        # scoreが存在しない場合のフォールバック
                        if 'score' not in source_data:
                            source_data['score'] = getattr(chunk, 'score', 0.0)
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
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
            # エラー時はOllamaフォールバックを試行
            processing_time = time.time() - start_time
            return self._simple_ollama_query(query_text, top_k, processing_time, error=str(e))
    
    def _simple_ollama_query(self, query_text: str, top_k: int, processing_time: float, error: str = None) -> QueryResult:
        """シンプルなOllamaクエリ（メモリ不足時のフォールバック）"""
        
        try:
            if self.llm_generator and self.llm_generator.ollama:
                # 拡張されたプロンプトで詳細な回答を生成
                enhanced_prompt = self._build_enhanced_rag_prompt(query_text, "")
                
                result = self.llm_generator.ollama.generate_text(
                    model_name="llama3.2:3b",
                    prompt=enhanced_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048  # 文字数を大幅に拡張
                )
                
                if result.get("success", False):
                    answer = result.get("generated_text", "")
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
                    
                    # コンテキストテキストを構築
                    for result in search_results:
                        context_texts.append(f"[出典: {result.metadata.get('title', '不明')}]\n{result.text}")
                    
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
                
                result = self.llm_generator.ollama.generate_text(
                    model_name="llama3.2:3b",
                    prompt=enhanced_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048  # 文字数を大幅に拡張
                )
                
                if result.get("success", False):
                    answer = result.get("generated_text", "")
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
                            'score': getattr(result, 'hybrid_score', getattr(result, 'score', 0.0))
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
        """拡張されたRAGプロンプトを構築（2000文字程度の詳細な回答用）"""
        
        if context:
            prompt = f"""# 道路設計の専門家としての回答

あなたは経験豊富な道路設計の専門家です。以下の参考資料を基に、質問に対して**詳細で実用的な回答**を提供してください。

## 参考資料
{context}

## 質問
{query}

## 回答の指示
1. **具体的で詳細な説明**を提供してください
2. **数値や基準値**は参考資料から正確に引用してください
3. **実務での注意点やポイント**を含めてください
4. **関連する法規や基準**があれば言及してください
5. **1500-2000文字程度**の充実した回答をお願いします
6. 参考資料の情報を根拠として、**[出典: …]という形で出典を明記**してください

## 回答"""
        else:
            prompt = f"""# 道路設計の専門家としての回答

あなたは経験豊富な道路設計の専門家です。以下の質問に対して、一般的な知識を基に**詳細で実用的な回答**を提供してください。

## 質問
{query}

## 回答の指示
1. **具体的で詳細な説明**を提供してください
2. **実務での注意点やポイント**を含めてください
3. **関連する法規や基準**があれば言及してください
4. **1500-2000文字程度**の充実した回答をお願いします
5. 参考資料がないため、一般的な道路設計の知識を活用してください

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