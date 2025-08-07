"""
統合クエリエンジン（リファクタリング版）
検索・生成システムを統合したメインエンジン
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ollama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import RAGConfig
from ..indexing import EmbeddingModel, MetadataManager, VectorStore
from ..retrieval import HybridSearch, Reranker, SearchQuery
from ..specialized import NumericalProcessor, ValidationResult, VersionManager
from ..utils import (
    RAGException, GenerationError, SearchError,
    setup_logger, log_performance,
    validate_query, validate_top_k, validate_search_type, validate_filters,
    format_citations, format_sources, format_metadata,
    ContextLogger
)
from .citation_engine import CitationEngine

logger = setup_logger(__name__)


@dataclass
class QueryResult:
    """クエリ結果データクラス"""
    query: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMGenerator:
    """LLM生成器クラス（リファクタリング版）"""
    
    def __init__(self, config: RAGConfig):
        """初期化"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.use_ollama_fallback = False
        self.device = None
        
    @log_performance(logger)
    def initialize(self):
        """モデルの初期化"""
        with ContextLogger(logger, "LLM initialization"):
            try:
                # HuggingFaceモデルの初期化
                self._initialize_huggingface_model()
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace model: {e}")
                self._initialize_ollama_fallback()
    
    def _initialize_huggingface_model(self):
        """HuggingFaceモデルの初期化"""
        model_name = self.config.get('generation', {}).get('model_name')
        if not model_name:
            raise GenerationError("Model name not specified in config")
        
        # デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # モデルとトークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # モデルのロード（メモリ最適化）
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        
        logger.info(f"Loaded model: {model_name}")
    
    def _initialize_ollama_fallback(self):
        """Ollamaフォールバックの初期化"""
        self.use_ollama_fallback = True
        logger.info("Using Ollama as fallback LLM")
        
        # Ollamaの利用可能性をチェック
        try:
            models = ollama.list()
            if not models:
                logger.warning("No Ollama models available")
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
    
    @log_performance(logger)
    def generate(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """テキスト生成"""
        if self.use_ollama_fallback:
            return self._generate_with_ollama(prompt, max_length, temperature, top_p)
        else:
            return self._generate_with_huggingface(prompt, max_length, temperature, top_p, **kwargs)
    
    def _generate_with_huggingface(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> str:
        """HuggingFaceモデルでの生成"""
        if not self.model or not self.tokenizer:
            raise GenerationError("Model not initialized")
        
        # トークナイズ
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
        
        # デコード
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # プロンプトを除去
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _generate_with_ollama(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Ollamaでの生成"""
        try:
            response = ollama.generate(
                model="llama2",
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'top_p': top_p,
                    'num_predict': max_length
                }
            )
            return response['response']
        except Exception as e:
            raise GenerationError(f"Ollama generation failed: {e}")


class RoadDesignQueryEngine:
    """道路設計クエリエンジン（リファクタリング版）"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        vector_store_path: Optional[Path] = None,
        metadata_db_path: Optional[Path] = None
    ):
        """初期化"""
        # 設定のロード
        if config_path:
            self.config = RAGConfig.from_yaml(config_path)
        else:
            self.config = RAGConfig()
        
        # パスの設定
        self.vector_store_path = vector_store_path or Path("data/vector_store")
        self.metadata_db_path = metadata_db_path or Path("data/metadata.db")
        
        # コンポーネントの初期化
        self.embedding_model = None
        self.vector_store = None
        self.hybrid_search = None
        self.reranker = None
        self.llm_generator = None
        self.citation_engine = None
        self.metadata_manager = None
        
        self.is_initialized = False
    
    @log_performance(logger)
    def initialize(self, mode: str = "full"):
        """システムの初期化
        
        Args:
            mode: 初期化モード（'full', 'lightweight', 'minimal'）
        """
        with ContextLogger(logger, f"Query engine initialization (mode={mode})"):
            try:
                if mode == "lightweight":
                    self._initialize_lightweight_mode()
                elif mode == "minimal":
                    self._initialize_minimal_mode()
                else:
                    self._initialize_full_mode()
                
                self.is_initialized = True
                logger.info(f"Query engine initialized in {mode} mode")
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                raise RAGException(f"Failed to initialize query engine: {e}")
    
    def _initialize_full_mode(self):
        """フルモードの初期化"""
        # 埋め込みモデル
        self.embedding_model = EmbeddingModel(self.config)
        self.embedding_model.initialize()
        
        # ベクトルストア
        self.vector_store = VectorStore(
            self.config,
            self.embedding_model,
            self.vector_store_path
        )
        self.vector_store.initialize()
        
        # ハイブリッド検索
        self.hybrid_search = HybridSearch(
            self.vector_store,
            self.config
        )
        
        # リランカー
        self.reranker = Reranker(self.config)
        self.reranker.initialize()
        
        # LLMジェネレーター
        self.llm_generator = LLMGenerator(self.config)
        self.llm_generator.initialize()
        
        # 引用エンジン
        self.citation_engine = CitationEngine(
            self.hybrid_search,
            self.llm_generator,
            self.config
        )
        
        # メタデータマネージャー
        self.metadata_manager = MetadataManager(self.metadata_db_path)
    
    def _initialize_lightweight_mode(self):
        """軽量モードの初期化（埋め込みとベクトル検索のみ）"""
        # 埋め込みモデル
        self.embedding_model = EmbeddingModel(self.config)
        self.embedding_model.initialize()
        
        # ベクトルストア
        self.vector_store = VectorStore(
            self.config,
            self.embedding_model,
            self.vector_store_path
        )
        self.vector_store.initialize()
        
        # ハイブリッド検索
        self.hybrid_search = HybridSearch(
            self.vector_store,
            self.config
        )
        
        # Ollamaフォールバック付きLLM
        self.llm_generator = LLMGenerator(self.config)
        self.llm_generator.use_ollama_fallback = True
        
        logger.info("Initialized in lightweight mode (no reranker, using Ollama)")
    
    def _initialize_minimal_mode(self):
        """最小モードの初期化（Ollamaのみ）"""
        # Ollamaのみ使用
        self.llm_generator = LLMGenerator(self.config)
        self.llm_generator.use_ollama_fallback = True
        
        logger.info("Initialized in minimal mode (Ollama only)")
    
    @log_performance(logger)
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> QueryResult:
        """クエリの実行
        
        Args:
            query_text: クエリテキスト
            top_k: 取得する文書数
            search_type: 検索タイプ
            filters: フィルター条件
            include_sources: ソースを含めるか
        
        Returns:
            クエリ結果
        """
        if not self.is_initialized:
            raise RAGException("Query engine must be initialized before use")
        
        start_time = time.time()
        
        try:
            # バリデーション
            query_text = validate_query(query_text)
            top_k = validate_top_k(top_k)
            search_type = validate_search_type(search_type)
            filters = validate_filters(filters)
            
            logger.info(f"Processing query: {query_text}")
            
            # 検索と生成
            if self.citation_engine:
                result = self._query_with_citation(
                    query_text, top_k, search_type, filters, include_sources
                )
            elif self.hybrid_search:
                result = self._query_with_hybrid_search(
                    query_text, top_k, search_type, filters
                )
            else:
                result = self._query_with_ollama_only(query_text)
            
            # 処理時間の記録
            result.processing_time = time.time() - start_time
            
            # メタデータのフォーマット
            result.metadata = format_metadata(result.metadata)
            
            logger.info(
                f"Query completed in {result.processing_time:.2f}s, "
                f"confidence: {result.confidence_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
            # エラー時のフォールバック
            processing_time = time.time() - start_time
            return self._create_error_result(query_text, str(e), processing_time)
    
    def _query_with_citation(
        self,
        query_text: str,
        top_k: int,
        search_type: str,
        filters: Optional[Dict[str, Any]],
        include_sources: bool
    ) -> QueryResult:
        """引用エンジンを使用したクエリ"""
        response = self.citation_engine.query(
            query_text=query_text,
            top_k=top_k,
            include_sources=include_sources,
            filters=filters
        )
        
        # 結果の変換
        citations = format_citations(response.citations)
        sources = format_sources(response.source_chunks, include_metadata=True)
        
        return QueryResult(
            query=query_text,
            answer=response.answer,
            citations=citations,
            sources=sources,
            confidence_score=response.confidence_score,
            metadata=response.generation_metadata
        )
    
    def _query_with_hybrid_search(
        self,
        query_text: str,
        top_k: int,
        search_type: str,
        filters: Optional[Dict[str, Any]]
    ) -> QueryResult:
        """ハイブリッド検索を使用したクエリ"""
        # 検索の実行
        search_query = SearchQuery(
            text=query_text,
            search_type=search_type,
            filters=filters
        )
        
        search_results = self.hybrid_search.search(search_query, top_k=top_k)
        
        # リランキング（可能な場合）
        if self.reranker:
            search_results = self.reranker.rerank(query_text, search_results)
        
        # プロンプトの構築
        context = "\n\n".join([r.text for r in search_results[:top_k]])
        prompt = self._build_rag_prompt(query_text, context)
        
        # 回答の生成
        answer = self.llm_generator.generate(prompt)
        
        # 結果の作成
        sources = format_sources([r.__dict__ for r in search_results])
        
        return QueryResult(
            query=query_text,
            answer=answer,
            citations=[],
            sources=sources,
            confidence_score=self._calculate_confidence(search_results),
            metadata={'search_type': search_type, 'top_k': top_k}
        )
    
    def _query_with_ollama_only(self, query_text: str) -> QueryResult:
        """Ollamaのみを使用したクエリ"""
        if not self.llm_generator:
            raise GenerationError("No LLM generator available")
        
        # 直接生成
        answer = self.llm_generator.generate(query_text)
        
        return QueryResult(
            query=query_text,
            answer=answer,
            citations=[],
            sources=[],
            confidence_score=0.5,
            metadata={'mode': 'ollama_only'}
        )
    
    def _create_error_result(
        self,
        query_text: str,
        error_message: str,
        processing_time: float
    ) -> QueryResult:
        """エラー結果の作成"""
        return QueryResult(
            query=query_text,
            answer=f"申し訳ございません。エラーが発生しました: {error_message}",
            citations=[],
            sources=[],
            confidence_score=0.0,
            processing_time=processing_time,
            metadata={'error': error_message}
        )
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """RAGプロンプトの構築"""
        return f"""以下のコンテキストを参考に、質問に日本語で回答してください。

コンテキスト:
{context}

質問: {query}

回答:"""
    
    def _calculate_confidence(self, search_results: List[Any]) -> float:
        """信頼度スコアの計算"""
        if not search_results:
            return 0.0
        
        scores = []
        for result in search_results[:3]:  # 上位3件のスコアを使用
            if hasattr(result, 'score'):
                scores.append(result.score)
            elif hasattr(result, 'final_score'):
                scores.append(result.final_score)
        
        if scores:
            return float(np.mean(scores))
        return 0.5
    
    @log_performance(logger)
    def batch_query(
        self,
        queries: List[str],
        **kwargs
    ) -> List[QueryResult]:
        """バッチクエリの実行"""
        results = []
        
        for query in queries:
            try:
                result = self.query(query, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                results.append(
                    self._create_error_result(query, str(e), 0.0)
                )
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報の取得"""
        info = {
            'initialized': self.is_initialized,
            'components': {
                'embedding_model': self.embedding_model is not None,
                'vector_store': self.vector_store is not None,
                'hybrid_search': self.hybrid_search is not None,
                'reranker': self.reranker is not None,
                'llm_generator': self.llm_generator is not None,
                'citation_engine': self.citation_engine is not None,
                'metadata_manager': self.metadata_manager is not None
            },
            'config': {
                'vector_store_path': str(self.vector_store_path),
                'metadata_db_path': str(self.metadata_db_path)
            }
        }
        
        # GPU情報
        if torch.cuda.is_available():
            info['gpu'] = {
                'available': True,
                'device_name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
        else:
            info['gpu'] = {'available': False}
        
        return info
    
    def reload_config(self, config_path: Path):
        """設定の再読み込み"""
        self.config = RAGConfig.from_yaml(config_path)
        logger.info(f"Config reloaded from {config_path}")
        
        # 再初期化が必要な場合は通知
        if self.is_initialized:
            logger.warning(
                "Configuration reloaded. "
                "Call initialize() again to apply changes."
            )


# グローバルエンジンインスタンス（シングルトン）
_global_engine: Optional[RoadDesignQueryEngine] = None


def get_query_engine(
    config_path: Optional[Path] = None,
    force_reinit: bool = False
) -> RoadDesignQueryEngine:
    """クエリエンジンのシングルトンインスタンスを取得"""
    global _global_engine
    
    if _global_engine is None or force_reinit:
        _global_engine = RoadDesignQueryEngine(config_path)
        _global_engine.initialize()
    
    return _global_engine


def set_query_engine(engine: RoadDesignQueryEngine):
    """グローバルエンジンを設定"""
    global _global_engine
    _global_engine = engine


# 便利な関数
@log_performance(logger)
def query_road_design(
    query_text: str,
    **kwargs
) -> QueryResult:
    """道路設計に関するクエリを実行"""
    engine = get_query_engine()
    return engine.query(query_text, **kwargs)


@log_performance(logger)
async def batch_query_road_design(
    queries: List[str],
    **kwargs
) -> List[QueryResult]:
    """非同期バッチクエリ"""
    engine = get_query_engine()
    
    # 非同期実行
    tasks = []
    for query in queries:
        task = asyncio.create_task(
            asyncio.to_thread(engine.query, query, **kwargs)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # エラーハンドリング
    processed_results = []
    for query, result in zip(queries, results):
        if isinstance(result, Exception):
            logger.error(f"Async query failed for '{query}': {result}")
            processed_results.append(
                engine._create_error_result(query, str(result), 0.0)
            )
        else:
            processed_results.append(result)
    
    return processed_results