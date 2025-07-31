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
    """LLM生成器"""
    
    def __init__(self, config: RAGConfig):
        """
        Args:
            config: RAG設定
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        
    def _load_model(self):
        """モデルを読み込み"""
        
        llm_config = self.config.llm
        
        # ファインチューニング済みモデルを優先
        if llm_config.use_finetuned and os.path.exists(llm_config.finetuned_model_path):
            model_path = llm_config.finetuned_model_path
            logger.info(f"Loading fine-tuned model: {model_path}")
        else:
            model_path = llm_config.base_model
            logger.info(f"Loading base model: {model_path}")
            
        try:
            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # モデルの読み込み
            model_kwargs = {
                'torch_dtype': getattr(torch, llm_config.torch_dtype),
                'device_map': llm_config.device_map,
            }
            
            if llm_config.load_in_8bit:
                model_kwargs['load_in_8bit'] = True
                
            if llm_config.max_memory:
                model_kwargs['max_memory'] = llm_config.max_memory
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
            
    def generate(self, 
                prompt: str, 
                context: str,
                max_new_tokens: Optional[int] = None) -> str:
        """テキストを生成"""
        
        if not self.model or not self.tokenizer:
            return self._fallback_generation(prompt, context)
            
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
            ).to(self.device)
            
            # 生成実行
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=llm_config.temperature,
                    top_p=llm_config.top_p,
                    repetition_penalty=llm_config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # デコード
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fallback_generation(prompt, context)
            
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
        """エンジンを初期化"""
        
        logger.info("Initializing RoadDesignQueryEngine...")
        
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
            self.llm_generator = LLMGenerator(self.config)
            
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
            raise
            
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
            # 検索クエリを構築
            search_query = SearchQuery(
                text=query_text,
                search_type=search_type,
                filters=filters
            )
            
            # 引用エンジンでクエリを実行
            response = self.citation_engine.query(
                query_text=query_text,
                top_k=top_k,
                include_sources=include_sources,
                filters=filters
            )
            
            processing_time = time.time() - start_time
            
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
            
            result = QueryResult(
                query=query_text,
                answer=response.answer,
                citations=[cite.__dict__ for cite in response.citations],
                sources=sources,
                confidence_score=response.confidence_score,
                processing_time=processing_time,
                metadata=response.generation_metadata
            )
            
            logger.info(f"Query completed in {processing_time:.2f}s, confidence: {response.confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
            # エラー時のフォールバック
            processing_time = time.time() - start_time
            return QueryResult(
                query=query_text,
                answer=f"申し訳ございませんが、クエリの処理中にエラーが発生しました: {str(e)}",
                citations=[],
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
            
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