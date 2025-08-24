"""
非同期処理対応のクエリエンジン
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)

from .query_engine import QueryResult, RoadDesignQueryEngine
from ..retrieval.hybrid_search import SearchQuery
from ..utils.exceptions import (
    SearchError,
    GenerationError,
    QueryTimeoutError,
    RAGException
)


@dataclass
class AsyncQueryResult(QueryResult):
    """非同期クエリ結果"""
    execution_time_breakdown: Optional[Dict[str, float]] = None


class AsyncRoadDesignQueryEngine:
    """非同期処理対応のRAGクエリエンジン"""
    
    def __init__(self, base_engine: Optional[RoadDesignQueryEngine] = None):
        """
        初期化
        
        Args:
            base_engine: ベースとなる同期エンジン
        """
        self.base_engine = base_engine or RoadDesignQueryEngine()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
    
    async def initialize(self, mode: str = "full"):
        """非同期初期化"""
        if self._initialized:
            return
            
        logger.info(f"Initializing AsyncQueryEngine in {mode} mode")
        
        # 同期的な初期化をexecutorで実行
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.base_engine.initialize,
            mode
        )
        
        self._initialized = True
        logger.info("AsyncQueryEngine initialized successfully")
    
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        timeout: Optional[float] = 30.0
    ) -> AsyncQueryResult:
        """
        非同期クエリ実行
        
        Args:
            query_text: クエリテキスト
            top_k: 取得する上位結果数
            search_type: 検索タイプ
            filters: フィルター条件
            include_sources: ソース情報を含めるか
            timeout: タイムアウト秒数
        
        Returns:
            AsyncQueryResult: クエリ結果
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        time_breakdown = {}
        
        try:
            # タイムアウト付きで実行
            if timeout:
                result = await asyncio.wait_for(
                    self._execute_query(
                        query_text, top_k, search_type, 
                        filters, include_sources, time_breakdown
                    ),
                    timeout=timeout
                )
            else:
                result = await self._execute_query(
                    query_text, top_k, search_type,
                    filters, include_sources, time_breakdown
                )
            
            # 実行時間の記録
            total_time = time.time() - start_time
            time_breakdown['total'] = total_time
            
            # AsyncQueryResultに変換
            return AsyncQueryResult(
                query=result.query,
                answer=result.answer,
                citations=result.citations,
                sources=result.sources,
                confidence_score=result.confidence_score,
                processing_time=total_time,
                metadata=result.metadata,
                execution_time_breakdown=time_breakdown
            )
            
        except asyncio.TimeoutError:
            raise QueryTimeoutError(
                f"Query timed out after {timeout} seconds",
                timeout_seconds=timeout,
                query=query_text
            )
        except Exception as e:
            logger.error(f"Async query failed: {e}")
            raise
    
    async def _execute_query(
        self,
        query_text: str,
        top_k: int,
        search_type: str,
        filters: Optional[Dict[str, Any]],
        include_sources: bool,
        time_breakdown: Dict[str, float]
    ) -> QueryResult:
        """クエリの実際の実行"""
        
        # 検索と生成を並列実行
        search_task = asyncio.create_task(
            self._async_search(query_text, top_k, search_type, filters)
        )
        
        # 検索結果を待つ
        search_start = time.time()
        search_results = await search_task
        time_breakdown['search'] = time.time() - search_start
        
        # 生成処理
        generation_start = time.time()
        result = await self._async_generate(
            query_text, search_results, include_sources
        )
        time_breakdown['generation'] = time.time() - generation_start
        
        return result
    
    async def _async_search(
        self,
        query_text: str,
        top_k: int,
        search_type: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """非同期検索"""
        loop = asyncio.get_event_loop()
        
        # ハイブリッド検索の場合、ベクトル検索とキーワード検索を並列実行
        if search_type == "hybrid" and self.base_engine.hybrid_search:
            # ベクトル検索とキーワード検索を並列実行
            vector_task = loop.run_in_executor(
                self.executor,
                self._vector_search,
                query_text, top_k, filters
            )
            
            keyword_task = loop.run_in_executor(
                self.executor,
                self._keyword_search,
                query_text, top_k, filters
            )
            
            # 両方の結果を待つ
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task
            )
            
            # 結果をマージ
            return self._merge_search_results(vector_results, keyword_results)
        else:
            # 単一検索の場合
            return await loop.run_in_executor(
                self.executor,
                self._single_search,
                query_text, top_k, search_type, filters
            )
    
    def _vector_search(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """ベクトル検索（同期）"""
        if not self.base_engine.vector_store:
            return []
        
        try:
            # 埋め込み生成
            embedding = self.base_engine.embedding_model.encode([query_text])[0]
            
            # ベクトル検索
            results = self.base_engine.vector_store.search(
                query_embedding=embedding,
                top_k=top_k * 2,  # マージ用に多めに取得
                filters=filters
            )
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Vector search failed", search_type="vector", query=query_text)
    
    def _keyword_search(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """キーワード検索（同期）"""
        if not self.base_engine.hybrid_search:
            return []
        
        try:
            # キーワード検索を実行
            search_query = SearchQuery(
                text=query_text,
                search_type="keyword",
                filters=filters
            )
            
            results = self.base_engine.hybrid_search.search(
                query=search_query,
                top_k=top_k * 2  # マージ用に多めに取得
            )
            return results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise SearchError(f"Keyword search failed", search_type="keyword", query=query_text)
    
    def _single_search(
        self,
        query_text: str,
        top_k: int,
        search_type: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """単一検索（同期）"""
        try:
            search_query = SearchQuery(
                text=query_text,
                search_type=search_type,
                filters=filters
            )
            
            if self.base_engine.hybrid_search:
                return self.base_engine.hybrid_search.search(
                    query=search_query,
                    top_k=top_k
                )
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed", search_type=search_type, query=query_text)
    
    def _merge_search_results(
        self,
        vector_results: List[Any],
        keyword_results: List[Any]
    ) -> List[Any]:
        """検索結果のマージ"""
        # 重複排除とスコアの統合
        merged = {}
        
        # ベクトル検索結果（重み0.7）
        for result in vector_results:
            doc_id = getattr(result, 'id', str(result))
            merged[doc_id] = {
                'result': result,
                'vector_score': getattr(result, 'score', 0.0) * 0.7,
                'keyword_score': 0.0
            }
        
        # キーワード検索結果（重み0.3）
        for result in keyword_results:
            doc_id = getattr(result, 'id', str(result))
            if doc_id in merged:
                merged[doc_id]['keyword_score'] = getattr(result, 'score', 0.0) * 0.3
            else:
                merged[doc_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': getattr(result, 'score', 0.0) * 0.3
                }
        
        # 合計スコアで並べ替え
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x['vector_score'] + x['keyword_score'],
            reverse=True
        )
        
        return [item['result'] for item in sorted_results]
    
    async def _async_generate(
        self,
        query_text: str,
        search_results: List[Any],
        include_sources: bool
    ) -> QueryResult:
        """非同期生成"""
        loop = asyncio.get_event_loop()
        
        # 同期的な生成処理をexecutorで実行
        return await loop.run_in_executor(
            self.executor,
            self._generate_response,
            query_text, search_results, include_sources
        )
    
    def _generate_response(
        self,
        query_text: str,
        search_results: List[Any],
        include_sources: bool
    ) -> QueryResult:
        """レスポンス生成（同期）"""
        try:
            # コンテキスト作成
            context = self._build_context(search_results)
            
            # LLM生成
            if self.base_engine.llm_generator:
                answer = self.base_engine.llm_generator.generate(
                    query=query_text,
                    context=context
                )
            else:
                answer = "生成モデルが利用できません"
            
            # ソース情報の処理
            sources = []
            if include_sources:
                for result in search_results[:5]:
                    sources.append({
                        'text': getattr(result, 'text', str(result)),
                        'score': getattr(result, 'score', 0.0),
                        'metadata': getattr(result, 'metadata', {})
                    })
            
            return QueryResult(
                query=query_text,
                answer=answer,
                citations=[],
                sources=sources,
                confidence_score=0.8,
                processing_time=0.0,
                metadata={'async': True}
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Failed to generate response", fallback_available=False)
    
    def _build_context(self, search_results: List[Any]) -> str:
        """コンテキスト構築"""
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):
            text = getattr(result, 'text', str(result))
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    async def batch_query(
        self,
        queries: List[str],
        top_k: int = 5,
        max_concurrent: int = 3
    ) -> List[AsyncQueryResult]:
        """
        バッチクエリの非同期実行
        
        Args:
            queries: クエリリスト
            top_k: 各クエリの取得結果数
            max_concurrent: 最大同時実行数
        
        Returns:
            結果リスト
        """
        if not self._initialized:
            await self.initialize()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_query(query_text: str) -> AsyncQueryResult:
            async with semaphore:
                return await self.query(query_text, top_k=top_k)
        
        # 全クエリを並列実行
        tasks = [limited_query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーハンドリング
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch query {i} failed: {result}")
                # エラー時のフォールバック結果
                final_results.append(AsyncQueryResult(
                    query=queries[i],
                    answer=f"エラー: {str(result)}",
                    citations=[],
                    sources=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    metadata={'error': str(result)}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def close(self):
        """リソースのクリーンアップ"""
        self.executor.shutdown(wait=True)
        logger.info("AsyncQueryEngine closed")


# ヘルパー関数
async def create_async_engine(mode: str = "full") -> AsyncRoadDesignQueryEngine:
    """非同期エンジンの作成と初期化"""
    engine = AsyncRoadDesignQueryEngine()
    await engine.initialize(mode)
    return engine