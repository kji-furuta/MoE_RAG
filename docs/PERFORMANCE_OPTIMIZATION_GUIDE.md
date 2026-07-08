# Redisキャッシュとパフォーマンス最適化実装ガイド

## パフォーマンス最適化の全体像

```
┌──────────────┐
│   User Query │
└──────┬───────┘
       ▼
┌──────────────┐     Hit      ┌──────────────┐
│ Redis Cache  │─────────────▶│ Fast Response│
└──────┬───────┘               └──────────────┘
       │ Miss
       ▼
┌──────────────┐
│ Vector Search│
└──────┬───────┘
       ▼
┌──────────────┐
│ LLM Process  │
└──────┬───────┘
       ▼
┌──────────────┐
│ Cache & Return│
└──────────────┘
```

## 1. Redisキャッシュ実装

### キャッシュ戦略

```python
# app/cache/redis_cache.py
import redis
import json
import hashlib
import pickle
from typing import Optional, Any, Dict
from datetime import timedelta
import asyncio
from functools import wraps

class RedisCache:
    """Redis キャッシュマネージャー"""
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600  # 1時間
    ):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # バイナリデータを扱うため
        )
        self.default_ttl = default_ttl
        
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """キャッシュキーを生成"""
        # パラメータをソートして一貫性のあるキーを生成
        sorted_params = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    # ===== 基本的なキャッシュ操作 =====
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュに値を設定"""
        try:
            serialized = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """キャッシュから削除"""
        return self.redis_client.delete(key) > 0
    
    def exists(self, key: str) -> bool:
        """キーの存在確認"""
        return self.redis_client.exists(key) > 0
    
    # ===== RAG専用キャッシュ機能 =====
    
    def cache_query_result(
        self, 
        query: str, 
        result: Dict[str, Any], 
        ttl: int = 3600
    ) -> bool:
        """クエリ結果をキャッシュ"""
        key = self._generate_key("query", {"q": query})
        return self.set(key, result, ttl)
    
    def get_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """キャッシュからクエリ結果を取得"""
        key = self._generate_key("query", {"q": query})
        return self.get(key)
    
    def cache_embedding(
        self,
        text: str,
        embedding: list,
        model_name: str,
        ttl: int = 86400  # 24時間
    ) -> bool:
        """埋め込みベクトルをキャッシュ"""
        key = self._generate_key("embedding", {
            "text": text,
            "model": model_name
        })
        return self.set(key, embedding, ttl)
    
    def get_embedding(self, text: str, model_name: str) -> Optional[list]:
        """キャッシュから埋め込みベクトルを取得"""
        key = self._generate_key("embedding", {
            "text": text,
            "model": model_name
        })
        return self.get(key)
    
    def cache_document_chunks(
        self,
        doc_id: str,
        chunks: list,
        ttl: int = 86400
    ) -> bool:
        """文書チャンクをキャッシュ"""
        key = f"doc_chunks:{doc_id}"
        return self.set(key, chunks, ttl)
    
    def get_document_chunks(self, doc_id: str) -> Optional[list]:
        """キャッシュから文書チャンクを取得"""
        key = f"doc_chunks:{doc_id}"
        return self.get(key)
    
    # ===== デコレーター =====
    
    def cached(self, prefix: str, ttl: Optional[int] = None):
        """関数の結果をキャッシュするデコレーター"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # キャッシュキーを生成
                cache_key = self._generate_key(prefix, {
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                
                # キャッシュから取得を試みる
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # キャッシュミスの場合は実行
                result = await func(*args, **kwargs)
                
                # 結果をキャッシュ
                self.set(cache_key, result, ttl or self.default_ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # キャッシュキーを生成
                cache_key = self._generate_key(prefix, {
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                
                # キャッシュから取得を試みる
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # キャッシュミスの場合は実行
                result = func(*args, **kwargs)
                
                # 結果をキャッシュ
                self.set(cache_key, result, ttl or self.default_ttl)
                
                return result
            
            # 非同期関数か同期関数かを判定
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    # ===== 統計情報 =====
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        info = self.redis_client.info('stats')
        return {
            'total_connections': info.get('total_connections_received', 0),
            'total_commands': info.get('total_commands_processed', 0),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': self._calculate_hit_rate(info),
            'used_memory': self.redis_client.info('memory').get('used_memory_human', 'N/A')
        }
    
    def _calculate_hit_rate(self, info: dict) -> float:
        """ヒット率を計算"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
```

## 2. クエリ最適化実装

### ベクトル検索の最適化

```python
# app/optimization/query_optimizer.py
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import faiss
import time

@dataclass
class OptimizedSearchResult:
    """最適化された検索結果"""
    documents: List[Dict[str, Any]]
    search_time: float
    cache_hit: bool
    optimization_applied: List[str]

class QueryOptimizer:
    """クエリ最適化エンジン"""
    
    def __init__(self, cache: RedisCache, vector_dim: int = 768):
        self.cache = cache
        self.vector_dim = vector_dim
        self.index = None
        self._initialize_faiss_index()
        
    def _initialize_faiss_index(self):
        """FAISSインデックスの初期化（高速ベクトル検索）"""
        # IVFフラットインデックス（大規模データ用）
        nlist = 100  # クラスタ数
        self.quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, 
            self.vector_dim, 
            nlist,
            faiss.METRIC_L2
        )
        
    def optimize_vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        use_cache: bool = True
    ) -> OptimizedSearchResult:
        """ベクトル検索の最適化"""
        start_time = time.time()
        optimizations = []
        
        # 1. キャッシュチェック
        if use_cache:
            cache_key = f"vsearch:{hash(query_vector.tobytes())}:{top_k}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return OptimizedSearchResult(
                    documents=cached_result,
                    search_time=time.time() - start_time,
                    cache_hit=True,
                    optimization_applied=["cache_hit"]
                )
        
        # 2. 近似最近傍探索（ANN）
        if self.index and self.index.is_trained:
            distances, indices = self.index.search(
                query_vector.reshape(1, -1), 
                top_k
            )
            optimizations.append("ann_search")
        else:
            # フォールバック：通常の検索
            distances, indices = self._brute_force_search(query_vector, top_k)
            optimizations.append("brute_force")
        
        # 3. 結果の後処理
        documents = self._post_process_results(indices[0], distances[0])
        
        # 4. キャッシュに保存
        if use_cache:
            self.cache.set(cache_key, documents, ttl=1800)  # 30分
            
        return OptimizedSearchResult(
            documents=documents,
            search_time=time.time() - start_time,
            cache_hit=False,
            optimization_applied=optimizations
        )
    
    def optimize_text_query(
        self,
        query: str,
        preprocess: bool = True
    ) -> str:
        """テキストクエリの最適化"""
        optimized_query = query
        
        if preprocess:
            # 1. 小文字化と正規化
            optimized_query = optimized_query.lower().strip()
            
            # 2. ストップワードの除去（オプション）
            # optimized_query = remove_stopwords(optimized_query)
            
            # 3. 同義語展開
            optimized_query = self._expand_synonyms(optimized_query)
            
        return optimized_query
    
    def _expand_synonyms(self, query: str) -> str:
        """同義語展開"""
        # 道路設計関連の同義語辞書
        synonyms = {
            "道路": ["道路", "道", "路線"],
            "設計": ["設計", "計画", "デザイン"],
            "速度": ["速度", "スピード", "velocity"],
            "曲線": ["曲線", "カーブ", "曲がり"],
        }
        
        for word, syns in synonyms.items():
            if word in query:
                # 同義語をORで連結
                query = query.replace(word, f"({' OR '.join(syns)})")
        
        return query

class BatchProcessor:
    """バッチ処理最適化"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
    def process_embeddings_batch(
        self,
        texts: List[str],
        model,
        cache: RedisCache
    ) -> List[np.ndarray]:
        """埋め込みのバッチ処理"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 1. キャッシュチェック
        for i, text in enumerate(texts):
            cached_embedding = cache.get_embedding(text, model.name)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 2. キャッシュミスのテキストをバッチ処理
        if uncached_texts:
            # バッチごとに処理
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i+self.batch_size]
                batch_embeddings = model.encode(batch)
                
                # 結果を適切な位置に配置し、キャッシュに保存
                for j, embedding in enumerate(batch_embeddings):
                    idx = uncached_indices[i+j]
                    embeddings[idx] = embedding
                    cache.cache_embedding(
                        texts[idx], 
                        embedding.tolist(), 
                        model.name
                    )
        
        return embeddings
```

## 3. データベースクエリ最適化

```python
# app/optimization/db_optimizer.py
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import asyncpg
import asyncio

class DatabaseOptimizer:
    """データベースクエリ最適化"""
    
    def __init__(self, database_url: str):
        # コネクションプーリング設定
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,          # 常時接続数
            max_overflow=40,       # 最大追加接続数
            pool_timeout=30,       # タイムアウト
            pool_recycle=3600,     # 接続リサイクル時間
            pool_pre_ping=True     # 接続確認
        )
        
        # インデックス最適化フラグ
        self.indexes_optimized = False
        
    async def optimize_indexes(self):
        """インデックスの最適化"""
        if self.indexes_optimized:
            return
            
        optimization_queries = [
            # ベクトル検索用インデックス
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
            ON embeddings USING ivfflat (vector vector_l2_ops)
            WITH (lists = 100);
            """,
            
            # メタデータ検索用インデックス
            """
            CREATE INDEX IF NOT EXISTS idx_documents_metadata 
            ON documents USING gin (metadata);
            """,
            
            # タイムスタンプインデックス
            """
            CREATE INDEX IF NOT EXISTS idx_documents_created 
            ON documents (created_at DESC);
            """,
            
            # 複合インデックス
            """
            CREATE INDEX IF NOT EXISTS idx_documents_type_status 
            ON documents (document_type, status);
            """
        ]
        
        async with asyncpg.create_pool(self.database_url) as pool:
            async with pool.acquire() as conn:
                for query in optimization_queries:
                    await conn.execute(query)
        
        self.indexes_optimized = True
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """クエリパフォーマンスの分析"""
        with self.engine.connect() as conn:
            # EXPLAIN ANALYZEでクエリプランを取得
            result = conn.execute(text(f"EXPLAIN ANALYZE {query}"))
            
            plan = []
            for row in result:
                plan.append(row[0])
            
            # パフォーマンス指標を抽出
            execution_time = self._extract_execution_time(plan)
            planning_time = self._extract_planning_time(plan)
            
            return {
                "query": query,
                "execution_time": execution_time,
                "planning_time": planning_time,
                "plan": plan,
                "optimization_suggestions": self._suggest_optimizations(plan)
            }
    
    def _suggest_optimizations(self, plan: List[str]) -> List[str]:
        """最適化の提案"""
        suggestions = []
        
        # Seq Scanの検出
        if any("Seq Scan" in line for line in plan):
            suggestions.append("Sequential scan detected. Consider adding an index.")
        
        # 高コストの操作を検出
        if any("Sort" in line and "external" in line for line in plan):
            suggestions.append("External sort detected. Consider increasing work_mem.")
        
        # ネステッドループの検出
        if any("Nested Loop" in line for line in plan):
            suggestions.append("Nested loop detected. Consider using hash join for large datasets.")
        
        return suggestions
```

## 4. 統合実装

```python
# app/performance/integrated_optimizer.py
from typing import Dict, Any, Optional
import time
import asyncio

class IntegratedPerformanceOptimizer:
    """統合パフォーマンス最適化システム"""
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        database_url: Optional[str] = None
    ):
        # Redis キャッシュ
        self.cache = RedisCache(host=redis_host, port=redis_port)
        
        # クエリ最適化
        self.query_optimizer = QueryOptimizer(self.cache)
        
        # バッチ処理
        self.batch_processor = BatchProcessor()
        
        # DB最適化（オプション）
        self.db_optimizer = DatabaseOptimizer(database_url) if database_url else None
        
        # メトリクス
        self.performance_metrics = {}
        
    async def optimize_rag_query(
        self,
        query: str,
        use_cache: bool = True,
        optimize_db: bool = True
    ) -> Dict[str, Any]:
        """RAGクエリの完全最適化"""
        start_time = time.time()
        optimizations_applied = []
        
        # 1. クエリキャッシュチェック
        if use_cache:
            cached_result = self.cache.get_query_result(query)
            if cached_result:
                return {
                    "result": cached_result,
                    "cache_hit": True,
                    "response_time": time.time() - start_time,
                    "optimizations": ["query_cache"]
                }
        
        # 2. テキストクエリ最適化
        optimized_query = self.query_optimizer.optimize_text_query(query)
        optimizations_applied.append("text_optimization")
        
        # 3. 埋め込み生成（キャッシュ付き）
        embedding = await self._get_or_create_embedding(optimized_query)
        if embedding.get("cache_hit"):
            optimizations_applied.append("embedding_cache")
        
        # 4. ベクトル検索最適化
        search_result = self.query_optimizer.optimize_vector_search(
            embedding["vector"],
            top_k=10,
            use_cache=True
        )
        optimizations_applied.extend(search_result.optimization_applied)
        
        # 5. データベースクエリ最適化
        if optimize_db and self.db_optimizer:
            await self.db_optimizer.optimize_indexes()
            optimizations_applied.append("db_optimization")
        
        # 6. 結果の生成
        final_result = {
            "query": query,
            "documents": search_result.documents,
            "response_time": time.time() - start_time,
            "optimizations": optimizations_applied,
            "cache_hit": False
        }
        
        # 7. 結果をキャッシュ
        if use_cache:
            self.cache.cache_query_result(query, final_result, ttl=1800)
        
        # 8. メトリクス更新
        self._update_metrics(final_result)
        
        return final_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートの生成"""
        cache_stats = self.cache.get_stats()
        
        return {
            "cache_performance": {
                "hit_rate": cache_stats["hit_rate"],
                "total_hits": cache_stats["keyspace_hits"],
                "total_misses": cache_stats["keyspace_misses"],
                "memory_usage": cache_stats["used_memory"]
            },
            "query_performance": self.performance_metrics,
            "optimization_recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """最適化の推奨事項を生成"""
        recommendations = []
        
        cache_stats = self.cache.get_stats()
        
        # キャッシュヒット率が低い場合
        if cache_stats["hit_rate"] < 50:
            recommendations.append(
                "Cache hit rate is low. Consider increasing cache TTL or pre-warming cache."
            )
        
        # 平均応答時間が遅い場合
        if self.performance_metrics.get("avg_response_time", 0) > 1.0:
            recommendations.append(
                "Average response time is high. Consider adding more indexes or upgrading hardware."
            )
        
        return recommendations
```

## 実装による効果

### パフォーマンス改善の期待値

| メトリクス | 最適化前 | 最適化後 | 改善率 |
|-----------|---------|---------|--------|
| **クエリ応答時間** | 2000ms | 200ms | 90%削減 |
| **埋め込み生成** | 500ms | 50ms | 90%削減 |
| **ベクトル検索** | 800ms | 100ms | 87.5%削減 |
| **キャッシュヒット率** | 0% | 70% | +70% |
| **同時処理数** | 10 | 100 | 10倍 |
| **メモリ使用量** | 8GB | 4GB | 50%削減 |

### 主な最適化ポイント

1. **多層キャッシング**: クエリ、埋め込み、検索結果の各レベルでキャッシュ
2. **バッチ処理**: 複数のリクエストを効率的に処理
3. **インデックス最適化**: データベースクエリの高速化
4. **接続プーリング**: データベース接続の再利用
5. **非同期処理**: I/O待機時間の削減
