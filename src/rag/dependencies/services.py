"""
RAGシステムのサービス定義とDIコンテナ設定

このモジュールは、RAGシステムで使用される
全てのサービスを定義し、DIコンテナに登録します。
"""

from typing import Optional, Any, Dict
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from .container import DIContainer, ServiceScopeEnum, injectable, inject
from .dependency_manager import RAGDependencyManager, DependencyCheckResult

logger = logging.getLogger(__name__)


# ===== インターフェース定義 =====

class IVectorStore(ABC):
    """ベクトルストアのインターフェース"""
    
    @abstractmethod
    async def add_documents(self, documents: list) -> None:
        """ドキュメントを追加"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list:
        """検索を実行"""
        pass
    
    @abstractmethod
    async def delete(self, ids: list) -> None:
        """ドキュメントを削除"""
        pass


class IEmbeddingModel(ABC):
    """埋め込みモデルのインターフェース"""
    
    @abstractmethod
    def encode(self, text: str) -> list:
        """テキストをエンコード"""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: list) -> list:
        """バッチエンコード"""
        pass


class IDocumentProcessor(ABC):
    """ドキュメント処理のインターフェース"""
    
    @abstractmethod
    async def process(self, file_path: Path) -> dict:
        """ドキュメントを処理"""
        pass


class IQueryEngine(ABC):
    """クエリエンジンのインターフェース"""
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> dict:
        """クエリを実行"""
        pass


# ===== 実装クラス =====

@injectable(ServiceScopeEnum.SINGLETON)
class ConfigurationService:
    """設定サービス"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/rag_config.yaml")
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """設定をロード"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """デフォルト設定を取得"""
        return {
            "embedding": {
                "model_name": "sentence-transformers/multilingual-e5-large",
                "device": "cuda",
                "batch_size": 32
            },
            "vector_store": {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "documents"
            },
            "search": {
                "top_k": 5,
                "rerank": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> dict:
        """全設定を取得"""
        return self._config.copy()


@injectable(ServiceScopeEnum.SINGLETON)
class DependencyCheckService:
    """依存関係チェックサービス"""
    
    def __init__(self):
        self.manager = RAGDependencyManager()
        self.last_check_result: Optional[DependencyCheckResult] = None
    
    def check_dependencies(self, use_cache: bool = True) -> DependencyCheckResult:
        """依存関係をチェック"""
        self.last_check_result = self.manager.check_all_dependencies(use_cache=use_cache)
        return self.last_check_result
    
    def get_report(self, format: str = "text") -> str:
        """レポートを取得"""
        return self.manager.get_dependency_report(format=format)
    
    def install_missing(self, level=None, dry_run=False) -> dict:
        """不足している依存関係をインストール"""
        return self.manager.install_missing_dependencies(level=level, dry_run=dry_run)
    
    def is_ready(self) -> bool:
        """システムが準備完了か確認"""
        if not self.last_check_result:
            self.check_dependencies()
        return self.last_check_result.can_run if self.last_check_result else False


class QdrantVectorStore(IVectorStore):
    """Qdrantベクトルストアの実装"""
    
    def __init__(self, config: ConfigurationService):
        self.config = config
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """初期化"""
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                host=self.config.get("vector_store.host", "localhost"),
                port=self.config.get("vector_store.port", 6333)
            )
            logger.info("Qdrant vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def add_documents(self, documents: list) -> None:
        """ドキュメントを追加"""
        # 実装
        pass
    
    async def search(self, query: str, top_k: int = 5) -> list:
        """検索を実行"""
        # 実装
        return []
    
    async def delete(self, ids: list) -> None:
        """ドキュメントを削除"""
        # 実装
        pass


class InMemoryVectorStore(IVectorStore):
    """インメモリベクトルストア（フォールバック）"""
    
    def __init__(self):
        self.documents = []
        logger.warning("Using in-memory vector store (no persistence)")
    
    async def add_documents(self, documents: list) -> None:
        """ドキュメントを追加"""
        self.documents.extend(documents)
    
    async def search(self, query: str, top_k: int = 5) -> list:
        """検索を実行"""
        # 簡易実装
        return self.documents[:top_k]
    
    async def delete(self, ids: list) -> None:
        """ドキュメントを削除"""
        # 簡易実装
        pass


class EmbeddingModelService(IEmbeddingModel):
    """埋め込みモデルサービス"""
    
    def __init__(self, config: ConfigurationService):
        self.config = config
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """初期化"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.get("embedding.model_name")
            device = self.config.get("embedding.device", "cpu")
            
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Embedding model {model_name} loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # フォールバックとして簡易実装
            self.model = None
    
    def encode(self, text: str) -> list:
        """テキストをエンコード"""
        if self.model:
            return self.model.encode(text).tolist()
        else:
            # フォールバック：ランダムベクトル
            import random
            return [random.random() for _ in range(384)]
    
    def encode_batch(self, texts: list) -> list:
        """バッチエンコード"""
        if self.model:
            return self.model.encode(texts).tolist()
        else:
            return [self.encode(text) for text in texts]


@inject(
    vector_store=IVectorStore,
    embedding_model=IEmbeddingModel
)
class RAGQueryEngine(IQueryEngine):
    """RAGクエリエンジン"""
    
    def __init__(self, vector_store: IVectorStore, embedding_model: IEmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        logger.info("RAG Query Engine initialized")
    
    async def query(self, query: str, **kwargs) -> dict:
        """クエリを実行"""
        # クエリをエンコード
        query_embedding = self.embedding_model.encode(query)
        
        # ベクトル検索
        results = await self.vector_store.search(query, top_k=kwargs.get('top_k', 5))
        
        # 結果を返す
        return {
            "query": query,
            "results": results,
            "answer": "Sample answer based on retrieved documents"
        }


# ===== DIコンテナの設定 =====

def configure_rag_services(container: DIContainer) -> None:
    """RAGサービスをDIコンテナに設定"""
    
    # 設定サービス
    container.register_singleton(ConfigurationService)
    
    # 依存関係チェックサービス
    container.register_singleton(DependencyCheckService)
    
    # ベクトルストア（条件付き登録）
    def create_vector_store(config: ConfigurationService, dep_check: DependencyCheckService) -> IVectorStore:
        """ベクトルストアを作成"""
        # Qdrantが利用可能か確認
        if dep_check.is_ready():
            try:
                return QdrantVectorStore(config)
            except Exception as e:
                logger.warning(f"Failed to create Qdrant store: {e}")
        
        # フォールバック
        return InMemoryVectorStore()
    
    container.register(
        IVectorStore,
        lambda: create_vector_store(
            container.resolve(ConfigurationService),
            container.resolve(DependencyCheckService)
        ),
        ServiceScopeEnum.SINGLETON
    )
    
    # 埋め込みモデル
    container.register(
        IEmbeddingModel,
        lambda: EmbeddingModelService(container.resolve(ConfigurationService)),
        ServiceScopeEnum.SINGLETON
    )
    
    # クエリエンジン
    container.register(
        IQueryEngine,
        RAGQueryEngine,
        ServiceScopeEnum.SINGLETON,
        dependencies={
            "vector_store": IVectorStore,
            "embedding_model": IEmbeddingModel
        }
    )
    
    logger.info("RAG services configured in DI container")


def create_rag_container() -> DIContainer:
    """設定済みのRAGコンテナを作成"""
    container = DIContainer()
    configure_rag_services(container)
    return container
