"""
RAG設定管理モジュール
YAML設定ファイルの読み込みと管理
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """埋め込みモデル設定"""
    model_name: str = "intfloat/multilingual-e5-large"
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    """ベクトルストア設定"""
    type: str = "qdrant"
    path: str = "./qdrant_data"
    url: Optional[str] = None
    collection_name: str = "road_design_docs"
    prefer_grpc: bool = True
    timeout: int = 60


@dataclass
class LLMConfig:
    """LLM設定"""
    use_finetuned: bool = True
    finetuned_model_path: str = "./outputs/latest"
    base_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    device_map: str = "auto"
    max_memory: Dict[int, str] = None
    load_in_8bit: bool = True
    torch_dtype: str = "float16"
    temperature: float = 0.1
    max_new_tokens: int = 2048
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    # MoE統合設定
    use_moe: bool = False
    moe_model_path: Optional[str] = None
    moe_num_experts: int = 8
    moe_experts_per_token: int = 2
    
    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {0: "36GB", 1: "36GB"}


@dataclass
class DocumentProcessingConfig:
    """文書処理設定"""
    extract_images: bool = True
    extract_tables: bool = True
    dpi: int = 300
    languages: list = None
    gpu: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["ja", "en"]


@dataclass
class ChunkingConfig:
    """チャンキング設定"""
    strategy: str = "semantic"
    chunk_size: int = 512
    overlap: int = 128
    min_chunk_size: int = 100
    threshold: float = 0.7


@dataclass
class RetrievalConfig:
    """検索設定"""
    hybrid_search_enabled: bool = True
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 10
    rerank_top_k: int = 5
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@dataclass
class RAGConfig:
    """RAG全体設定"""
    system_name: str = "土木道路設計特化型RAGシステム"
    version: str = "1.0.0"
    language: str = "ja"
    log_level: str = "INFO"
    
    embedding: EmbeddingConfig = None
    vector_store: VectorStoreConfig = None
    llm: LLMConfig = None
    document_processing: DocumentProcessingConfig = None
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.document_processing is None:
            self.document_processing = DocumentProcessingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()


def load_config(config_path: Optional[str] = None, resolve_model_paths: bool = True) -> RAGConfig:
    """設定ファイルを読み込み"""
    
    if config_path is None:
        # デフォルト設定ファイルのパス
        config_path = Path(__file__).parent / "rag_config.yaml"
        
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return RAGConfig()
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        # 設定データからオブジェクトを構築
        config = RAGConfig()
        
        # システム設定
        if 'system' in config_data:
            system_config = config_data['system']
            config.system_name = system_config.get('name', config.system_name)
            config.version = system_config.get('version', config.version)
            config.language = system_config.get('language', config.language)
            config.log_level = system_config.get('log_level', config.log_level)
            
        # 埋め込み設定
        if 'embedding' in config_data:
            embedding_config = config_data['embedding']
            config.embedding = EmbeddingConfig(
                model_name=embedding_config.get('model_name', config.embedding.model_name),
                device=embedding_config.get('device', config.embedding.device),
                batch_size=embedding_config.get('batch_size', config.embedding.batch_size),
                max_length=embedding_config.get('max_length', config.embedding.max_length),
                normalize_embeddings=embedding_config.get('normalize_embeddings', config.embedding.normalize_embeddings)
            )
            
        # ベクトルストア設定
        if 'vector_store' in config_data:
            vs_config = config_data['vector_store']
            if 'qdrant' in vs_config:
                qdrant_config = vs_config['qdrant']
                config.vector_store = VectorStoreConfig(
                    type=vs_config.get('type', config.vector_store.type),
                    path=qdrant_config.get('path', config.vector_store.path),
                    url=qdrant_config.get('url', config.vector_store.url),
                    collection_name=qdrant_config.get('collection_name', config.vector_store.collection_name),
                    prefer_grpc=qdrant_config.get('prefer_grpc', config.vector_store.prefer_grpc),
                    timeout=qdrant_config.get('timeout', config.vector_store.timeout)
                )
                
        # LLM設定
        if 'llm' in config_data:
            llm_config = config_data['llm']
            config.llm = LLMConfig(
                use_finetuned=llm_config.get('use_finetuned', config.llm.use_finetuned),
                finetuned_model_path=llm_config.get('finetuned_model_path', config.llm.finetuned_model_path),
                base_model=llm_config.get('base_model', config.llm.base_model),
                device_map=llm_config.get('device_map', config.llm.device_map),
                max_memory=llm_config.get('max_memory', config.llm.max_memory),
                load_in_8bit=llm_config.get('load_in_8bit', config.llm.load_in_8bit),
                torch_dtype=llm_config.get('torch_dtype', config.llm.torch_dtype),
                temperature=llm_config.get('temperature', config.llm.temperature),
                max_new_tokens=llm_config.get('max_new_tokens', config.llm.max_new_tokens),
                top_p=llm_config.get('top_p', config.llm.top_p),
                repetition_penalty=llm_config.get('repetition_penalty', config.llm.repetition_penalty),
                # MoE設定
                use_moe=llm_config.get('use_moe', config.llm.use_moe),
                moe_model_path=llm_config.get('moe_model_path', config.llm.moe_model_path),
                moe_num_experts=llm_config.get('moe_num_experts', config.llm.moe_num_experts),
                moe_experts_per_token=llm_config.get('moe_experts_per_token', config.llm.moe_experts_per_token)
            )
            
        # 文書処理設定
        if 'document_processing' in config_data:
            doc_config = config_data['document_processing']
            
            # PDF設定
            pdf_config = doc_config.get('pdf_parser', {})
            ocr_config = doc_config.get('ocr', {})
            
            config.document_processing = DocumentProcessingConfig(
                extract_images=pdf_config.get('extract_images', config.document_processing.extract_images),
                extract_tables=pdf_config.get('extract_tables', config.document_processing.extract_tables),
                dpi=pdf_config.get('dpi', config.document_processing.dpi),
                languages=ocr_config.get('languages', config.document_processing.languages),
                gpu=ocr_config.get('gpu', config.document_processing.gpu)
            )
            
        # チャンキング設定
        if 'document_processing' in config_data and 'chunking' in config_data['document_processing']:
            chunk_config = config_data['document_processing']['chunking']
            
            semantic_config = chunk_config.get('semantic', {})
            
            config.chunking = ChunkingConfig(
                strategy=chunk_config.get('strategy', config.chunking.strategy),
                chunk_size=chunk_config.get('chunk_size', config.chunking.chunk_size),
                overlap=chunk_config.get('overlap', config.chunking.overlap),
                min_chunk_size=chunk_config.get('min_chunk_size', config.chunking.min_chunk_size),
                threshold=semantic_config.get('threshold', config.chunking.threshold)
            )
            
        # 検索設定
        if 'retrieval' in config_data:
            retrieval_config = config_data['retrieval']
            
            hybrid_config = retrieval_config.get('hybrid_search', {})
            rerank_config = retrieval_config.get('reranking', {})
            
            config.retrieval = RetrievalConfig(
                hybrid_search_enabled=hybrid_config.get('enabled', config.retrieval.hybrid_search_enabled),
                vector_weight=hybrid_config.get('vector_weight', config.retrieval.vector_weight),
                keyword_weight=hybrid_config.get('keyword_weight', config.retrieval.keyword_weight),
                top_k=retrieval_config.get('top_k', config.retrieval.top_k),
                rerank_top_k=retrieval_config.get('rerank_top_k', config.retrieval.rerank_top_k),
                reranking_enabled=rerank_config.get('enabled', config.retrieval.reranking_enabled),
                reranking_model=rerank_config.get('model', config.retrieval.reranking_model)
            )
            
        # モデルパスの解決と設定検証
        if resolve_model_paths:
            try:
                from .config_validator import validate_config
                
                # 設定を検証し、自動修正
                issues, fixed_count = validate_config(config, auto_fix=True)
                
                if fixed_count > 0:
                    logger.info(f"Auto-fixed {fixed_count} configuration issues")
                    
                # エラーがある場合は警告
                errors = [i for i in issues if i.level == "error"]
                if errors:
                    logger.warning(f"Configuration has {len(errors)} unresolved errors")
                    for error in errors:
                        logger.warning(f"  - [{error.component}] {error.message}")
                    
            except Exception as e:
                logger.warning(f"Configuration validation failed: {e}")
                
        # 追加のモデルパス解決
        if resolve_model_paths and config.llm.use_finetuned:
            try:
                from .model_path_resolver import resolve_model_path
                
                resolved_path = resolve_model_path(
                    config.llm.finetuned_model_path,
                    preferred_type=None,  # 最新のモデルを自動選択
                    base_output_dir="./outputs" 
                )
                
                if resolved_path != config.llm.finetuned_model_path:
                    logger.info(f"Model path resolved: {config.llm.finetuned_model_path} -> {resolved_path}")
                    config.llm.finetuned_model_path = resolved_path
                    
            except Exception as e:
                logger.warning(f"Model path resolution failed: {e}")
                logger.info("Falling back to base model")
                config.llm.use_finetuned = False
                
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return RAGConfig()


def save_config(config: RAGConfig, config_path: str):
    """設定をファイルに保存"""
    
    config_data = {
        'system': {
            'name': config.system_name,
            'version': config.version,
            'language': config.language,
            'log_level': config.log_level
        },
        'embedding': {
            'model_name': config.embedding.model_name,
            'device': config.embedding.device,
            'batch_size': config.embedding.batch_size,
            'max_length': config.embedding.max_length,
            'normalize_embeddings': config.embedding.normalize_embeddings
        },
        'vector_store': {
            'type': config.vector_store.type,
            'qdrant': {
                'path': config.vector_store.path,
                'url': config.vector_store.url,
                'collection_name': config.vector_store.collection_name,
                'prefer_grpc': config.vector_store.prefer_grpc,
                'timeout': config.vector_store.timeout
            }
        },
        'llm': {
            'use_finetuned': config.llm.use_finetuned,
            'finetuned_model_path': config.llm.finetuned_model_path,
            'base_model': config.llm.base_model,
            'device_map': config.llm.device_map,
            'max_memory': config.llm.max_memory,
            'load_in_8bit': config.llm.load_in_8bit,
            'torch_dtype': config.llm.torch_dtype,
            'temperature': config.llm.temperature,
            'max_new_tokens': config.llm.max_new_tokens,
            'top_p': config.llm.top_p,
            'repetition_penalty': config.llm.repetition_penalty,
            # MoE設定
            'use_moe': config.llm.use_moe,
            'moe_model_path': config.llm.moe_model_path,
            'moe_num_experts': config.llm.moe_num_experts,
            'moe_experts_per_token': config.llm.moe_experts_per_token
        },
        'document_processing': {
            'pdf_parser': {
                'extract_images': config.document_processing.extract_images,
                'extract_tables': config.document_processing.extract_tables,
                'dpi': config.document_processing.dpi
            },
            'ocr': {
                'languages': config.document_processing.languages,
                'gpu': config.document_processing.gpu
            },
            'chunking': {
                'strategy': config.chunking.strategy,
                'chunk_size': config.chunking.chunk_size,
                'overlap': config.chunking.overlap,
                'min_chunk_size': config.chunking.min_chunk_size,
                'semantic': {
                    'threshold': config.chunking.threshold
                }
            }
        },
        'retrieval': {
            'hybrid_search': {
                'enabled': config.retrieval.hybrid_search_enabled,
                'vector_weight': config.retrieval.vector_weight,
                'keyword_weight': config.retrieval.keyword_weight
            },
            'top_k': config.retrieval.top_k,
            'rerank_top_k': config.retrieval.rerank_top_k,
            'reranking': {
                'enabled': config.retrieval.reranking_enabled,
                'model': config.retrieval.reranking_model
            }
        }
    }
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
    logger.info(f"Configuration saved to: {config_path}")


# グローバル設定インスタンス
_global_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """グローバル設定インスタンスを取得"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: RAGConfig):
    """グローバル設定インスタンスを設定"""
    global _global_config
    _global_config = config


def reload_config(config_path: Optional[str] = None):
    """設定を再読み込み"""
    global _global_config
    _global_config = load_config(config_path)