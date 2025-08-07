"""
設定管理モジュール（リファクタリング版）
型ヒントと検証機能を強化
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
import yaml

from ..utils import (
    ConfigurationError,
    setup_logger,
    validate_config
)

logger = setup_logger(__name__)


class EmbeddingConfig(TypedDict, total=False):
    """埋め込み設定の型定義"""
    model_name: str
    dimension: int
    batch_size: int
    max_length: int
    device: str


class SearchConfig(TypedDict, total=False):
    """検索設定の型定義"""
    vector_weight: float
    keyword_weight: float
    top_k: int
    similarity_threshold: float
    enable_reranking: bool


class GenerationConfig(TypedDict, total=False):
    """生成設定の型定義"""
    model_name: str
    max_length: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


class VectorStoreConfig(TypedDict, total=False):
    """ベクトルストア設定の型定義"""
    type: str
    host: str
    port: int
    collection_name: str
    distance_metric: str


class DocumentConfig(TypedDict, total=False):
    """ドキュメント処理設定の型定義"""
    chunk_size: int
    chunk_overlap: int
    max_chunks: int
    supported_formats: List[str]
    enable_ocr: bool
    enable_table_extraction: bool


class ConfigManager:
    """設定管理クラス"""
    
    # デフォルト設定
    DEFAULT_CONFIG: Dict[str, Any] = {
        'embedding': {
            'model_name': 'intfloat/multilingual-e5-large',
            'dimension': 1024,
            'batch_size': 32,
            'max_length': 512,
            'device': 'cuda'
        },
        'search': {
            'vector_weight': 0.7,
            'keyword_weight': 0.3,
            'top_k': 10,
            'similarity_threshold': 0.5,
            'enable_reranking': True
        },
        'generation': {
            'model_name': 'cyberagent/calm2-7b-chat',
            'max_length': 1024,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1
        },
        'vector_store': {
            'type': 'qdrant',
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'road_design_docs',
            'distance_metric': 'cosine'
        },
        'document': {
            'chunk_size': 512,
            'chunk_overlap': 128,
            'max_chunks': 10000,
            'supported_formats': ['.pdf', '.txt', '.md', '.docx'],
            'enable_ocr': True,
            'enable_table_extraction': True
        }
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        use_env_vars: bool = True
    ):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            config_dict: 設定辞書
            use_env_vars: 環境変数を使用するか
        """
        self.config: Dict[str, Any] = self.DEFAULT_CONFIG.copy()
        
        # 設定の読み込み優先順位：
        # 1. config_dict
        # 2. config_path
        # 3. 環境変数
        # 4. デフォルト値
        
        if config_path:
            self._load_from_file(config_path)
        
        if config_dict:
            self._merge_config(config_dict)
        
        if use_env_vars:
            self._load_from_env()
        
        # 設定の検証
        self._validate_config()
    
    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """ファイルから設定を読み込み"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                    file_config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    file_config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {config_path.suffix}"
                    )
            
            self._merge_config(file_config)
            logger.info(f"Loaded config from {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {e}")
    
    def _load_from_env(self) -> None:
        """環境変数から設定を読み込み"""
        env_mappings = {
            'RAG_EMBEDDING_MODEL': ('embedding', 'model_name'),
            'RAG_GENERATION_MODEL': ('generation', 'model_name'),
            'RAG_VECTOR_STORE_HOST': ('vector_store', 'host'),
            'RAG_VECTOR_STORE_PORT': ('vector_store', 'port'),
            'RAG_CHUNK_SIZE': ('document', 'chunk_size'),
            'RAG_MAX_LENGTH': ('generation', 'max_length'),
            'RAG_TEMPERATURE': ('generation', 'temperature'),
            'HF_TOKEN': ('auth', 'huggingface_token'),
            'OPENAI_API_KEY': ('auth', 'openai_api_key')
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                # 型変換
                if key in ['port', 'chunk_size', 'max_length']:
                    value = int(value)
                elif key in ['temperature', 'top_p', 'vector_weight']:
                    value = float(value)
                
                # 設定に反映
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = value
                
                logger.debug(f"Loaded {section}.{key} from {env_var}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """設定をマージ"""
        def deep_merge(base: Dict, update: Dict) -> Dict:
            """再帰的な辞書マージ"""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        self.config = deep_merge(self.config, new_config)
    
    def _validate_config(self) -> None:
        """設定の検証"""
        # 必須フィールドの確認
        required_sections = ['embedding', 'search', 'generation', 'vector_store', 'document']
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required config section: {section}")
        
        # 値の範囲チェック
        validations = [
            ('search.vector_weight', 0.0, 1.0),
            ('search.keyword_weight', 0.0, 1.0),
            ('generation.temperature', 0.0, 2.0),
            ('generation.top_p', 0.0, 1.0),
            ('document.chunk_overlap', 0, None)
        ]
        
        for path, min_val, max_val in validations:
            value = self.get(path)
            if value is not None:
                if min_val is not None and value < min_val:
                    raise ConfigurationError(
                        f"Config {path} = {value} is below minimum {min_val}"
                    )
                if max_val is not None and value > max_val:
                    raise ConfigurationError(
                        f"Config {path} = {value} is above maximum {max_val}"
                    )
        
        # 重みの合計チェック
        vector_weight = self.get('search.vector_weight', 0)
        keyword_weight = self.get('search.keyword_weight', 0)
        total_weight = vector_weight + keyword_weight
        
        if abs(total_weight - 1.0) > 0.01:  # 浮動小数点誤差を考慮
            logger.warning(
                f"Search weights sum to {total_weight}, normalizing to 1.0"
            )
            self.config['search']['vector_weight'] = vector_weight / total_weight
            self.config['search']['keyword_weight'] = keyword_weight / total_weight
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        ドット記法で設定値を取得
        
        Args:
            key: 設定キー（例: 'embedding.model_name'）
            default: デフォルト値
        
        Returns:
            設定値
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        ドット記法で設定値を設定
        
        Args:
            key: 設定キー
            value: 設定値
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """埋め込み設定を取得"""
        return self.config.get('embedding', {})
    
    def get_search_config(self) -> SearchConfig:
        """検索設定を取得"""
        return self.config.get('search', {})
    
    def get_generation_config(self) -> GenerationConfig:
        """生成設定を取得"""
        return self.config.get('generation', {})
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """ベクトルストア設定を取得"""
        return self.config.get('vector_store', {})
    
    def get_document_config(self) -> DocumentConfig:
        """ドキュメント処理設定を取得"""
        return self.config.get('document', {})
    
    def save(
        self,
        path: Union[str, Path],
        format: str = 'yaml'
    ) -> None:
        """
        設定をファイルに保存
        
        Args:
            path: 保存先パス
            format: 保存形式（'yaml' or 'json'）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            if format == 'yaml':
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True
                )
            elif format == 'json':
                json.dump(
                    self.config,
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
        
        logger.info(f"Config saved to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書として取得"""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """文字列表現"""
        return f"ConfigManager(sections={list(self.config.keys())})"


# シングルトンインスタンス
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    config_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> ConfigManager:
    """
    設定マネージャーのシングルトンインスタンスを取得
    
    Args:
        config_path: 設定ファイルパス
        force_reload: 強制的に再読み込みするか
    
    Returns:
        設定マネージャー
    """
    global _global_config_manager
    
    if _global_config_manager is None or force_reload:
        _global_config_manager = ConfigManager(config_path)
    
    return _global_config_manager