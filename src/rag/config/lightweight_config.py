"""
軽量化RAG設定パッチ
Docker環境用に埋め込みモデルを軽量化
"""

import os
import logging

logger = logging.getLogger(__name__)

def apply_lightweight_rag_config():
    """RAGシステムを軽量設定に変更"""
    
    # 環境変数で軽量設定を強制
    lightweight_config = {
        # 軽量な埋め込みモデルを使用
        "RAG_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "RAG_EMBEDDING_DIMENSION": "384",
        
        # CPUで実行
        "RAG_USE_CPU": "true",
        "RAG_DEVICE": "cpu",
        
        # バッチサイズを小さく
        "RAG_BATCH_SIZE": "8",
        
        # キャッシュを有効化
        "RAG_USE_CACHE": "true",
        "RAG_CACHE_DIR": "/workspace/.cache/embeddings",
        
        # 自動初期化を無効化
        "RAG_DISABLE_AUTO_INIT": "true",
        "RAG_LAZY_LOAD": "true",
        
        # メモリ制限
        "RAG_MAX_MEMORY_GB": "2",
        
        # タイムアウト設定
        "RAG_TIMEOUT_SECONDS": "30"
    }
    
    for key, value in lightweight_config.items():
        if key not in os.environ:
            os.environ[key] = value
    
    logger.info("Lightweight RAG configuration applied")
    
    # 既存のRAG設定ファイルをオーバーライド
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("/workspace/src/rag/config/rag_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 軽量設定に変更
            if 'embedding' in config:
                config['embedding']['model_name'] = 'sentence-transformers/all-MiniLM-L6-v2'
                config['embedding']['dimension'] = 384
                config['embedding']['device'] = 'cpu'
                config['embedding']['batch_size'] = 8
            
            if 'search' in config:
                config['search']['top_k'] = 5  # 検索結果を減らす
            
            if 'reranking' in config:
                config['reranking']['enabled'] = False  # リランキングを無効化
            
            # 一時ファイルに保存
            temp_config = config_path.with_suffix('.yaml.lightweight')
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            # 元のファイルを置き換え
            temp_config.replace(config_path)
            logger.info("RAG config file updated with lightweight settings")
            
    except Exception as e:
        logger.warning(f"Could not update RAG config file: {e}")

# Docker環境で自動適用
if os.environ.get("DOCKER_CONTAINER") == "true":
    apply_lightweight_rag_config()
