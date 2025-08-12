#!/usr/bin/env python3
"""
文書インデックス作成スクリプト
道路設計文書を処理してベクトルストアに登録
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from loguru import logger
import uuid

# Matplotlib用の環境変数を設定
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.document_processing.document_processor import RoadDesignDocumentProcessor
from src.rag.indexing.vector_store import QdrantVectorStore
from src.rag.indexing.embedding_model import EmbeddingModelFactory
from src.rag.indexing.metadata_manager import MetadataManager, DocumentMetadata, DocumentType, DocumentStatus
from src.rag.config.rag_config import load_config


def setup_logging(log_level: str = "INFO"):
    """ログ設定"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # ファイルログも追加
    log_file = Path("/workspace/logs") / "indexing.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)
    logger.add(
        log_file,
        level=log_level,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def load_document_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """文書設定ファイルを読み込み"""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # デフォルト設定
    return {
        "document_types": {
            "道路構造令": {
                "type": "road_standard",
                "category": "構造基準",
                "keywords": ["道路構造", "幅員", "勾配", "曲線"]
            },
            "設計指針": {
                "type": "design_guide", 
                "category": "設計指針",
                "keywords": ["設計", "計画", "基準"]
            },
            "技術基準": {
                "type": "technical_manual",
                "category": "技術基準",
                "keywords": ["技術", "仕様", "基準"]
            }
        },
        "default_metadata": {
            "publisher": "国土交通省",
            "applicable_standards": ["道路構造令", "道路設計基準"]
        }
    }


def extract_document_metadata(pdf_path: Path, config: Dict[str, Any]) -> DocumentMetadata:
    """ファイル名や内容から文書メタデータを推定"""
    
    filename = pdf_path.name
    
    # ファイル名から文書タイプを推定
    document_type = DocumentType.OTHER
    category = "その他"
    keywords = []
    
    for doc_name, doc_config in config.get("document_types", {}).items():
        if doc_name in filename:
            document_type = DocumentType(doc_config["type"])
            category = doc_config["category"]
            keywords = doc_config["keywords"]
            break
            
    # ファイルハッシュを計算
    import hashlib
    hash_md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()
    
    # メタデータオブジェクトを作成
    metadata = DocumentMetadata(
        id=str(uuid.uuid4()),
        title=filename.replace('.pdf', ''),
        filename=filename,
        file_path=str(pdf_path),
        file_hash=file_hash,
        document_type=document_type,
        category=category,
        keywords=keywords,
        status=DocumentStatus.ACTIVE,
        **config.get("default_metadata", {})
    )
    
    return metadata


def index_single_document(pdf_path: Path,
                         processor: RoadDesignDocumentProcessor,
                         vector_store: QdrantVectorStore,
                         embedding_model,
                         metadata_manager: MetadataManager,
                         config: Dict[str, Any]) -> bool:
    """単一文書をインデックス化"""
    
    try:
        logger.info(f"Processing document: {pdf_path}")
        
        # 1. 文書メタデータを作成
        doc_metadata = extract_document_metadata(pdf_path, config)
        
        # 2. 文書を処理
        processed_doc = processor.process_document(
            str(pdf_path), 
            document_metadata=doc_metadata.to_dict()
        )
        
        # 処理結果の確認
        if processed_doc is None:
            logger.error(f"Document processing failed for {pdf_path}")
            return False
        
        # デバッグ情報を出力
        logger.info(f"Document processed - Sections: {len(processed_doc.sections)}, Tables: {len(processed_doc.tables)}, Chunks: {len(processed_doc.chunks)}")
        
        # 3. メタデータマネージャーに登録
        if processed_doc.stats and hasattr(processed_doc.stats, 'get'):
            # text_statsの安全な取得
            text_stats = processed_doc.stats.get('text_stats', {})
            structure_stats = processed_doc.stats.get('structure_stats', {})
            
            doc_metadata.page_count = text_stats.get('estimated_pages', 0)
            doc_metadata.section_count = structure_stats.get('sections_count', 0)
            doc_metadata.table_count = structure_stats.get('tables_count', 0)
            doc_metadata.figure_count = structure_stats.get('figures_count', 0)
        else:
            # statsが利用できない場合のデフォルト値
            logger.warning(f"Document stats not available for {pdf_path}, using default values")
            doc_metadata.page_count = 0
            doc_metadata.section_count = 0
            doc_metadata.table_count = 0
            doc_metadata.figure_count = 0
        
        metadata_manager.add_document(doc_metadata)
        
        # 4. 埋め込み生成とベクトルストア登録
        if processed_doc.chunks and len(processed_doc.chunks) > 0:
            logger.info(f"Generating embeddings for {len(processed_doc.chunks)} chunks...")
            
            texts = [chunk.text for chunk in processed_doc.chunks]
            chunk_ids = [chunk.id for chunk in processed_doc.chunks]
            metadatas = [chunk.metadata for chunk in processed_doc.chunks]
            
            # 埋め込み生成
            embeddings = embedding_model.encode(texts, batch_size=16, show_progress=True)
            
            # ベクトルストレージに追加
            try:
                vector_store.add_documents(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
                logger.info(f"Successfully added {len(texts)} vectors to store")
                
                # 追加後にベクトル数を確認
                try:
                    stats = vector_store.get_collection_info()
                    logger.info(f"Vector store now contains {stats.get('vectors_count', 'unknown')} vectors")
                except Exception as stats_error:
                    logger.warning(f"Could not get vector stats: {stats_error}")
                
                logger.info(f"Successfully indexed document: {pdf_path}")
                return True
            except Exception as vector_error:
                logger.error(f"Failed to add vectors to store: {vector_error}")
                import traceback
                logger.error(f"Vector store error traceback: {traceback.format_exc()}")
                return False
        else:
            logger.warning(f"No chunks generated for document: {pdf_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to index document {pdf_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def get_shared_qdrant_client():
    """メインアプリケーションのQdrantクライアントを取得"""
    try:
        # メインアプリケーションのRAGシステムからクライアントを取得
        import sys
        import asyncio
        sys.path.insert(0, '/workspace')
        
        # メインアプリケーションのRAGシステムをインポート
        from app.main_unified import rag_app
        
        # RAGシステムが初期化されていない場合は初期化を試行
        if not rag_app.is_initialized:
            logger.info("RAG system not initialized, attempting to initialize...")
            try:
                # 非同期初期化を同期的に実行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(rag_app.initialize())
                loop.close()
            except Exception as init_error:
                logger.warning(f"Failed to initialize RAG system: {init_error}")
                return None
        
        if rag_app.is_initialized and rag_app.query_engine:
            logger.info("Using shared Qdrant client from initialized RAG system")
            return rag_app.query_engine.vector_store.client
        else:
            logger.warning("RAG system not properly initialized, using standalone client")
            return None
    except Exception as e:
        logger.warning(f"Failed to get shared Qdrant client: {e}")
        return None

def index_documents(input_paths: List[str],
                   config_file: Optional[str] = None,
                   output_dir: str = "./outputs/rag_index",
                   embedding_model_type: str = "multilingual-e5-large",
                   vector_store_path: str = "./qdrant_data",
                   metadata_db_path: str = "./metadata/metadata.db",
                   batch_size: int = 10,
                   force_reindex: bool = False) -> Dict[str, Any]:
    """複数文書をインデックス化"""
    
    # 出力ディレクトリの作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定読み込み
    config = load_document_config(config_file)
    
    # コンポーネントの初期化
    logger.info("Initializing components...")
    
    processor = RoadDesignDocumentProcessor(
        output_dir=str(output_dir / "processed_documents")
    )
    
    embedding_model = EmbeddingModelFactory.create(embedding_model_type)
    embedding_dim = EmbeddingModelFactory.get_embedding_dim(embedding_model_type)
    
    # 共有Qdrantクライアントを取得
    shared_client = get_shared_qdrant_client()
    
    if shared_client:
        # 共有クライアントがある場合はそれを使用
        logger.info("Using shared Qdrant client from main application")
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=embedding_dim,
            client=shared_client
        )
    else:
        # 共有クライアントがない場合は別のパスでスタンドアロンクライアントを作成
        import time
        indexing_path = f"./qdrant_indexing_{int(time.time())}"
        logger.info(f"Creating standalone Qdrant client at: {indexing_path}")
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim=embedding_dim,
            path=indexing_path
        )
    
    metadata_manager = MetadataManager(db_path=metadata_db_path)
    
    # 処理対象ファイルの収集
    pdf_files = []
    for path_str in input_paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files.append(path)
        elif path.is_dir():
            pdf_files.extend(path.rglob('*.pdf'))
            
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # バッチ処理
    results = {
        'total_files': len(pdf_files),
        'processed_files': 0,
        'failed_files': 0,
        'skipped_files': 0,
        'processing_errors': []
    }
    
    for i, pdf_file in enumerate(pdf_files):
        logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")
        
        # 既存チェック（force_reindexがFalseの場合）
        if not force_reindex:
            # ファイルハッシュで既存文書をチェック
            import hashlib
            hash_md5 = hashlib.md5()
            with open(pdf_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            file_hash = hash_md5.hexdigest()
            
            # 既存文書を検索
            existing_docs = metadata_manager.search_documents()
            if any(doc.file_hash == file_hash for doc in existing_docs):
                logger.info(f"Document already indexed, skipping: {pdf_file}")
                results['skipped_files'] += 1
                continue
        
        # インデックス化実行
        success = index_single_document(
            pdf_file, processor, vector_store, embedding_model,
            metadata_manager, config
        )
        
        if success:
            results['processed_files'] += 1
        else:
            results['failed_files'] += 1
            results['processing_errors'].append(str(pdf_file))
            
        # バッチ処理の区切り
        if (i + 1) % batch_size == 0:
            logger.info(f"Processed {i+1}/{len(pdf_files)} files...")
            
    # 結果統計の出力
    logger.info("Indexing completed!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Processed: {results['processed_files']}")
    logger.info(f"Failed: {results['failed_files']}")
    logger.info(f"Skipped: {results['skipped_files']}")
    
    if results['processing_errors']:
        logger.warning(f"Failed files: {results['processing_errors']}")
        
    # ベクトルストア統計
    try:
        vector_stats = vector_store.get_collection_info()
        logger.info(f"Vector store stats: {vector_stats}")
    except Exception as e:
        logger.warning(f"Failed to get vector store stats: {e}")
        vector_stats = {"error": str(e), "vectors_count": "unknown"}
    
    # メタデータ統計
    metadata_stats = metadata_manager.get_statistics()
    logger.info(f"Metadata stats: {metadata_stats}")
    
    # JSONシリアライズ可能な形式に変換
    def convert_to_serializable(obj, visited=None):
        if visited is None:
            visited = set()
        
        # 循環参照チェック
        obj_id = id(obj)
        if obj_id in visited:
            return str(obj)  # 循環参照の場合は文字列として返す
        
        # 基本型はそのまま返す
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
            
        visited.add(obj_id)
        
        try:
            # pandas DataFrameの処理
            if hasattr(obj, 'to_dict') and hasattr(obj, 'shape'):  # pandas DataFrame
                return {
                    'type': 'DataFrame',
                    'shape': obj.shape,
                    'columns': list(obj.columns) if hasattr(obj, 'columns') else [],
                    'data_preview': str(obj.head()) if len(obj) > 0 else 'Empty DataFrame'
                }
            # numpy配列の処理
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: convert_to_serializable(v, visited) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item, visited) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v, visited) for k, v in obj.items()}
            elif hasattr(obj, 'name'):  # Enumの場合
                return obj.name
            elif hasattr(obj, 'value'):  # Enumの場合
                return obj.value
            else:
                return str(obj)
        finally:
            visited.discard(obj_id)
    
    # 結果をファイルに保存
    results_file = output_dir / "indexing_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'vector_stats': convert_to_serializable(vector_stats),
            'metadata_stats': metadata_stats,
            'config': config
        }, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Results saved to: {results_file}")
    
    return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="道路設計文書のインデックス作成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイルをインデックス化
  python index_documents.py document.pdf
  
  # ディレクトリ内の全PDFをインデックス化
  python index_documents.py /path/to/documents/
  
  # 複数のパスを指定
  python index_documents.py doc1.pdf doc2.pdf /path/to/dir/
  
  # 設定ファイルを指定
  python index_documents.py documents/ --config config.json
  
  # 再インデックス化を強制
  python index_documents.py documents/ --force-reindex
        """
    )
    
    parser.add_argument(
        'input_paths',
        nargs='+',
        help='処理するPDFファイルまたはディレクトリのパス'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='文書設定ファイルのパス（JSON形式）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/rag_index',
        help='出力ディレクトリ（デフォルト: ./outputs/rag_index）'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='multilingual-e5-large',
        choices=['multilingual-e5-large', 'multilingual-e5-base', 'sentence-bert-ja'],
        help='使用する埋め込みモデル（デフォルト: multilingual-e5-large）'
    )
    
    parser.add_argument(
        '--vector-store-path',
        type=str,
        default='./qdrant_data',
        help='ベクトルストレージのパス（デフォルト: ./qdrant_data）'
    )
    
    parser.add_argument(
        '--metadata-db-path',
        type=str,
        default='./metadata/metadata.db',
        help='メタデータDBのパス（デフォルト: ./metadata/metadata.db）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='バッチサイズ（デフォルト: 10）'
    )
    
    parser.add_argument(
        '--force-reindex',
        action='store_true',
        help='既存文書の再インデックス化を強制'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='ログレベル（デフォルト: INFO）'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    
    logger.info("Starting document indexing...")
    logger.info(f"Input paths: {args.input_paths}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    
    try:
        results = index_documents(
            input_paths=args.input_paths,
            config_file=args.config,
            output_dir=args.output_dir,
            embedding_model_type=args.embedding_model,
            vector_store_path=args.vector_store_path,
            metadata_db_path=args.metadata_db_path,
            batch_size=args.batch_size,
            force_reindex=args.force_reindex
        )
        
        # 終了コードの決定
        if results['failed_files'] > 0:
            logger.warning("Some files failed to process")
            sys.exit(1)
        elif results['processed_files'] == 0:
            logger.warning("No files were processed")
            sys.exit(1)
        else:
            logger.info("All files processed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()