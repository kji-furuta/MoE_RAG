"""
RAGシステム用Webインターフェース
既存のmain_unified.pyと統合したRAG API
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.core.query_engine import RoadDesignQueryEngine, QueryResult
from src.rag.indexing.metadata_manager import MetadataManager


# Pydanticモデル
class QueryRequest(BaseModel):
    """クエリリクエスト"""
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(5, description="取得する結果数", ge=1, le=20)
    search_type: str = Field("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
    include_sources: bool = Field(True, description="ソース情報を含めるか")
    filters: Optional[Dict[str, Any]] = Field(None, description="検索フィルター")


class QueryResponse(BaseModel):
    """クエリレスポンス"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class BatchQueryRequest(BaseModel):
    """バッチクエリリクエスト"""
    queries: List[str] = Field(..., description="クエリリスト")
    top_k: int = Field(5, description="取得する結果数", ge=1, le=20)
    search_type: str = Field("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")


class SystemInfoResponse(BaseModel):
    """システム情報レスポンス"""
    status: str
    system_info: Dict[str, Any]
    timestamp: str


class DocumentUploadResponse(BaseModel):
    """文書アップロードレスポンス"""
    status: str
    message: str
    document_id: Optional[str] = None
    processing_status: str


# RAGアプリケーションクラス
class RAGApplication:
    """RAGアプリケーション"""
    
    def __init__(self):
        self.query_engine: Optional[RoadDesignQueryEngine] = None
        self.metadata_manager: Optional[MetadataManager] = None
        self.is_initialized = False
        self.initialization_error = None
        
    async def initialize(self):
        """非同期でシステムを初期化"""
        try:
            logger.info("Initializing RAG system...")
            
            # クエリエンジンの初期化
            self.query_engine = RoadDesignQueryEngine()
            await asyncio.get_event_loop().run_in_executor(
                None, self.query_engine.initialize
            )
            
            # メタデータマネージャーの初期化
            self.metadata_manager = MetadataManager()
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Failed to initialize RAG system: {e}")
            
    def check_initialized(self):
        """初期化チェック"""
        if not self.is_initialized:
            if self.initialization_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAG system initialization failed: {self.initialization_error}"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="RAG system is not yet initialized"
                )


# グローバルアプリケーションインスタンス
rag_app = RAGApplication()

# FastAPIアプリ
app = FastAPI(
    title="道路設計RAGシステム API",
    description="土木道路設計特化型RAGシステムのWebAPI",
    version="1.0.0"
)


# イベントハンドラー
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    logger.info("Starting RAG API server...")
    await rag_app.initialize()


# RAG API エンドポイント
@app.get("/rag/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy" if rag_app.is_initialized else "initializing",
        "timestamp": datetime.now().isoformat(),
        "service": "Road Design RAG System"
    }


@app.get("/rag/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """システム情報を取得"""
    rag_app.check_initialized()
    
    system_info = rag_app.query_engine.get_system_info()
    
    return SystemInfoResponse(
        status="success",
        system_info=system_info,
        timestamp=datetime.now().isoformat()
    )


@app.post("/rag/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """文書検索・質問応答"""
    rag_app.check_initialized()
    
    try:
        # クエリを実行
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_app.query_engine.query,
            request.query,
            request.top_k,
            request.search_type,
            request.filters,
            request.include_sources
        )
        
        return QueryResponse(**result.to_dict())
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/batch-query")
async def batch_query_documents(request: BatchQueryRequest):
    """バッチクエリ"""
    rag_app.check_initialized()
    
    if len(request.queries) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries allowed in batch"
        )
        
    try:
        # バッチクエリを実行
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_app.query_engine.batch_query,
            request.queries,
            request.top_k,
            request.search_type
        )
        
        return {
            "status": "success",
            "results": [result.to_dict() for result in results],
            "total_queries": len(request.queries),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search")
async def search_documents(
    q: str = Query(..., description="検索クエリ"),
    top_k: int = Query(5, description="取得する結果数", ge=1, le=20),
    search_type: str = Query("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
):
    """簡易検索API"""
    
    request = QueryRequest(
        query=q,
        top_k=top_k,
        search_type=search_type,
        include_sources=True
    )
    
    return await query_documents(request)


@app.get("/rag/documents")
async def list_documents(
    limit: int = Query(50, description="取得件数", ge=1, le=100),
    offset: int = Query(0, description="オフセット", ge=0),
    document_type: Optional[str] = Query(None, description="文書タイプフィルター")
):
    """文書一覧を取得"""
    rag_app.check_initialized()
    
    try:
        # メタデータから文書を検索
        from src.rag.indexing.metadata_manager import DocumentType
        
        filters = {}
        if document_type:
            try:
                filters['document_type'] = DocumentType(document_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid document_type: {document_type}")
                
        documents = rag_app.metadata_manager.search_documents(**filters)
        
        # ページネーション
        total = len(documents)
        paginated_docs = documents[offset:offset + limit]
        
        return {
            "status": "success",
            "documents": [doc.to_dict() for doc in paginated_docs],
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/statistics")
async def get_statistics():
    """システム統計を取得"""
    rag_app.check_initialized()
    
    try:
        stats = rag_app.metadata_manager.get_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/upload-document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    category: Optional[str] = None,
    document_type: Optional[str] = None
):
    """文書をアップロードしてインデックス化"""
    
    # ファイル形式チェック
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
        
    try:
        # 一時ファイルに保存
        upload_dir = Path("./temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{file.filename}"
        temp_path = upload_dir / temp_filename
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # バックグラウンドでインデックス化を実行
        background_tasks.add_task(
            process_uploaded_document,
            str(temp_path),
            title or file.filename,
            category or "その他",
            document_type or "other"
        )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded and queued for processing",
            document_id=temp_filename,
            processing_status="queued"
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_uploaded_document(
    file_path: str,
    title: str,
    category: str,
    document_type: str
):
    """アップロードされた文書を処理（バックグラウンドタスク）"""
    
    try:
        logger.info(f"Processing uploaded document: {file_path}")
        
        # インデックス作成スクリプトを実行
        import subprocess
        
        result = subprocess.run([
            sys.executable,
            "scripts/rag/index_documents.py",
            file_path,
            "--output-dir", "./outputs/rag_index"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Document processed successfully: {file_path}")
        else:
            logger.error(f"Document processing failed: {result.stderr}")
            
        # 一時ファイルを削除
        os.remove(file_path)
        
    except Exception as e:
        logger.error(f"Background document processing failed: {e}")


# ストリーミングレスポンス用エンドポイント
@app.post("/rag/stream-query")
async def stream_query(request: QueryRequest):
    """ストリーミングクエリ（リアルタイム応答）"""
    rag_app.check_initialized()
    
    async def generate_response():
        """レスポンスを段階的に生成"""
        
        # 検索フェーズ
        yield f"data: {json.dumps({'phase': 'search', 'message': '文書を検索中...'})}\n\n"
        await asyncio.sleep(0.1)
        
        try:
            # 実際のクエリ実行
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                rag_app.query_engine.query,
                request.query,
                request.top_k,
                request.search_type,
                request.filters,
                request.include_sources
            )
            
            # 結果フェーズ
            yield f"data: {json.dumps({'phase': 'result', 'data': result.to_dict()})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'phase': 'error', 'error': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


# WebSocket対応（将来拡張用）
@app.websocket("/rag/ws")
async def websocket_endpoint(websocket):
    """WebSocket接続（将来の対話型インターフェース用）"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# 開発用サーバー起動
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8051, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "rag_interface:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )