"""
RAG (Retrieval-Augmented Generation) 関連のAPIルーター
"""

import os
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse

from ..models.rag import (
    QueryRequest, QueryResponse, BatchQueryRequest,
    SystemInfoResponse, DocumentUploadResponse,
    SavedSearchResult, SaveSearchRequest, SearchHistoryResponse
)
from ..dependencies import logger, PROJECT_ROOT, DATA_DIR, RAG_AVAILABLE

router = APIRouter(prefix="/rag", tags=["rag"])

# RAGアプリケーションのインスタンス（main_unified.pyから移動が必要）
rag_app = None

# 検索履歴の保存先
SEARCH_HISTORY_FILE = DATA_DIR / "rag" / "search_history.json"
search_history: List[SavedSearchResult] = []


def load_search_history():
    """検索履歴を読み込む"""
    global search_history
    try:
        if SEARCH_HISTORY_FILE.exists():
            with open(SEARCH_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                search_history = [SavedSearchResult(**item) for item in data]
                logger.info(f"検索履歴を読み込みました: {len(search_history)}件")
        else:
            search_history = []
    except Exception as e:
        logger.error(f"検索履歴読み込みエラー: {str(e)}")
        search_history = []


def save_search_history():
    """検索履歴を保存する"""
    try:
        SEARCH_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # SavedSearchResultオブジェクトを辞書に変換
        history_data = [item.dict() for item in search_history]
        
        with open(SEARCH_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"検索履歴を保存しました: {len(search_history)}件")
    except Exception as e:
        logger.error(f"検索履歴保存エラー: {str(e)}")


def initialize_rag():
    """RAGシステムを初期化"""
    global rag_app, RAG_AVAILABLE
    
    try:
        from src.rag.app import RAGApplication
        
        logger.info("RAGシステムを初期化中...")
        rag_app = RAGApplication()
        RAG_AVAILABLE = True
        logger.info("RAGシステムの初期化が完了しました")
        
        # 検索履歴を読み込む
        load_search_history()
        
    except ImportError as e:
        logger.warning(f"RAGモジュールのインポートエラー: {e}")
        RAG_AVAILABLE = False
    except Exception as e:
        logger.error(f"RAGシステムの初期化エラー: {e}")
        RAG_AVAILABLE = False


# 起動時にRAGを初期化
initialize_rag()


@router.get("/health")
async def rag_health_check():
    """RAGシステムのヘルスチェック"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        health_status = rag_app.health_check()
        return health_status
    except Exception as e:
        logger.error(f"RAGヘルスチェックエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/system-info", response_model=SystemInfoResponse)
async def rag_get_system_info():
    """RAGシステム情報を取得"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        system_info = rag_app.get_system_info()
        return SystemInfoResponse(
            status="active",
            system_info=system_info,
            timestamp=datetime.now(JST).isoformat()
        )
    except Exception as e:
        logger.error(f"システム情報取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system info: {str(e)}"
        )


@router.post("/update-settings")
async def rag_update_settings(settings: Dict[str, Any]):
    """RAGシステムの設定を更新"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        rag_app.update_settings(settings)
        return {"status": "success", "message": "Settings updated"}
    except Exception as e:
        logger.error(f"設定更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update settings: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def rag_query_documents(request: QueryRequest):
    """RAGクエリの実行"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        logger.info(f"RAGクエリ受信: {request.query[:100]}...")
        
        # RAGクエリを実行
        result = rag_app.query(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type,
            include_sources=request.include_sources,
            filters=request.filters
        )
        
        # レスポンスを構築
        response = QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence_score", 0.0),
            processing_time=result.get("processing_time", 0.0),
            metadata=result.get("metadata", {})
        )
        
        logger.info(f"RAGクエリ完了: 処理時間={response.processing_time:.2f}秒")
        return response
        
    except Exception as e:
        logger.error(f"RAGクエリエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/batch-query")
async def rag_batch_query_documents(request: BatchQueryRequest):
    """バッチRAGクエリの実行"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        results = []
        for query in request.queries:
            result = rag_app.query(
                query=query,
                top_k=request.top_k,
                search_type=request.search_type
            )
            results.append(result)
        
        return {"queries": request.queries, "results": results}
        
    except Exception as e:
        logger.error(f"バッチクエリエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch query failed: {str(e)}"
        )


@router.post("/search")
async def rag_search_documents(query: str, top_k: int = 5):
    """文書検索（簡易版）"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        results = rag_app.search(query=query, top_k=top_k)
        return {"query": query, "results": results}
        
    except Exception as e:
        logger.error(f"検索エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/documents")
async def rag_list_documents():
    """登録済み文書一覧を取得"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        documents = rag_app.list_documents()
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        logger.error(f"文書一覧取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/statistics")
async def rag_get_statistics():
    """RAGシステムの統計情報を取得"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        stats = rag_app.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"統計情報取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def rag_delete_document(document_id: str):
    """文書を削除"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        result = rag_app.delete_document(document_id)
        return {"status": "success", "document_id": document_id}
        
    except Exception as e:
        logger.error(f"文書削除エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/upload-document", response_model=DocumentUploadResponse)
async def rag_upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """文書をアップロードしてインデックス化"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        # メタデータをパース
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("メタデータのパースに失敗しました")
        
        # ファイルを保存
        upload_dir = DATA_DIR / "rag" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = upload_dir / f"{file_id}{file_extension}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"文書アップロード: {file.filename} -> {file_path}")
        
        # RAGシステムにインデックス化
        result = rag_app.index_document(
            file_path=str(file_path),
            metadata=doc_metadata
        )
        
        return DocumentUploadResponse(
            status="success",
            message=f"Document {file.filename} uploaded and indexed successfully",
            document_id=file_id,
            processing_status="completed",
            metadata=doc_metadata
        )
        
    except Exception as e:
        logger.error(f"文書アップロードエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post("/save-search", response_model=SavedSearchResult)
async def save_search_result(request: SaveSearchRequest):
    """検索結果を保存"""
    global search_history
    
    try:
        # 新しい検索結果を作成
        saved_result = SavedSearchResult(
            id=str(uuid.uuid4()),
            query=request.query,
            results=request.results,
            created_at=datetime.now(JST).isoformat(),
            metadata=request.metadata or {}
        )
        
        # 履歴に追加
        search_history.append(saved_result)
        
        # 最新100件のみ保持
        if len(search_history) > 100:
            search_history = search_history[-100:]
        
        # 保存
        save_search_history()
        
        logger.info(f"検索結果を保存: {saved_result.id}")
        return saved_result
        
    except Exception as e:
        logger.error(f"検索結果保存エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save search result: {str(e)}"
        )


@router.get("/search-history", response_model=SearchHistoryResponse)
async def get_search_history(limit: int = 50):
    """検索履歴を取得"""
    try:
        # 最新のものから取得
        recent_history = list(reversed(search_history))[:limit]
        
        return SearchHistoryResponse(
            history=recent_history,
            total_count=len(search_history)
        )
        
    except Exception as e:
        logger.error(f"検索履歴取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search history: {str(e)}"
        )


@router.get("/search-history/{result_id}")
async def get_saved_search_result(result_id: str):
    """保存された検索結果を取得"""
    try:
        for result in search_history:
            if result.id == result_id:
                return result
        
        raise HTTPException(
            status_code=404,
            detail=f"Search result {result_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"検索結果取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search result: {str(e)}"
        )


@router.get("/export-results/{result_id}")
async def export_search_results(result_id: str, format: str = "json"):
    """検索結果をエクスポート"""
    try:
        # 検索結果を取得
        result = None
        for r in search_history:
            if r.id == result_id:
                result = r
                break
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Search result {result_id} not found"
            )
        
        # エクスポート形式に応じて処理
        export_dir = DATA_DIR / "rag" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            file_path = export_dir / f"search_result_{timestamp}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result.dict(), f, ensure_ascii=False, indent=2)
            
            return FileResponse(
                path=str(file_path),
                filename=f"search_result_{timestamp}.json",
                media_type="application/json"
            )
        
        elif format == "csv":
            import csv
            
            file_path = export_dir / f"search_result_{timestamp}.csv"
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Query", "Result", "Score", "Source"])
                
                for r in result.results:
                    writer.writerow([
                        result.query,
                        r.get("content", ""),
                        r.get("score", 0),
                        r.get("source", "")
                    ])
            
            return FileResponse(
                path=str(file_path),
                filename=f"search_result_{timestamp}.csv",
                media_type="text/csv"
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export format: {format}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"エクスポートエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export results: {str(e)}"
        )


@router.post("/stream-query")
async def rag_stream_query(request: QueryRequest):
    """ストリーミングRAGクエリ"""
    if not RAG_AVAILABLE or not rag_app:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available"
        )
    
    try:
        async def generate():
            """ストリーミングレスポンスを生成"""
            # RAGクエリを実行
            result = rag_app.query(
                query=request.query,
                top_k=request.top_k,
                search_type=request.search_type,
                include_sources=request.include_sources,
                filters=request.filters
            )
            
            # 結果を段階的に送信
            answer = result.get("answer", "")
            for i in range(0, len(answer), 50):  # 50文字ずつ送信
                chunk = answer[i:i+50]
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.05)  # 少し遅延を入れる
            
            # 最終結果を送信
            yield f"data: {json.dumps({'done': True, 'sources': result.get('sources', [])})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"ストリーミングクエリエラー: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Streaming query failed: {str(e)}"
        )