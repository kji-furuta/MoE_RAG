"""
MoE-RAG統合API
FastAPIを使用したMoE-RAGハイブリッド検索API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml
import traceback

# パスの追加
sys.path.append(str(Path(__file__).parent.parent))

# MoE-RAG統合モジュール
from src.moe_rag_integration.hybrid_query_engine import HybridMoERAGEngine
from src.moe_rag_integration.expert_router import ExpertRouter
from src.moe_rag_integration.response_fusion import ResponseFusion

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション
app = FastAPI(
    title="MoE-RAG Hybrid Search API",
    description="土木工学専門のMoE-RAGハイブリッド検索システム",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# リクエスト/レスポンスモデル
class HybridQueryRequest(BaseModel):
    """ハイブリッドクエリリクエスト"""
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(5, description="取得する文書数")
    use_reranking: bool = Field(True, description="リランキングを使用するか")
    expert_override: Optional[List[str]] = Field(None, description="エキスパートの手動指定")
    stream: bool = Field(False, description="ストリーミングレスポンス")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "設計速度80km/hの道路における最小曲線半径は？",
                "top_k": 5,
                "use_reranking": True,
                "expert_override": None,
                "stream": False
            }
        }


class HybridQueryResponse(BaseModel):
    """ハイブリッドクエリレスポンス"""
    answer: str
    confidence_score: float
    selected_experts: List[str]
    expert_contributions: Dict[str, float]
    documents: List[Dict[str, Any]]
    query_id: str
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "設計速度80km/hの道路における最小曲線半径は280mです。",
                "confidence_score": 0.92,
                "selected_experts": ["道路設計", "法規・基準"],
                "expert_contributions": {"道路設計": 0.7, "法規・基準": 0.3},
                "documents": [],
                "query_id": "q_20240101_123456",
                "processing_time": 1.234
            }
        }


class ExpertInfoResponse(BaseModel):
    """エキスパート情報レスポンス"""
    experts: List[Dict[str, Any]]
    total_experts: int
    active_experts: List[str]


class SystemInfoResponse(BaseModel):
    """システム情報レスポンス"""
    status: str
    moe_info: Dict[str, Any]
    rag_info: Dict[str, Any]
    integration_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]


# グローバル変数
hybrid_engine: Optional[HybridMoERAGEngine] = None
expert_router: Optional[ExpertRouter] = None
config: Dict[str, Any] = {}
query_cache: Dict[str, Any] = {}
active_websockets: List[WebSocket] = []


# 初期化関数
async def initialize_system():
    """システムの初期化"""
    global hybrid_engine, expert_router, config
    
    try:
        # 設定ファイルの読み込み
        config_path = Path("src/moe_rag_integration/config/integration_config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # ハイブリッドエンジンの初期化
        logger.info("Initializing Hybrid MoE-RAG Engine...")
        hybrid_engine = HybridMoERAGEngine(
            moe_model_path=config.get('moe', {}).get('model_path'),
            rag_config_path=config.get('rag', {}).get('config_path', 'src/rag/config/rag_config.yaml'),
            moe_weight=config.get('integration', {}).get('weights', {}).get('moe', 0.4),
            rag_weight=config.get('integration', {}).get('weights', {}).get('rag', 0.6)
        )
        
        # エキスパートルーターの初期化
        logger.info("Initializing Expert Router...")
        expert_router = ExpertRouter(
            num_experts_per_tok=config.get('moe', {}).get('num_experts_per_tok', 2)
        )
        
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        logger.error(traceback.format_exc())
        return False


# スタートアップイベント
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    success = await initialize_system()
    if not success:
        logger.error("System initialization failed. Some features may not work.")


# シャットダウンイベント
@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("Shutting down MoE-RAG API...")
    # WebSocket接続のクローズ
    for ws in active_websockets:
        await ws.close()


# APIエンドポイント
@app.get("/api/moe-rag/health")
async def health_check():
    """ヘルスチェック"""
    if hybrid_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "System not initialized"}
        )
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "moe": "operational",
            "rag": "operational",
            "router": "operational"
        }
    }


@app.post("/api/moe-rag/query", response_model=HybridQueryResponse)
async def hybrid_query(request: HybridQueryRequest):
    """ハイブリッドクエリ実行"""
    if hybrid_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = datetime.now()
    query_id = f"q_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # キャッシュチェック
        cache_key = f"{request.query}_{request.top_k}_{request.use_reranking}"
        if cache_key in query_cache and config.get('integration', {}).get('cache', {}).get('enabled', True):
            cached_result = query_cache[cache_key]
            cached_result['query_id'] = query_id
            cached_result['from_cache'] = True
            return cached_result
        
        # エキスパートのオーバーライド処理
        if request.expert_override:
            # 手動指定されたエキスパートを使用
            logger.info(f"Using manual expert override: {request.expert_override}")
        
        # ハイブリッドクエリの実行
        result = await hybrid_engine.query(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            stream=request.stream
        )
        
        # 処理時間の計算
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # レスポンスの構築
        response = HybridQueryResponse(
            answer=result.answer,
            confidence_score=result.confidence_score,
            selected_experts=result.moe_result.selected_experts,
            expert_contributions=result.expert_contributions,
            documents=[doc.dict() for doc in result.rag_documents[:request.top_k]],
            query_id=query_id,
            processing_time=processing_time
        )
        
        # キャッシュに保存
        if config.get('integration', {}).get('cache', {}).get('enabled', True):
            query_cache[cache_key] = response.dict()
            # キャッシュサイズ制限
            max_cache_size = config.get('integration', {}).get('cache', {}).get('max_size', 1000)
            if len(query_cache) > max_cache_size:
                # 古いエントリを削除
                oldest_key = next(iter(query_cache))
                del query_cache[oldest_key]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/moe-rag/stream")
async def stream_query(request: HybridQueryRequest):
    """ストリーミングクエリ実行"""
    if hybrid_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    async def generate():
        try:
            # ハイブリッドクエリの実行
            result = await hybrid_engine.query(
                query=request.query,
                top_k=request.top_k,
                use_reranking=request.use_reranking,
                stream=True
            )
            
            # チャンクごとに送信
            chunks = result.answer.split('。')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    data = {
                        "chunk": chunk + '。',
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "experts": result.moe_result.selected_experts if i == 0 else None
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)  # ストリーミング効果のための遅延
            
            # 完了通知
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/moe-rag/experts", response_model=ExpertInfoResponse)
async def get_expert_info():
    """エキスパート情報取得"""
    if expert_router is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    experts = []
    for expert_name in expert_router.expert_domains.keys():
        info = expert_router.get_expert_info(expert_name)
        experts.append({
            "name": expert_name,
            "keywords": info.get('keywords', []),
            "patterns": info.get('patterns', []),
            "active": True
        })
    
    return ExpertInfoResponse(
        experts=experts,
        total_experts=len(experts),
        active_experts=[e['name'] for e in experts if e['active']]
    )


@app.post("/api/moe-rag/analyze-query")
async def analyze_query(query: str):
    """クエリ分析"""
    if expert_router is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # エキスパートルーティング
        routing_decision = expert_router.route(query)
        
        # クエリ複雑度分析
        complexity = expert_router.analyze_query_complexity(query)
        
        return {
            "query": query,
            "routing": routing_decision.to_dict(),
            "complexity": complexity,
            "recommended_experts": routing_decision.primary_experts,
            "backup_experts": routing_decision.secondary_experts
        }
        
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/moe-rag/info", response_model=SystemInfoResponse)
async def get_system_info():
    """システム情報取得"""
    if hybrid_engine is None:
        return SystemInfoResponse(
            status="not_initialized",
            moe_info={},
            rag_info={},
            integration_config={},
            performance_metrics={}
        )
    
    try:
        system_info = hybrid_engine.get_system_info()
        
        # パフォーマンスメトリクス
        performance_metrics = {
            "cache_size": len(query_cache),
            "cache_hit_rate": 0.0,  # 実装が必要
            "active_connections": len(active_websockets),
            "total_queries": 0  # 実装が必要
        }
        
        return SystemInfoResponse(
            status="operational",
            moe_info=system_info.get('moe_info', {}),
            rag_info=system_info.get('rag_info', {}),
            integration_config=system_info.get('integration_config', {}),
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/moe-rag/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocketエンドポイント"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        while True:
            # クライアントからのメッセージを受信
            data = await websocket.receive_json()
            
            if data.get('type') == 'query':
                # クエリ処理
                request = HybridQueryRequest(**data.get('payload', {}))
                result = await hybrid_engine.query(
                    query=request.query,
                    top_k=request.top_k,
                    use_reranking=request.use_reranking
                )
                
                # 結果を送信
                await websocket.send_json({
                    'type': 'response',
                    'payload': {
                        'answer': result.answer,
                        'experts': result.moe_result.selected_experts,
                        'confidence': result.confidence_score
                    }
                })
            
            elif data.get('type') == 'ping':
                # ハートビート
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_websockets.remove(websocket)


@app.delete("/api/moe-rag/cache")
async def clear_cache():
    """キャッシュクリア"""
    global query_cache
    cache_size = len(query_cache)
    query_cache.clear()
    
    return {
        "message": "Cache cleared",
        "cleared_entries": cache_size
    }


# エラーハンドラー
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """一般的な例外ハンドラー"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


# メイン実行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "moe_rag_api:app",
        host="0.0.0.0",
        port=8051,
        reload=True,
        log_level="info"
    )