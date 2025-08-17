"""
MoE-RAG統合エンドポイント
main_unified.pyにインポートして使用
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, List, Optional
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# パス追加
sys.path.append(str(Path(__file__).parent.parent))

# MoE-RAG統合モジュールのインポート（エラーハンドリング付き）
try:
    from src.moe_rag_integration.expert_router import ExpertRouter
    from src.moe_rag_integration.response_fusion import ResponseFusion
    MOE_RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MoE-RAG modules not available: {e}")
    MOE_RAG_AVAILABLE = False
    ExpertRouter = None
    ResponseFusion = None

logger = logging.getLogger(__name__)

# APIルーター
router = APIRouter(prefix="/api/moe-rag", tags=["MoE-RAG"])

# グローバル変数
expert_router = None
response_fusion = None

def initialize_moe_rag():
    """MoE-RAGコンポーネントを初期化"""
    global expert_router, response_fusion
    
    if not MOE_RAG_AVAILABLE:
        return False
    
    try:
        expert_router = ExpertRouter()
        response_fusion = ResponseFusion()
        logger.info("MoE-RAG components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MoE-RAG: {e}")
        return False

@router.get("/status")
async def moe_rag_status():
    """MoE-RAGシステムステータス"""
    return {
        "available": MOE_RAG_AVAILABLE,
        "initialized": expert_router is not None,
        "components": {
            "expert_router": expert_router is not None,
            "response_fusion": response_fusion is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/experts")
async def get_experts():
    """利用可能なエキスパート一覧"""
    if not MOE_RAG_AVAILABLE or expert_router is None:
        raise HTTPException(status_code=503, detail="MoE-RAG not available")
    
    experts = [
        {"id": "structural", "name": "構造設計", "description": "構造計算、耐震設計、荷重解析", "icon": "🏗️", "color": "#3498db"},
        {"id": "road", "name": "道路設計", "description": "線形設計、視距、設計速度", "icon": "🛣️", "color": "#e74c3c"},
        {"id": "geotech", "name": "地盤工学", "description": "土質調査、支持力、液状化対策", "icon": "⛰️", "color": "#8e44ad"},
        {"id": "hydraulics", "name": "水理・排水", "description": "流量計算、排水計画、治水", "icon": "💧", "color": "#2980b9"},
        {"id": "materials", "name": "材料工学", "description": "コンクリート、鋼材、材料試験", "icon": "🧱", "color": "#f39c12"},
        {"id": "construction", "name": "施工管理", "description": "工程管理、品質管理、安全管理", "icon": "👷", "color": "#27ae60"},
        {"id": "regulations", "name": "法規・基準", "description": "道路構造令、設計基準、仕様書", "icon": "📋", "color": "#34495e"},
        {"id": "environmental", "name": "環境・維持管理", "description": "環境影響、維持管理、劣化診断", "icon": "🌿", "color": "#16a085"}
    ]
    
    return {
        "experts": experts,
        "total": len(experts),
        "active": len(experts)
    }

@router.post("/analyze")
async def analyze_query(query: str = Query(..., description="分析するクエリ")):
    """クエリ分析とエキスパート選択"""
    if not MOE_RAG_AVAILABLE or expert_router is None:
        raise HTTPException(status_code=503, detail="MoE-RAG not available")
    
    try:
        # エキスパートルーティング
        routing_decision = expert_router.route(query)
        
        # クエリ複雑度分析
        complexity = expert_router.analyze_query_complexity(query)
        
        return {
            "query": query,
            "primary_experts": routing_decision.primary_experts,
            "secondary_experts": routing_decision.secondary_experts,
            "routing_strategy": routing_decision.routing_strategy,
            "confidence": routing_decision.confidence,
            "complexity": complexity,
            "keywords_detected": routing_decision.keywords_detected
        }
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ui")
async def moe_rag_ui():
    """MoE-RAG UIへのリダイレクト"""
    # static/moe_rag_ui.htmlの内容を返す
    ui_path = Path(__file__).parent / "static" / "moe_rag_ui.html"
    
    if ui_path.exists():
        with open(ui_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>MoE-RAG UI</title></head>
        <body>
            <h1>MoE-RAG統合システム</h1>
            <p>UIファイルが見つかりません。</p>
            <p>パス: /static/moe_rag_ui.html</p>
            <a href="/">メインページに戻る</a>
        </body>
        </html>
        """)

# 初期化を実行
initialize_moe_rag()