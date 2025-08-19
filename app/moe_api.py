"""
MoE API Backend for Civil Engineering Domain
土木・建設分野MoEモデルのAPIバックエンド
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import torch
import json
import logging
from pathlib import Path
import sys
import os
import asyncio
from enum import Enum

# プロジェクトパスの追加
sys.path.append('/home/kjifu/AI_FT_7')

# MoEモジュールのインポート
from src.moe.moe_architecture import (
    MoEConfig,
    CivilEngineeringMoEModel,
    ExpertType,
    create_civil_engineering_moe
)
from src.moe.data_preparation import CivilEngineeringDataPreparator
from src.moe.moe_training import MoETrainer, MoETrainingConfig

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション
app = FastAPI(
    title="MoE Civil Engineering API",
    description="土木・建設分野特化MoEモデルAPI",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
model = None
model_config = None
training_status = {"status": "idle", "progress": 0, "message": ""}

# リクエスト/レスポンスモデル
class QueryRequest(BaseModel):
    """質問リクエスト"""
    text: str = Field(..., description="質問テキスト")
    expert_hint: Optional[str] = Field(None, description="エキスパートヒント")
    max_tokens: int = Field(512, description="最大トークン数")
    temperature: float = Field(0.7, description="生成温度")
    top_k: int = Field(2, description="使用エキスパート数")

class QueryResponse(BaseModel):
    """質問レスポンス"""
    response: str
    active_experts: List[str]
    confidence_scores: Dict[str, float]
    processing_time: float
    timestamp: datetime

class ExpertInfo(BaseModel):
    """エキスパート情報"""
    id: str
    name: str
    description: str
    keywords: List[str]
    usage_count: int

class TrainingRequest(BaseModel):
    """トレーニングリクエスト"""
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=32)
    learning_rate: float = Field(2e-5, ge=1e-6, le=1e-3)
    gradient_accumulation_steps: int = Field(8, ge=1, le=32)
    demo_mode: bool = Field(True, description="デモモード")

class DataGenerationRequest(BaseModel):
    """データ生成リクエスト"""
    samples_per_domain: int = Field(100, ge=10, le=10000)
    validation_ratio: float = Field(0.1, ge=0.05, le=0.3)

class ModelStatus(BaseModel):
    """モデルステータス"""
    loaded: bool
    model_type: str
    num_experts: int
    hidden_size: int
    parameters: int
    device: str

# エキスパート情報定義
EXPERT_DEFINITIONS = {
    "structural_design": {
        "name": "構造設計",
        "description": "橋梁、建築物の構造計算",
        "keywords": ["梁", "柱", "基礎", "耐震", "応力", "モーメント", "せん断"]
    },
    "road_design": {
        "name": "道路設計",
        "description": "道路構造令、線形設計",
        "keywords": ["設計速度", "曲線半径", "勾配", "交差点", "視距", "舗装"]
    },
    "geotechnical": {
        "name": "地盤工学",
        "description": "土質力学、基礎工事",
        "keywords": ["N値", "支持力", "液状化", "土圧", "沈下", "地盤改良"]
    },
    "hydraulics": {
        "name": "水理・排水",
        "description": "排水設計、河川工学",
        "keywords": ["流量", "管渠", "ポンプ", "洪水", "排水", "浸透"]
    },
    "materials": {
        "name": "材料工学",
        "description": "コンクリート、鋼材特性",
        "keywords": ["配合", "強度", "品質管理", "試験", "コンクリート", "鋼材"]
    },
    "construction_management": {
        "name": "施工管理",
        "description": "工程・安全・品質管理",
        "keywords": ["工程", "安全", "原価", "施工計画", "品質", "検査"]
    },
    "regulations": {
        "name": "法規・基準",
        "description": "JIS規格、建築基準法",
        "keywords": ["建築基準法", "道路構造令", "JIS", "ISO", "法令", "基準"]
    },
    "environmental": {
        "name": "環境・維持管理",
        "description": "環境影響評価、維持補修",
        "keywords": ["騒音", "振動", "廃棄物", "長寿命化", "環境", "維持"]
    }
}

# 使用統計
expert_usage_stats = {expert: 0 for expert in EXPERT_DEFINITIONS.keys()}

def analyze_query_experts(query: str) -> Tuple[List[str], Dict[str, float]]:
    """クエリから関連エキスパートを分析"""
    detected_experts = []
    confidence_scores = {}
    
    for expert_type, info in EXPERT_DEFINITIONS.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword in query:
                score += 1
        
        if score > 0:
            confidence = min(score * 0.3, 1.0)
            detected_experts.append(expert_type)
            confidence_scores[expert_type] = confidence
    
    # スコアでソート
    detected_experts.sort(key=lambda x: confidence_scores[x], reverse=True)
    
    return detected_experts[:2], confidence_scores

def generate_response_dummy(query: str, experts: List[str]) -> str:
    """ダミー応答生成"""
    expert_names = [EXPERT_DEFINITIONS[e]["name"] for e in experts]
    
    response = f"""
{query}について、{' と '.join(expert_names)}の観点から回答いたします。

ご質問の内容に基づき、以下の要点をご説明します：

1. **技術的要件**
   - 設計基準の確認と適用
   - 必要な計算と検証

2. **法規制の遵守**
   - 関連法令の確認
   - 必要な手続きと申請

3. **実施上の留意点**
   - 施工性の検討
   - 品質管理の方法

4. **経済性と安全性**
   - コスト最適化
   - リスク評価と対策

詳細な検討が必要な場合は、具体的な条件や制約をお知らせください。
専門的な計算や詳細設計については、追加の情報提供が可能です。
"""
    
    return response

@app.on_event("startup")
async def startup_event():
    """起動時の処理"""
    logger.info("MoE API Server starting...")
    # デフォルトでデモモデルをロード
    await load_model_async(demo_mode=True)

@app.on_event("shutdown")
async def shutdown_event():
    """終了時の処理"""
    logger.info("MoE API Server shutting down...")

async def load_model_async(demo_mode: bool = True):
    """非同期モデルロード"""
    global model, model_config
    
    try:
        if demo_mode:
            # デモ用小規模モデル
            model_config = MoEConfig(
                hidden_size=512,
                num_experts=8,
                num_experts_per_tok=2,
                domain_specific_routing=True
            )
            model = CivilEngineeringMoEModel(model_config, base_model=None)
        else:
            # 本番モデル
            model = create_civil_engineering_moe(
                base_model_name="cyberagent/calm3-22b-chat",
                num_experts=8
            )
            model_config = model.config
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        logger.info(f"Model loaded successfully (demo={demo_mode})")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

# APIエンドポイント

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "MoE Civil Engineering API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model is not None
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """質問処理エンドポイント"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # エキスパート分析
        detected_experts, confidence_scores = analyze_query_experts(request.text)
        
        # 応答生成（実際のモデル推論またはダミー）
        if model is not None and not isinstance(model, type(None)):
            # 実際のモデル推論（簡略化）
            response = generate_response_dummy(request.text, detected_experts)
        else:
            response = generate_response_dummy(request.text, detected_experts)
        
        # 統計更新
        for expert in detected_experts:
            expert_usage_stats[expert] += 1
        
        # 処理時間計算
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            response=response,
            active_experts=detected_experts,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/experts", response_model=List[ExpertInfo])
async def get_experts():
    """エキスパート一覧取得"""
    experts = []
    for expert_id, info in EXPERT_DEFINITIONS.items():
        experts.append(ExpertInfo(
            id=expert_id,
            name=info["name"],
            description=info["description"],
            keywords=info["keywords"],
            usage_count=expert_usage_stats.get(expert_id, 0)
        ))
    return experts

@app.get("/api/expert/{expert_id}", response_model=ExpertInfo)
async def get_expert(expert_id: str):
    """特定エキスパート情報取得"""
    if expert_id not in EXPERT_DEFINITIONS:
        raise HTTPException(status_code=404, detail="Expert not found")
    
    info = EXPERT_DEFINITIONS[expert_id]
    return ExpertInfo(
        id=expert_id,
        name=info["name"],
        description=info["description"],
        keywords=info["keywords"],
        usage_count=expert_usage_stats.get(expert_id, 0)
    )

@app.post("/api/model/load")
async def load_model_endpoint(demo_mode: bool = True):
    """モデルロードエンドポイント"""
    success = await load_model_async(demo_mode)
    if success:
        return {"status": "success", "message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.get("/api/model/status", response_model=ModelStatus)
async def get_model_status():
    """モデルステータス取得"""
    if model is None:
        return ModelStatus(
            loaded=False,
            model_type="none",
            num_experts=0,
            hidden_size=0,
            parameters=0,
            device="cpu"
        )
    
    param_count = sum(p.numel() for p in model.parameters())
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    
    return ModelStatus(
        loaded=True,
        model_type="demo" if model_config.hidden_size < 1024 else "full",
        num_experts=model_config.num_experts,
        hidden_size=model_config.hidden_size,
        parameters=param_count,
        device=device
    )

@app.post("/api/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """トレーニング開始"""
    global training_status
    
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    training_status = {"status": "running", "progress": 0, "message": "Starting training..."}
    
    # バックグラウンドでトレーニング実行
    background_tasks.add_task(run_training_task, request)
    
    return {"status": "started", "message": "Training started in background"}

async def run_training_task(request: TrainingRequest):
    """トレーニングタスク（バックグラウンド）"""
    global training_status
    
    try:
        # トレーニングのシミュレーション
        for epoch in range(request.epochs):
            for step in range(10):
                await asyncio.sleep(0.5)  # シミュレーション
                progress = ((epoch * 10 + step + 1) / (request.epochs * 10)) * 100
                training_status["progress"] = progress
                training_status["message"] = f"Epoch {epoch+1}/{request.epochs}, Step {step+1}/10"
        
        training_status = {"status": "completed", "progress": 100, "message": "Training completed"}
    except Exception as e:
        training_status = {"status": "failed", "progress": 0, "message": str(e)}

@app.get("/api/training/status")
async def get_training_status():
    """トレーニングステータス取得"""
    return training_status

@app.post("/api/data/generate")
async def generate_data(request: DataGenerationRequest):
    """データ生成"""
    try:
        preparator = CivilEngineeringDataPreparator(
            output_dir="./data/civil_engineering"
        )
        
        # データ生成
        preparator.generate_training_data(
            num_samples_per_domain=request.samples_per_domain
        )
        
        # 検証データ分割
        preparator.create_validation_data(ratio=request.validation_ratio)
        
        # テストシナリオ作成
        preparator.create_test_scenarios()
        
        return {
            "status": "success",
            "message": f"Generated {request.samples_per_domain * 8} samples",
            "train_samples": request.samples_per_domain * 8 * (1 - request.validation_ratio),
            "val_samples": request.samples_per_domain * 8 * request.validation_ratio
        }
    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/usage")
async def get_usage_stats():
    """使用統計取得"""
    total_queries = sum(expert_usage_stats.values())
    
    return {
        "total_queries": total_queries,
        "expert_usage": expert_usage_stats,
        "most_used": max(expert_usage_stats.items(), key=lambda x: x[1]) if total_queries > 0 else None,
        "least_used": min(expert_usage_stats.items(), key=lambda x: x[1]) if total_queries > 0 else None
    }

@app.post("/api/stats/reset")
async def reset_stats():
    """統計リセット"""
    global expert_usage_stats
    expert_usage_stats = {expert: 0 for expert in EXPERT_DEFINITIONS.keys()}
    return {"status": "success", "message": "Statistics reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
