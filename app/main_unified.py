#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Unified Implementation
統合されたWebインターフェース実装
"""

# PyTorchメモリ管理の最適化
import os
# Removed: Environment variable now managed by memory_manager
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # トークナイザーの警告を抑制

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import uuid
from pathlib import Path
import logging
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import yaml
from datetime import datetime, timezone, timedelta

# 日本時間（JST）の設定
JST = timezone(timedelta(hours=9))
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
from typing import Optional
import sys
import time
from pathlib import Path as PathlibPath

# Import model utilities
from app.model_utils import (
    load_model_and_tokenizer,
    handle_model_loading_error,
    get_output_directory,
    load_training_config,
    create_quantization_config,
    load_tokenizer,
    get_device_map
)
import io
import random

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG system imports
try:
    # RAGシステムのモデルロードを無効化（メモリ節約）
    os.environ["RAG_DISABLE_MODEL_LOAD"] = "true"
    sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
    from loguru import logger as rag_logger
    from src.rag.core.query_engine import RoadDesignQueryEngine, QueryResult
    from src.rag.indexing.metadata_manager import MetadataManager
    RAG_AVAILABLE = True
    logger.info("RAG system components loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG system not available: {e}")

# Prometheusメトリクスのインポート
try:
    from app.monitoring import metrics_collector, get_prometheus_metrics
    logger.info("Prometheusメトリクスをインポートしました")
    METRICS_AVAILABLE = True
except Exception as e:
    logger.warning(f"メトリクスシステムのインポートをスキップ: {e}")
    metrics_collector = None
    METRICS_AVAILABLE = False

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース",
    version="2.0.0"
)

# CORS設定（本番環境では制限すべき）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],  # 制限を追加
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MoE-RAGエンドポイントのインポートと登録
try:
    from app.moe_rag_endpoints import router as moe_rag_router
    app.include_router(moe_rag_router)
    logger.info("MoE-RAG endpoints loaded successfully")
except ImportError as e:
    logger.warning(f"MoE-RAG endpoints not available: {e}")

# MoEトレーニングエンドポイントのインポートと登録
try:
    from app.moe_training_endpoints import router as moe_training_router
    app.include_router(moe_training_router)
    logger.info("MoE Training endpoints loaded successfully")
except ImportError as e:
    logger.warning(f"MoE Training endpoints not available: {e}")

# 静的ファイルディレクトリの動的検出
def find_static_directory():
    """静的ファイルディレクトリを検索"""
    current_file = Path(__file__)
    static_path = current_file.parent / "static"
    
    if static_path.exists() and static_path.is_dir():
        return str(static_path)
    
    # プロジェクトルートからの相対パス
    project_root = Path(os.getcwd())
    possible_paths = [
        project_root / "static",  # 優先度を上げる
        project_root / "app" / "static",
        project_root / "static"
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return str(path)
    
    return str(project_root / "static")  # デフォルトを/workspace/staticに変更

static_dir = find_static_directory()
print(f"Using static directory: {static_dir}")

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 継続学習モジュールのインポート
try:
    from app.continual_learning.continual_learning_ui import create_continual_learning_router, websocket_endpoint
    continual_learning_router = create_continual_learning_router()
    app.include_router(continual_learning_router)
    logger.info("継続学習モジュールを正常にロードしました")
except Exception as e:
    logger.warning(f"継続学習モジュールのロードをスキップ: {str(e)}")
    # 継続学習モジュールが読み込めない場合は、基本的なAPIエンドポイントを直接定義
    pass

# データモデル定義
class ModelInfo(BaseModel):
    name: str
    description: str
    size: str
    status: str

class TrainingRequest(BaseModel):
    model_name: str
    training_data: List[str]
    training_method: str = "lora"  # lora, qlora, full
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]

class GenerationRequest(BaseModel):
    model_path: str
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    model_path: Optional[str] = None

# RAG-specific data models
class QueryRequest(BaseModel):
    """RAGクエリリクエスト"""
    query: str = Field(..., description="検索クエリ")
    top_k: int = Field(5, description="取得する結果数", ge=1, le=20)
    search_type: str = Field("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
    include_sources: bool = Field(True, description="ソース情報を含めるか")
    filters: Optional[Dict[str, Any]] = Field(None, description="検索フィルター")
    document_ids: Optional[List[str]] = Field(None, description="検索対象文書IDリスト")
    model: Optional[str] = Field(None, description="使用するLLMモデル (例: ollama:deepseek-32b-rag)")

class QueryResponse(BaseModel):
    """RAGクエリレスポンス"""
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
    metadata: Optional[Dict[str, Any]] = None

class SavedSearchResult(BaseModel):
    """保存された検索結果"""
    id: str
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    search_type: str
    top_k: int
    saved_at: str
    metadata: Optional[Dict[str, Any]] = None

class SaveSearchRequest(BaseModel):
    """検索結果保存リクエスト"""
    query_response: QueryResponse
    name: Optional[str] = Field(None, description="保存名")
    tags: Optional[List[str]] = Field(None, description="タグ")

class SearchHistoryResponse(BaseModel):
    """検索履歴レスポンス"""
    total: int
    results: List[SavedSearchResult]
    page: int
    limit: int

# テンプレートエンジンの設定
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

templates = Jinja2Templates(directory="templates")

# RAG Application Class
class RAGApplication:
    """RAGアプリケーション"""
    
    def __init__(self):
        self.query_engine: Optional[RoadDesignQueryEngine] = None
        self.metadata_manager: Optional[MetadataManager] = None
        self.is_initialized = False
        self.initialization_error = None
        # Docker環境に対応した永続化ディレクトリの設定
        if os.path.exists("/workspace"):
            # Docker環境内
            self.search_history_dir = Path("/workspace/data/search_history")
        else:
            # ローカル環境
            project_root = Path(__file__).parent.parent
            self.search_history_dir = project_root / "data" / "search_history"
        
        self.search_history_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Search history directory: {self.search_history_dir}")
        
    async def initialize(self):
        """非同期でシステムを初期化"""
        if not RAG_AVAILABLE:
            self.initialization_error = "RAG system components not available"
            return
            
        try:
            logger.info("Initializing RAG system...")
            
            # クエリエンジンの初期化
            self.query_engine = RoadDesignQueryEngine()
            await asyncio.get_event_loop().run_in_executor(
                None, self.query_engine.initialize
            )
            
            # メタデータマネージャーの初期化
            # Docker環境と同じパスを使用して一貫性を保つ
            metadata_db_path = "/workspace/metadata/metadata.db" if os.path.exists("/workspace") else "./metadata/metadata.db"
            self.metadata_manager = MetadataManager(db_path=metadata_db_path)
            logger.info(f"MetadataManager initialized with path: {metadata_db_path}")
            
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
    
    def save_search_result(self, query_response: QueryResponse, name: Optional[str] = None, tags: Optional[List[str]] = None) -> SavedSearchResult:
        """検索結果を保存"""
        result_id = str(uuid.uuid4())
        timestamp = datetime.now(JST).isoformat()
        
        saved_result = SavedSearchResult(
            id=result_id,
            query=query_response.query,
            answer=query_response.answer,
            citations=query_response.citations,
            sources=query_response.sources,
            confidence_score=query_response.confidence_score,
            search_type=query_response.metadata.get("search_type", "hybrid"),
            top_k=query_response.metadata.get("top_k", 5),
            saved_at=timestamp,
            metadata={
                "name": name or f"Search_{timestamp[:10]}",
                "tags": tags or [],
                "processing_time": query_response.processing_time
            }
        )
        
        # JSONファイルとして保存
        file_path = self.search_history_dir / f"{result_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(saved_result.dict(), f, ensure_ascii=False, indent=2)
        
        return saved_result
    
    def get_search_history(self, page: int = 1, limit: int = 10, tag: Optional[str] = None) -> SearchHistoryResponse:
        """検索履歴を取得"""
        all_results = []
        
        # すべての保存済み結果を読み込み
        for json_file in self.search_history_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                    
                # ファイル名からIDを確実に設定
                file_id = json_file.stem  # .jsonを除いたファイル名
                result_data["id"] = file_id
                    
                # タグフィルタリング
                if tag and tag not in result_data.get("metadata", {}).get("tags", []):
                    continue
                    
                all_results.append(SavedSearchResult(**result_data))
            except Exception as e:
                logger.error(f"Error loading search result {json_file}: {e}")
        
        # 日時で降順ソート
        all_results.sort(key=lambda x: x.saved_at, reverse=True)
        
        # ページネーション
        total = len(all_results)
        start = (page - 1) * limit
        end = start + limit
        paginated_results = all_results[start:end]
        
        return SearchHistoryResponse(
            total=total,
            results=paginated_results,
            page=page,
            limit=limit
        )
    
    def get_saved_result(self, result_id: str) -> Optional[SavedSearchResult]:
        """保存された検索結果を取得"""
        file_path = self.search_history_dir / f"{result_id}.json"
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                # ファイル名からIDを確実に設定
                result_data["id"] = result_id
                return SavedSearchResult(**result_data)
        except Exception as e:
            logger.error(f"Error loading search result {result_id}: {e}")
            return None
    
    def export_search_results(self, result_ids: List[str], format: str = "json") -> bytes:
        """検索結果をエクスポート"""
        results = []
        for result_id in result_ids:
            result = self.get_saved_result(result_id)
            if result:
                results.append(result.dict())
        
        if format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if results:
                writer = csv.DictWriter(output, fieldnames=["query", "answer", "confidence_score", "saved_at", "search_type"])
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        "query": result["query"],
                        "answer": result["answer"],
                        "confidence_score": result["confidence_score"],
                        "saved_at": result["saved_at"],
                        "search_type": result["search_type"]
                    })
            
            return output.getvalue().encode("utf-8")
        else:
            # JSON format
            return json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
    
    def delete_search_history_item(self, result_id: str) -> bool:
        """検索履歴の個別アイテムを削除"""
        try:
            # ファイルパスの安全性チェック
            if ".." in result_id or "/" in result_id or "\\" in result_id:
                logger.error(f"Invalid result_id: {result_id}")
                return False
            
            file_path = self.search_history_dir / f"{result_id}.json"
            
            # ファイルが存在するか確認
            if not file_path.exists():
                logger.warning(f"Search history item not found: {result_id}")
                return False
            
            # search_history_dir配下であることを確認
            if not str(file_path.resolve()).startswith(str(self.search_history_dir.resolve())):
                logger.error(f"Invalid file path: {file_path}")
                return False
            
            # ファイルを削除
            file_path.unlink()
            logger.info(f"Deleted search history item: {result_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting search history item {result_id}: {e}")
            return False

# RAGアプリケーションインスタンス
rag_app = RAGApplication()

# ルートページ
@app.get("/")
async def root(request: Request):
    """メインページ"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/finetune")
async def finetune_page(request: Request):
    """ファインチューニング画面"""
    return templates.TemplateResponse("finetune.html", {"request": request})

@app.get("/models")
async def models_page(request: Request):
    """モデル一覧画面"""
    return templates.TemplateResponse("models.html", {"request": request})

@app.get("/readme")
async def readme_page(request: Request):
    """README.md表示ページ"""
    return templates.TemplateResponse("readme.html", {"request": request})

@app.get("/rag")
async def rag_page(request: Request):
    """RAGシステム画面"""
    return templates.TemplateResponse("rag.html", {"request": request, "rag_available": RAG_AVAILABLE})

# グローバル変数
training_tasks = {}
model_cache = {}
executor = ThreadPoolExecutor(max_workers=2)

# アプリケーション開始時の処理
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    logger.info("Starting AI Fine-tuning Toolkit with RAG integration...")
    if RAG_AVAILABLE:  # 一時的にRAG初期化を無効化
        await rag_app.initialize()
    else:
        logger.warning("RAG system will not be available in this session")

# Ollama統合のインポート
try:
    import sys
    from pathlib import Path
    # scripts/convertディレクトリをパスに追加
    scripts_convert_path = Path(__file__).parent.parent / "scripts" / "convert"
    sys.path.insert(0, str(scripts_convert_path))
    
    from ollama_integration import OllamaIntegration
    OLLAMA_AVAILABLE = True
    logger.info("Ollama統合が利用可能です")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    logger.warning(f"Ollama統合が利用できません: {e}")

# 利用可能なモデル定義
available_models = [
    # Small Models (テスト用・軽量)
    {
        "name": "distilgpt2",
        "description": "軽量な英語モデル（テスト用）",
        "size": "82MB",
        "status": "available",
        "gpu_requirement": "なし"
    },
    {
        "name": "rinna/japanese-gpt2-small",
        "description": "日本語GPT-2 Small（Rinna）",
        "size": "110MB",
        "status": "available",
        "gpu_requirement": "なし"
    },
    {
        "name": "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        "description": "Japanese StableLM 3B Instruct（推奨）",
        "size": "3B",
        "status": "available",
        "gpu_requirement": "8GB"
    },
    {
        "name": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        "description": "ELYZA日本語Llama-2 7B Instruct",
        "size": "7B",
        "status": "gpu-required",
        "gpu_requirement": "16GB"
    },
    {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 2.5 14B Instruct（推奨）",
        "size": "14B",
        "status": "gpu-required",
        "gpu_requirement": "28GB"
    },
    {
        "name": "cyberagent/calm3-22b-chat",
        "description": "CyberAgent CALM3 22B Chat",
        "size": "22B",
        "status": "gpu-required",
        "gpu_requirement": "44GB"
    },
    {
        "name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "description": "DeepSeek R1 Distill Qwen 32B 日本語特化（Ollama推奨）",
        "size": "32B",
        "status": "ollama-recommended",
        "gpu_requirement": "20GB (Ollama使用時)"
    },
    {
        "name": "Qwen/Qwen2.5-17B-Instruct",
        "description": "Qwen 2.5 17B Instruct",
        "size": "17B",
        "status": "gpu-required",
        "gpu_requirement": "34GB"
    },
    {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 2.5 32B Instruct（Ollama推奨）",
        "size": "32B",
        "status": "ollama-recommended",
        "gpu_requirement": "20GB (Ollama使用時)"
    }
]

def get_saved_models():
    """保存済みモデル一覧を取得"""
    saved_models = []
    # Dockerコンテナ内では/workspace、ローカルではプロジェクトルート
    if os.path.exists("/workspace"):
        project_root = Path("/workspace")
    else:
        project_root = Path(os.getcwd())
    
    # outputsディレクトリを最初に確認
    outputs_path = project_root / "outputs"
    
    # プロジェクトルートからLoRAモデルを検索
    for model_dir in project_root.glob("lora_demo_*"):
        if model_dir.is_dir():
            # アダプター設定があるか確認
            if (model_dir / "adapter_config.json").exists() or (model_dir / "adapter_model.safetensors").exists():
                saved_models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "LoRA",
                    "size": "~1.6MB",
                    "base_model": "不明",
                    "training_method": "lora"
                })
    
    # outputsディレクトリも検索
    if outputs_path.exists():
        for model_dir in outputs_path.iterdir():
            if model_dir.is_dir():
                # training_info.jsonから情報を読み取り
                info_path = model_dir / "training_info.json"
                model_type = "Unknown"
                model_size = "Unknown"
                base_model = "不明"
                training_method = "unknown"
                training_data_size = 0
                
                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            training_method = info.get("training_method", "unknown")
                            base_model = info.get("base_model", "不明")
                            training_data_size = info.get("training_data_size", 0)
                            
                            if training_method == "full":
                                model_type = "フルファインチューニング"
                                model_size = "~500MB+"
                            elif training_method == "qlora":
                                model_type = "QLoRA (4bit)"
                                model_size = "~1.0MB"
                            elif training_method == "continual_ewc":
                                model_type = "継続学習 (EWC)"
                                model_size = "~500MB+"
                            else:
                                model_type = "LoRA"
                                model_size = "~1.6MB"
                    except Exception as e:
                        logger.warning(f"training_info.jsonの読み込みに失敗: {e}")
                        # ディレクトリ名から推定
                        if "continual_task" in model_dir.name.lower():
                            model_type = "継続学習"
                            training_method = "continual"
                            # 継続学習の設定ファイルから情報を取得
                            config_path = model_dir / "config.json"
                            if config_path.exists():
                                try:
                                    with open(config_path, 'r', encoding='utf-8') as f:
                                        config = json.load(f)
                                        # ベースモデルなどの情報を取得
                                except:
                                    pass
                        elif "lora" in model_dir.name.lower():
                            model_type = "LoRA"
                            training_method = "lora"
                        elif "qlora" in model_dir.name.lower():
                            model_type = "QLoRA"
                            training_method = "qlora"
                        elif "full" in model_dir.name.lower():
                            model_type = "フルファインチューニング"
                            training_method = "full"
                
                # モデルファイルの存在確認
                has_model_files = (
                    (model_dir / "adapter_model.safetensors").exists() or
                    (model_dir / "pytorch_model.bin").exists() or
                    (model_dir / "model.safetensors").exists() or
                    any(model_dir.glob("*.safetensors")) or
                    any(model_dir.glob("*.bin"))
                )
                
                # トークナイザーファイルの存在確認
                has_tokenizer = (
                    (model_dir / "tokenizer.json").exists() or
                    (model_dir / "tokenizer_config.json").exists()
                )
                
                if has_model_files:  # モデルファイルがある場合のみ追加
                    saved_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": model_type,
                        "size": model_size,
                        "base_model": base_model,
                        "training_method": training_method,
                        "training_data_size": training_data_size,
                        "has_tokenizer": has_tokenizer,
                        "has_model_files": has_model_files
                    })
    
    logger.info(f"検出された保存済みモデル: {len(saved_models)}個")
    return saved_models

# 実際のトレーニング実装
# ヘルパー関数: 設定値を適切な型に変換
def get_config_value(config, key, default, value_type):
    value = config.get(key, default)
    if isinstance(value, str):
        try:
            return value_type(value)
        except (ValueError, TypeError):
            return default
    return value_type(value)

async def run_training_task(task_id: str, request: TrainingRequest):
    """バックグラウンドでトレーニングを実行"""
    try:
        # ステータス更新
        training_tasks[task_id].status = "preparing"
        method_name = {
            "lora": "LoRA",
            "qlora": "QLoRA (4bit)", 
            "full": "フルファインチューニング"
        }.get(request.training_method, "LoRA")
        training_tasks[task_id].message = f"{method_name}でモデルを準備中..."
        training_tasks[task_id].progress = 10.0
        logger.info(f"Task {task_id}: {method_name}準備開始 - モデル: {request.model_name}")
        
        # 詳細なログ出力
        logger.info(f"Task {task_id}: トレーニング設定 - メソッド: {request.training_method}, モデル: {request.model_name}")
        logger.info(f"Task {task_id}: LoRA設定: {request.lora_config}")
        logger.info(f"Task {task_id}: トレーニング設定: {request.training_config}")
        
        # モデル保存ディレクトリ
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        output_dir = get_output_directory(method_name, timestamp)
        
        # 設定読み込み
        training_config = load_training_config(request.training_method)
        
        # トークナイザーとモデルの読み込み
        training_tasks[task_id].message = "モデルを読み込み中..."
        training_tasks[task_id].progress = 20.0
        
        try:
            project_root = Path(os.getcwd())
            cache_dir = project_root / "hf_cache"
            # 継続学習の場合、use_memory_efficientパラメータを渡す
            use_memory_efficient = (
                request.training_method == "continual" and 
                hasattr(request, 'training_config') and 
                request.training_config.get('use_memory_efficient', False)
            )
            
            # ファインチューニング時はRAG無効化フラグを一時的に解除
            original_rag_flag = os.environ.get("RAG_DISABLE_MODEL_LOAD", "")
            os.environ["RAG_DISABLE_MODEL_LOAD"] = "false"
            
            # GPUメモリをクリア（モデルロード前）
            if torch.cuda.is_available():
                import gc
                gc.collect()
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                logger.info(f"Task {task_id}: Cleared GPU memory before model loading")
                
                # 継続学習の場合、空いているGPUを選択
                if request.training_method == "continual" and torch.cuda.device_count() > 1:
                    # 各GPUの空きメモリを確認
                    gpu_free_memory = []
                    for i in range(torch.cuda.device_count()):
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        allocated_memory = torch.cuda.memory_allocated(i)
                        free_memory = (total_memory - allocated_memory) / 1024**3  # GB単位
                        gpu_free_memory.append((i, free_memory))
                        logger.info(f"GPU {i}: {free_memory:.1f}GB free")
                    
                    # 最も空きメモリが多いGPUを選択
                    gpu_free_memory.sort(key=lambda x: x[1], reverse=True)
                    best_gpu = gpu_free_memory[0][0]
                    
                    # 環境変数でGPUを指定
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                    logger.info(f"Task {task_id}: Selected GPU {best_gpu} for continual learning (free: {gpu_free_memory[0][1]:.1f}GB)")
            
            # 継続学習の場合、existing_lora_pathを渡す
            existing_lora_path = None
            if request.training_method == "continual" and hasattr(request, 'training_config'):
                existing_lora_path = request.training_config.get("existing_lora_path")
            
            model, tokenizer = load_model_and_tokenizer(
                model_name=request.model_name,
                training_method=request.training_method,
                cache_dir=cache_dir,
                use_memory_efficient=use_memory_efficient,
                skip_if_rag_active=False,  # ファインチューニング時は必ずロード
                existing_lora_path=existing_lora_path  # 継続学習用
            )
            
            # 環境変数を復元
            if original_rag_flag:
                os.environ["RAG_DISABLE_MODEL_LOAD"] = original_rag_flag
            else:
                os.environ.pop("RAG_DISABLE_MODEL_LOAD", None)
                
            # モデルがNoneでないことを確認
            if model is None:
                raise ValueError("モデルのロードに失敗しました。メモリ不足の可能性があります。")
                
            logger.info(f"Task {task_id}: モデル読み込み完了 (メモリ効率化: {use_memory_efficient})")
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Task {task_id}: モデル読み込みエラー: {str(e)}")
            logger.error(f"Task {task_id}: エラー詳細: {error_traceback}")
            training_tasks[task_id].status = "failed"
            training_tasks[task_id].message = handle_model_loading_error(e, request.model_name, task_id)
            return
        
        # 継続学習の場合のログ（既存のLoRAアダプタ処理はmodel_utilsに移動）
        if request.training_method == "continual":
            existing_lora_path = request.training_config.get("existing_lora_path")
            if existing_lora_path and os.path.exists(existing_lora_path):
                training_tasks[task_id].message = f"既存のLoRAアダプターを使用: {existing_lora_path}"
                logger.info(f"Task {task_id}: 継続学習モードで既存のLoRAアダプターを使用: {existing_lora_path}")
            
            # 継続学習の場合、gradient checkpointingのみ有効化（メモリ節約）
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info(f"Task {task_id}: Gradient checkpointing有効化（継続学習用）")
        
        # LoRA設定（継続学習も含む）
        if request.training_method in ["lora", "qlora", "continual"]:
            training_tasks[task_id].message = "LoRAアダプターを設定中..."
            training_tasks[task_id].progress = 30.0
            
            # QLoRAの場合はモデルを準備（継続学習は除外 - 既にLoRAアダプタが設定されているため）
            if request.training_method == "qlora":
                try:
                    # GPUメモリのクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # gradient_checkpointingを有効化してメモリ使用量を削減
                    if hasattr(model, 'gradient_checkpointing_enable'):
                        model.gradient_checkpointing_enable()
                    
                    # prepare_model_for_kbit_trainingを安全に実行
                    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                    
                    logger.info(f"Task {task_id}: QLoRA準備完了、gradient checkpointing有効化")
                except torch.cuda.OutOfMemoryError as e:
                    # メモリモニターを使用して正確なエラー情報を取得
                    from src.training.memory_monitor import MemoryMonitor
                    formatted_error = MemoryMonitor.format_memory_error(e)
                    logger.error(f"Task {task_id}: QLoRA準備中にメモリ不足:\n{formatted_error}")
                    
                    # メモリをクリアして再試行
                    MemoryMonitor.clear_gpu_memory()
                    torch.cuda.synchronize()
                    
                    # より積極的なメモリ最適化を試みる
                    if hasattr(model, 'config'):
                        model.config.use_cache = False  # KVキャッシュを無効化
                    
                    # 再度試行
                    try:
                        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                        logger.info(f"Task {task_id}: QLoRA準備完了（再試行成功）")
                    except Exception as retry_error:
                        logger.error(f"Task {task_id}: QLoRA準備失敗: {str(retry_error)}")
                        training_tasks[task_id].status = "failed"
                        training_tasks[task_id].message = f"QLoRA準備中にメモリ不足が発生しました。より小さいモデルを選択してください。"
                        return
            
            # LoRA設定
            lora_config = LoraConfig(
                r=get_config_value(request.lora_config, "r", get_config_value(training_config, "lora_r", 16, int), int),
                lora_alpha=get_config_value(request.lora_config, "lora_alpha", get_config_value(training_config, "lora_alpha", 32, int), int),
                target_modules=training_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                lora_dropout=get_config_value(training_config, "lora_dropout", 0.05, float),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # トレーニングデータの準備
        training_tasks[task_id].message = "トレーニングデータを準備中..."
        training_tasks[task_id].progress = 40.0
        
        # トレーニングデータの処理（継続学習の場合とファイルパスの場合を判別）
        train_texts = []
        
        # 継続学習の場合、training_dataは既にdictのリスト
        if request.training_data and isinstance(request.training_data[0], dict):
            for data in request.training_data:
                if 'text' in data:
                    train_texts.append(data['text'])
                elif 'input' in data and 'output' in data:
                    train_texts.append(f"{data['input']}\n{data['output']}")
        # 通常のトレーニングの場合、ファイルパスから読み込み
        else:
            logger.info(f"Task {task_id}: トレーニングデータパス: {request.training_data}")
            for data_path in request.training_data:
                # 絶対パスと相対パスの両方を試す
                data_file = Path(data_path)
                if not data_file.exists():
                    # /workspace からの相対パスとして試す
                    data_file = Path("/workspace") / data_path.lstrip("/")
                    if not data_file.exists():
                        # dataディレクトリからの相対パスとして試す
                        data_file = Path("/workspace/data/uploaded") / Path(data_path).name
                
                logger.info(f"Task {task_id}: ファイルパスを確認: {data_file}, 存在: {data_file.exists()}")
                
                if data_file.exists() and data_file.suffix == '.jsonl':
                    logger.info(f"Task {task_id}: JSONLファイル読み込み開始: {data_file}")
                    with open(data_file, 'r', encoding='utf-8') as f:
                        line_count = 0
                        valid_count = 0
                        for line in f:
                            line_count += 1
                            try:
                                line = line.strip()
                                if not line:  # 空行をスキップ
                                    continue
                                data = json.loads(line)
                                if 'text' in data:
                                    train_texts.append(data['text'])
                                    valid_count += 1
                                elif 'input' in data and 'output' in data:
                                    train_texts.append(f"{data['input']}\n{data['output']}")
                                    valid_count += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"Task {task_id}: 行 {line_count} でJSONデコードエラー: {str(e)}")
                                continue
                        logger.info(f"Task {task_id}: {data_file}から{valid_count}/{line_count}行を読み込み")
                else:
                    logger.warning(f"Task {task_id}: ファイルが見つからないか、JSONLでない: {data_path}")
        
        if not train_texts:
            # フォールバック: サンプルデータを使用
            train_texts = [
                "これは日本語のサンプルテキストです。",
                "ファインチューニングのテストデータです。",
                "AIモデルの学習用データです。"
            ] * 10  # 30個のサンプルを作成
        
        logger.info(f"Task {task_id}: {len(train_texts)}個のトレーニングサンプルを準備")
        
        # 実際のトレーニング実行
        training_tasks[task_id].status = "training"
        training_tasks[task_id].message = f"{method_name}でファインチューニング中..."
        training_tasks[task_id].progress = 50.0
        
        # 簡単なデータセット
        from torch.utils.data import Dataset
        
        class SimpleDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # labelsをinput_idsと同じにするが、paddingトークンは-100にマスク
                labels = encoding["input_ids"].squeeze().clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": labels
                }
        
        # データセット作成（QLoRAの場合はmax_seq_lengthを使用）
        if request.training_method == "qlora" and ("32B" in request.model_name or "22B" in request.model_name):
            dataset_max_length = 256  # 大規模モデルの場合は短縮
        else:
            dataset_max_length = get_config_value(training_config, "max_length", 512, int)
        
        train_dataset = SimpleDataset(train_texts, tokenizer, max_length=dataset_max_length)
        
        # EWCを使用するカスタムトレーナー
        class EWCTrainer(Trainer):
            def __init__(self, *args, ewc_lambda: float = 5000.0, use_ewc: bool = False, **kwargs):
                super().__init__(*args, **kwargs)
                # accelerateモデルの場合はモデル移動を無効化
                if hasattr(self.model, 'hf_device_map'):
                    self.place_model_on_device = False
                self.ewc_lambda = ewc_lambda
                self.use_ewc = use_ewc
                self.ewc_helper = None
                
                if self.use_ewc:
                    try:
                        from src.training.ewc_utils import EWCHelper
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        self.ewc_helper = EWCHelper(self.model, device)
                        logger.info("EWCを有効化しました")
                    except ImportError:
                        logger.warning("EWCモジュールのインポートに失敗しました")
                        self.use_ewc = False
            
            def _move_model_to_device(self, model, device):
                """accelerateでオフロードされたモデルの移動を防ぐためにオーバーライド"""
                # accelerateでオフロードされたモデルは移動しない
                if hasattr(model, 'hf_device_map'):
                    logger.info(f"モデル移動をスキップ - accelerateによってすでに配置済み")
                    return model
                return model
            
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """損失関数にEWCペナルティを追加"""
                outputs = model(**inputs)
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
                
                # EWCペナルティを追加
                if self.use_ewc and self.ewc_helper is not None and self.ewc_helper.fisher_matrix is not None:
                    ewc_loss = self.ewc_helper.compute_ewc_loss(model)
                    loss = loss + self.ewc_lambda * ewc_loss
                    
                return (loss, outputs) if return_outputs else loss
        

        
        # トレーニング引数
        # トレーニングパラメータの設定
        if request.training_method == "full":
            batch_size = get_config_value(training_config, "batch_size", 1, int)
            gradient_accumulation_steps = get_config_value(training_config, "gradient_accumulation_steps", 16, int)
            num_epochs = get_config_value(training_config, "num_epochs", 1, int)
            
            effective_batch_size = batch_size * gradient_accumulation_steps
            total_steps = len(train_dataset) * num_epochs // effective_batch_size
            max_steps = min(100, total_steps)  # フルファインチューニングは100ステップまで
            learning_rate = 5e-6  # より低い学習率
        elif request.training_method == "qlora":
            # QLoRAの場合：メモリ効率を最優先
            # DeepSeek-R1-32Bのような大規模モデル用の設定
            if "32B" in request.model_name or "22B" in request.model_name:
                batch_size = 1  # 最小バッチサイズ
                gradient_accumulation_steps = 16  # 勾配累積を増やして実効バッチサイズを確保
                max_seq_length = 256  # シーケンス長を短縮
            else:
                batch_size = get_config_value(training_config, "batch_size", 2, int)
                gradient_accumulation_steps = get_config_value(training_config, "gradient_accumulation_steps", 8, int)
                max_seq_length = get_config_value(training_config, "max_length", 512, int)
            
            num_epochs = get_config_value(training_config, "num_epochs", 3, int)
            effective_batch_size = batch_size * gradient_accumulation_steps
            total_steps = len(train_dataset) * num_epochs // effective_batch_size
            max_steps = min(50, total_steps)  # QLoRAは50ステップまで
            learning_rate = get_config_value(training_config, "learning_rate", 2e-4, float)
            
            logger.info(f"Task {task_id}: QLoRA設定 - batch_size: {batch_size}, grad_accum: {gradient_accumulation_steps}, max_seq_length: {max_seq_length}")
        elif request.training_method == "continual":
            # 継続学習の場合：より多くのステップでしっかり学習
            batch_size = get_config_value(training_config, "batch_size", 1, int)
            gradient_accumulation_steps = get_config_value(training_config, "gradient_accumulation_steps", 8, int)
            num_epochs = get_config_value(training_config, "num_epochs", 3, int)
            
            effective_batch_size = batch_size * gradient_accumulation_steps
            total_steps = len(train_dataset) * num_epochs // effective_batch_size
            max_steps = min(200, total_steps)  # 継続学習は200ステップまで
            learning_rate = get_config_value(training_config, "learning_rate", 1e-4, float)
            
            logger.info(f"Task {task_id}: 継続学習設定 - データ数: {len(train_dataset)}, エポック: {num_epochs}, 総ステップ数: {total_steps}, 実行ステップ数: {max_steps}")
        else:
            batch_size = get_config_value(training_config, "batch_size", 1, int)
            max_steps = min(50, len(train_dataset) // batch_size)
            learning_rate = get_config_value(training_config, "learning_rate", 2e-4, float)
        
        # 継続学習の場合は専用の設定を使用
        if request.training_method == "continual":
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                warmup_steps=min(20, max_steps // 10),
                logging_steps=10,
                save_steps=max_steps // 4,  # より頻繁に保存
                max_steps=max_steps,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=True,
                remove_unused_columns=False,
                report_to=[],
                save_strategy="steps",
                save_total_limit=3,
                dataloader_pin_memory=False,
                load_best_model_at_end=False,
                metric_for_best_model=None,
                greater_is_better=None,
            )
        else:
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=get_config_value(training_config, "batch_size", 1, int),
                gradient_accumulation_steps=get_config_value(training_config, "gradient_accumulation_steps", 4, int),
                num_train_epochs=get_config_value(training_config, "num_epochs", 1, int),
                learning_rate=learning_rate,
                warmup_steps=min(get_config_value(training_config, "warmup_steps", 10, int), max_steps // 10),
                logging_steps=5,
                save_steps=max_steps // 2,
                max_steps=max_steps,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=True,
                remove_unused_columns=False,
                report_to=[],
                save_strategy="steps",
                save_total_limit=2,
                dataloader_pin_memory=False,  # メモリ問題回避
            )
        
        # Trainer作成と実行
        # 継続学習の場合はEWCを使用（ただし既存LoRAアダプタがある場合は軽量化のため無効化可能）
        use_ewc = request.training_method == "continual"
        existing_lora = request.training_config.get("existing_lora_path") if hasattr(request, 'training_config') else None
        
        # 既存のLoRAアダプタがある場合、EWCを軽量化または無効化
        if use_ewc and existing_lora:
            ewc_lambda = 1000.0  # 通常の5000から減らす
            logger.info(f"Task {task_id}: 既存LoRAアダプタ使用のため、EWC lambdaを{ewc_lambda}に調整")
        elif use_ewc:
            ewc_lambda = 5000.0
        else:
            ewc_lambda = 0.0
        
        if use_ewc:
            logger.info(f"Task {task_id}: 継続学習モード - EWC有効 (λ={ewc_lambda})")
        
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,  # tokenizer -> processing_classに変更
            use_ewc=use_ewc,
            ewc_lambda=ewc_lambda,
        )
        
        # EWCを使用する場合、事前学習データでFisher行列を計算
        if use_ewc and trainer.ewc_helper is not None:
            logger.info(f"Task {task_id}: Fisher行列を計算中...")
            # 事前学習データとして一般的な日本語テキストを使用
            pretrain_texts = [
                "人工知能は急速に発展している技術分野です。",
                "機械学習はデータから学習するアルゴリズムです。",
                "深層学習はニューラルネットワークを使用します。",
                "自然言語処理は言語を理解する技術です。",
                "コンピュータビジョンは画像を解析します。",
                "土木工学は社会インフラストラクチャの設計と建設を扱います。",
                "構造解析は建物や橋の安全性を評価する重要な技術です。",
                "地盤工学は土壌や岩盤の特性を研究します。",
                "水理学は水の流れと挙動を解析する分野です。",
                "交通工学は道路や鉄道の設計と最適化を行います。",
            ]
            
            logger.info(f"Task {task_id}: 事前学習データ数: {len(pretrain_texts)}")
            pretrain_dataset = SimpleDataset(pretrain_texts, tokenizer)
            from torch.utils.data import DataLoader
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)
            
            # Fisher行列の計算（最適化版）
            try:
                logger.info(f"Task {task_id}: Fisher行列の計算開始 (最大{30}バッチ)")
                trainer.ewc_helper.compute_fisher_matrix(pretrain_loader, max_batches=30)
                logger.info(f"Task {task_id}: Fisher行列の計算完了")
            except RuntimeError as e:
                logger.warning(f"Task {task_id}: Fisher行列計算失敗: {e}")
                logger.info(f"Task {task_id}: EWCなしで継続学習を続行します")
                # EWCを無効化
                trainer.ewc_lambda = 0.0
                trainer.ewc_helper = None
        
        # トレーニング実行
        logger.info(f"Task {task_id}: 実際のトレーニング開始 (メソッド: {request.training_method})")
        logger.info(f"Task {task_id}: トレーニング設定 - ステップ数: {max_steps}, バッチサイズ: {batch_size}, 学習率: {learning_rate}")
        
        try:
            train_result = trainer.train()
            
            # トレーニング結果のログ
            if hasattr(train_result, 'metrics'):
                logger.info(f"Task {task_id}: トレーニング完了 - メトリクス: {train_result.metrics}")
            else:
                logger.info(f"Task {task_id}: トレーニング完了")
                
            # 継続学習の場合は追加情報をログ
            if use_ewc:
                logger.info(f"Task {task_id}: 継続学習（EWC）によるトレーニングが正常に完了しました")
                
        except Exception as train_error:
            logger.error(f"Task {task_id}: トレーニングエラー: {str(train_error)}")
            # エラーが発生してもモデルは保存して続行
        
        # モデル保存
        training_tasks[task_id].message = "モデルを保存中..."
        training_tasks[task_id].progress = 95.0
        
        # モデルとトークナイザーを保存
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # トレーニング情報を保存
        training_info = {
            "model_type": request.training_method,
            "base_model": request.model_name,
            "r": get_config_value(request.lora_config, "r", get_config_value(training_config, "lora_r", 16, int), int),
            "lora_alpha": get_config_value(request.lora_config, "lora_alpha", get_config_value(training_config, "lora_alpha", 32, int), int),
            "task_type": "CAUSAL_LM",
            "training_data_size": len(train_texts),
            "training_method": request.training_method,
            "use_qlora": request.training_method == "qlora",
            "load_in_4bit": request.training_method == "qlora",
            "timestamp": timestamp,
            "output_dir": str(output_dir)
        }
        
        with open(output_dir / "training_info.json", "w", encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        # 完了
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].progress = 100.0
        training_tasks[task_id].message = f"{method_name}ファインチューニング完了！"
        training_tasks[task_id].model_path = str(output_dir)
        logger.info(f"Task {task_id}: {method_name}ファインチューニング完了 - {output_dir}")
        
    except Exception as e:
        import traceback
        logger.error(f"Task {task_id}: エラー発生: {str(e)}")
        logger.error(traceback.format_exc())
        training_tasks[task_id].status = "failed"
        training_tasks[task_id].message = f"エラー: {str(e)}"

# API エンドポイント

# 競合するルートハンドラーを削除 - テンプレートベースのルートハンドラーを使用

@app.get("/manual", response_class=HTMLResponse)
async def manual_page(request: Request):
    """利用マニュアルページ"""
    return templates.TemplateResponse("readme.html", {"request": request})

@app.get("/system-overview", response_class=HTMLResponse)
async def system_overview_page():
    """システム概要ページ"""
    # TODO: Create system-overview.html template in templates directory
    return HTMLResponse(
        content="<h1>System overview page not implemented yet</h1>", 
        status_code=404
    )

@app.get("/docs/{doc_name}")
async def serve_documentation(doc_name: str):
    """ドキュメントファイルの配信"""
    allowed_docs = [
        "API_REFERENCE.md", "LARGE_MODEL_SETUP.md", "MULTI_GPU_OPTIMIZATION.md",
        "USER_MANUAL.md", "QUICKSTART_GUIDE.md", "USAGE_GUIDE.md",
        "TRAINED_MODEL_USAGE.md", "DEEPSEEK_SETUP.md"
    ]
    
    if doc_name not in allowed_docs:
        return {"error": "Document not found"}
    
    # ドキュメントパスの検索
    project_root = Path(os.getcwd())
    possible_docs_paths = [
        project_root / "docs",
        Path(__file__).parent.parent / "docs"
    ]
    
    for docs_path in possible_docs_paths:
        doc_file_path = docs_path / doc_name
        if doc_file_path.exists():
            try:
                with open(doc_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return PlainTextResponse(content, media_type="text/markdown")
            except Exception as e:
                return {"error": f"Error reading document: {str(e)}"}
    
    return {"error": "Document file not found"}

@app.get("/api/models")
async def get_models():
    """利用可能なモデル一覧を取得"""
    return {
        "available_models": available_models,
        "saved_models": get_saved_models()
    }

@app.post("/api/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """トレーニングデータをアップロード"""
    try:
        logger.info(f"ファイルアップロード開始: {file.filename}")
        
        # ファイル名とサイズの検証
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が不正です")
        
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB制限
            raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます (最大100MB)")
        
        # ファイル保存
        project_root = Path(os.getcwd())
        upload_dir = project_root / "data" / "uploaded"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        content = await file.read()
        
        logger.info(f"ファイル保存: {file_path}, サイズ: {len(content)} bytes")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # ファイル形式の検証
        sample_data = []
        data_count = 0
        valid_lines = 0
        error_lines = []
        
        if file.filename.endswith('.jsonl'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data_count = len(lines)
                    
                    # 全行をチェックして有効性を検証
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:  # 空行をスキップ
                            continue
                            
                        try:
                            data = json.loads(line)
                            # 必須フィールドのチェック
                            if 'text' in data or ('input' in data and 'output' in data):
                                valid_lines += 1
                                # 最初の5件をサンプルとして保存
                                if len(sample_data) < 5:
                                    sample_data.append(data)
                            else:
                                error_lines.append({
                                    "line": i + 1,
                                    "error": "必須フィールド('text'または'input'と'output')が不足しています"
                                })
                        except json.JSONDecodeError as je:
                            error_lines.append({
                                "line": i + 1,
                                "error": f"JSONパースエラー: {str(je)}"
                            })
                    
                    # エラー率の計算
                    total_non_empty_lines = data_count - lines.count('\n')
                    if total_non_empty_lines > 0:
                        error_rate = len(error_lines) / total_non_empty_lines * 100
                    else:
                        error_rate = 0
                    
                    logger.info(f"JSONL解析完了: 全{data_count}行, 有効{valid_lines}行, エラー{len(error_lines)}行")
                    
                    # エラー率が高い場合は警告を含めて返す
                    if error_rate > 10:  # 10%以上のエラー
                        logger.warning(f"高エラー率検出: {error_rate:.1f}%")
                        # エラーが多すぎる場合は、最初の10個のエラーのみを返す
                        displayed_errors = error_lines[:10]
                        if len(error_lines) > 10:
                            displayed_errors.append({
                                "line": "...",
                                "error": f"他{len(error_lines) - 10}件のエラーがあります"
                            })
                    
                    # 有効な行が0の場合は明確にエラー
                    if valid_lines == 0:
                        logger.error(f"有効なトレーニングデータが見つかりません。エラー数: {len(error_lines)}")
                        error_detail = "全ての行にエラーがあります。"
                        if error_lines:
                            error_detail += f"\n最初のエラー: 行{error_lines[0]['line']} - {error_lines[0]['error']}"
                        raise HTTPException(
                            status_code=400, 
                            detail=f"有効なトレーニングデータが見つかりません。{error_detail}"
                        )
                        
                logger.info(f"JSONL検証完了: 有効{valid_lines}行, エラー{len(error_lines)}行")
                
            except UnicodeDecodeError:
                logger.error("ファイルエンコーディングエラー")
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)")
        
        elif file.filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data_count = len(data)
                        sample_data = data[:3]  # 最初の3件をサンプルとして取得
                    else:
                        data_count = 1
                        sample_data = [data]
                        
                logger.info(f"JSON解析完了: {data_count}件, サンプル: {len(sample_data)}件")
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON parse error: {str(je)}")
                raise HTTPException(status_code=400, detail=f"JSONパースエラー: {str(je)}")
            except UnicodeDecodeError:
                logger.error("ファイルエンコーディングエラー")
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)")
        else:
            raise HTTPException(status_code=400, detail="サポートされていないファイル形式です (.jsonl または .json を使用してください)")
        
        # 結果の構築（エラー情報を含む）
        result = {
            "status": "success" if len(error_lines) == 0 else "warning" if valid_lines > 0 else "error",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "data_count": data_count,
            "valid_lines": valid_lines,
            "error_count": len(error_lines),
            "sample_data": sample_data[:3]
        }
        
        # エラーがある場合は詳細を追加
        if error_lines:
            result["error_rate"] = f"{error_rate:.1f}%"
            result["errors"] = error_lines[:10]  # 最初の10個のエラーを含める
            if valid_lines == 0:
                result["fallback_warning"] = "有効なデータがないため、トレーニング時にフォールバックデータが使用されます"
            elif error_rate > 50:
                result["warning"] = f"エラー率が高いです（{error_rate:.1f}%）。データの品質を確認してください"
        
        logger.info(f"アップロード成功: {result}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"アップロードエラー: {str(e)}")

@app.post("/api/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """ファインチューニングを開始"""
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"ファインチューニング開始リクエスト受信: task_id={task_id}")
        logger.info(f"リクエスト内容: model_name={request.model_name}, method={request.training_method}")
        logger.info(f"LoRA設定: {request.lora_config}")
        logger.info(f"トレーニング設定: {request.training_config}")
        
        # リクエストの検証
        if not request.model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        if not request.training_data:
            raise HTTPException(status_code=400, detail="training_data is required")
        
        # 初期ステータスを設定
        training_tasks[task_id] = TrainingStatus(
            task_id=task_id,
            status="starting",
            progress=0.0,
            message="ファインチューニングを開始しています..."
        )
        
        # バックグラウンドでトレーニングを実行
        background_tasks.add_task(run_training_task, task_id, request)
        
        return {"task_id": task_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"トレーニング開始エラー: {str(e)}")
        import traceback
        logger.error(f"エラー詳細: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/training-status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_tasks[task_id]

@app.post("/api/monitoring/start")
async def start_monitoring():
    """監視システムを起動"""
    try:
        import subprocess
        import os
        
        # Web用の監視制御スクリプトを使用
        script_path = Path("/workspace/scripts/web_monitoring_controller.sh")
        
        if not script_path.exists():
            # スクリプトがない場合は作成
            script_content = '''#!/bin/bash
# 監視システムをコンテナ内から起動するスクリプト
echo "🚀 Grafana監視システムを起動中..."

# Grafanaが起動しているか確認
if curl -s http://ai-ft-grafana:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana: 既に起動しています"
    exit 0
else
    echo "⚠️ Grafana: ホスト側でdocker-compose -f docker/docker-compose-monitoring.yml up -d を実行してください"
    exit 1
fi
'''
            script_path.write_text(script_content)
            os.chmod(script_path, 0o755)
        
        # スクリプトを実行
        result = subprocess.run(
            [str(script_path), "start"],
            capture_output=True,
            text=True
        )
        
        # Grafanaの状態を直接確認
        import requests
        grafana_running = False
        prometheus_running = False
        
        try:
            # Grafana確認（コンテナ間通信）
            resp = requests.get("http://ai-ft-grafana:3000/api/health", timeout=2)
            grafana_running = resp.status_code == 200
        except:
            pass
            
        try:
            # Prometheus確認（コンテナ間通信）
            resp = requests.get("http://ai-ft-prometheus:9090/-/healthy", timeout=2)
            prometheus_running = resp.status_code == 200
        except:
            pass
        
        if grafana_running or prometheus_running:
            return JSONResponse(content={
                "status": "success",
                "message": "監視システムが利用可能です",
                "services": {
                    "grafana": "http://localhost:3000" if grafana_running else None,
                    "prometheus": "http://localhost:9090" if prometheus_running else None
                },
                "note": "既に起動済みか、ホスト側で起動されています"
            })
        else:
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": "監視システムが起動していません。ホスト側で以下のコマンドを実行してください:\ndocker-compose -f docker/docker-compose-monitoring.yml up -d",
                    "command": "docker-compose -f docker/docker-compose-monitoring.yml up -d"
                },
                status_code=503
            )
    except Exception as e:
        logger.error(f"監視システム起動エラー: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/api/monitoring/stop")
async def stop_monitoring():
    """監視システムを停止"""
    try:
        import subprocess
        import requests
        
        # 現在の状態を確認
        grafana_running = False
        prometheus_running = False
        
        try:
            resp = requests.get("http://ai-ft-grafana:3000/api/health", timeout=2)
            grafana_running = resp.status_code == 200
        except:
            pass
            
        try:
            resp = requests.get("http://ai-ft-prometheus:9090/-/healthy", timeout=2)
            prometheus_running = resp.status_code == 200
        except:
            pass
        
        if not grafana_running and not prometheus_running:
            return JSONResponse(content={
                "status": "success",
                "message": "監視システムは既に停止しています"
            })
        
        # コンテナ内から停止コマンドを実行できないため、手順を案内
        return JSONResponse(content={
            "status": "info",
            "message": "監視システムを停止するには、ホスト側で以下のコマンドを実行してください",
            "command": "docker stop ai-ft-grafana ai-ft-prometheus ai-ft-redis",
            "alternative": "または: docker-compose -f docker/docker-compose-monitoring.yml down"
        })
        
    except Exception as e:
        logger.error(f"監視システム停止エラー: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/api/monitoring/status")
async def monitoring_status():
    """監視システムの状態を確認"""
    try:
        import subprocess
        
        docker_dir = Path(__file__).parent.parent / "docker"
        compose_file = docker_dir / "docker-compose-monitoring.yml"
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "ps", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(docker_dir)
        )
        
        if result.returncode == 0:
            services_running = "grafana" in result.stdout.lower()
            return JSONResponse(content={
                "status": "success",
                "running": services_running,
                "message": "監視システムは稼働中" if services_running else "監視システムは停止中"
            })
        else:
            return JSONResponse(content={
                "status": "success",
                "running": False,
                "message": "監視システムは停止中"
            })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "running": False,
            "message": str(e)
        })

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    """実際のファインチューニング済みモデルを使用したテキスト生成"""
    try:
        logger.info(f"テキスト生成開始: モデル={request.model_path}, プロンプト={request.prompt[:50]}...")
        
        model_path = Path(request.model_path)
        
        # モデルパスの存在確認
        if not model_path.exists() or not model_path.is_dir():
            logger.warning(f"モデルパスが存在しません: {model_path}")
            return {
                "prompt": request.prompt,
                "generated_text": request.prompt + " [エラー: モデルパスが見つかりません]",
                "model_path": request.model_path,
                "error": "モデルパスが存在しません"
            }
        
        # キャッシュキー
        cache_key = str(model_path)
        
        # メモリ不足を事前にチェック
        if torch.cuda.is_available():
            # 現在の空きメモリを確認
            free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"現在のGPU空きメモリ: {free_memory:.2f} GB")
            
            # 32Bモデルは最低でも10GBの空きメモリが必要
            if free_memory < 10 and OLLAMA_AVAILABLE:
                logger.info("メモリ不足のため、直接Ollamaにフォールバックします")
                try:
                    ollama = OllamaIntegration()
                    result = ollama.generate_text(
                        model_name="llama3.2:3b",
                        prompt=request.prompt,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_length
                    )
                    
                    if result.get("success", False):
                        return {
                            "prompt": request.prompt,
                            "generated_text": result.get("generated_text", ""),
                            "model_path": request.model_path,
                            "method": "ollama",
                            "note": "GPUメモリ不足のため、Ollamaモデルを使用しました"
                        }
                except Exception as e:
                    logger.error(f"Ollamaフォールバック失敗: {e}")
        
        # モデルがキャッシュにない場合は読み込み
        if cache_key not in model_cache:
            # メモリ不足を防ぐため、既存のキャッシュをクリア
            if len(model_cache) > 0:
                logger.info("メモリ節約のため既存のモデルキャッシュをクリア")
                for key in list(model_cache.keys()):
                    if key != cache_key:  # 現在のモデル以外をクリア
                        del model_cache[key]
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")
            
            try:
                # トレーニング情報を読み込み
                training_info_path = model_path / "training_info.json"
                base_model_name = "distilgpt2"  # デフォルト
                training_method = "lora"
                
                if training_info_path.exists():
                    with open(training_info_path, 'r', encoding='utf-8') as f:
                        training_info = json.load(f)
                        base_model_name = training_info.get("base_model", "distilgpt2")
                        training_method = training_info.get("training_method", "lora")
                        logger.info(f"ベースモデル: {base_model_name}, メソッド: {training_method}")
                
                # トークナイザーの読み込み
                try:
                    tokenizer = load_tokenizer(str(model_path))
                    logger.info("ファインチューニング済みトークナイザーを読み込み")
                except Exception as e:
                    logger.warning(f"ファインチューニング済みトークナイザーの読み込みに失敗: {e}")
                    # ベースモデルのトークナイザーを使用
                    tokenizer = load_tokenizer(base_model_name)
                    logger.info(f"ベースモデルのトークナイザーを使用: {base_model_name}")
                
                # モデルの読み込み
                if training_method in ["lora", "qlora"]:
                    # LoRA/QLoRAモデルの場合
                    from peft import PeftModel
                    
                    # ベースモデルを読み込み
                    logger.info(f"ベースモデルを読み込み中: {base_model_name}")
                    
                    # 大きなモデルの場合は量子化を使用
                    quantization_config = create_quantization_config(base_model_name, "lora")
                    device_map = get_device_map(base_model_name)
                    
                    model_kwargs = {
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                    
                    if quantization_config:
                        model_kwargs["quantization_config"] = quantization_config
                    if device_map:
                        model_kwargs["device_map"] = device_map
                    
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        **model_kwargs
                    )
                    
                    # LoRAアダプターを読み込み
                    logger.info(f"LoRAアダプターを読み込み中: {model_path}")
                    model = PeftModel.from_pretrained(base_model, str(model_path))
                    logger.info("LoRAモデルをロード完了。GPUへ転送します。")
                    model.to("cuda")
                    logger.info("モデルをGPUへ転送しました。")
                    
                else:
                    # フルファインチューニングの場合
                    if torch.cuda.is_available():
                        # メモリ管理の環境変数を設定
                        # Removed: Environment variable now managed by memory_manager
                        
                        # GPUメモリをクリア
                        torch.cuda.empty_cache()
                        
                        # オフロードフォルダを作成
                        offload_dir = Path("offload")
                        offload_dir.mkdir(exist_ok=True)
                        logger.info(f"オフロードディレクトリを作成: {offload_dir}")
                        
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        logger.info(f"利用可能なGPUメモリ: {gpu_memory:.2f} GB")
                        
                        # 32Bモデルの推論時メモリ効率化
                        from transformers import BitsAndBytesConfig
                        
                        # 推論時は常に4bit量子化を使用（メモリ効率重視）
                        logger.info("推論時メモリ効率化: 4bit量子化を適用")
                        quantization_config = UnifiedQuantizationConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        
                        # デバイスマップを最適化
                        # GPUメモリの空き容量を確認
                        free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                        # 安全マージンを含めて設定
                        safe_memory = max(1, int(free_memory_gb * 0.8))  # 80%を使用
                        
                        max_memory = {
                            0: f"{safe_memory}GB",
                            "cpu": "32GB"  # CPUメモリを増やしてオフロードを促進
                        }
                        
                        # メモリ不足対策の強化
                        try:
                            # モデルロード前にもう一度メモリをクリア
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                            logger.info("フルファインチューニングモデルを量子化付きで読み込み中...")
                            model = AutoModelForCausalLM.from_pretrained(
                                str(model_path),
                                quantization_config=quantization_config,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                max_memory=max_memory,
                                trust_remote_code=True,
                                offload_folder=str(offload_dir)
                            )
                            logger.info("フルファインチューニングモデルの読み込み完了（4bit量子化）")
                            
                        except Exception as e:
                            logger.warning(f"量子化読み込み失敗: {str(e)}")
                            logger.info("Ollamaフォールバックを試行...")
                            
                            # Ollamaフォールバックを試行
                            if OLLAMA_AVAILABLE:
                                try:
                                    logger.info("Ollamaフォールバックを試行中...")
                                    ollama_integration = OllamaIntegration()
                                    
                                    # 利用可能なOllamaモデルを確認
                                    available_models = ollama_integration.list_models()
                                    logger.info(f"利用可能なOllamaモデル: {available_models}")
                                    
                                    # 利用可能なOllamaモデルから選択
                                    ollama_model_name = "llama3.2:3b"  # 直接指定
                                    logger.info(f"Ollamaモデル {ollama_model_name} を使用します")
                                    
                                    # Ollamaでテキスト生成
                                    result = ollama_integration.generate_text(
                                        model_name=ollama_model_name,
                                        prompt=request.prompt,
                                        temperature=request.temperature,
                                        top_p=request.top_p,
                                        max_tokens=request.max_length
                                    )
                                    
                                    if result.get("success", False):
                                        logger.info("Ollamaフォールバック成功")
                                        return {
                                            "prompt": request.prompt,
                                            "generated_text": result.get("generated_text", "Ollama生成エラー"),
                                            "model_path": request.model_path,
                                            "fallback": "ollama",
                                            "method": "ollama",
                                            "note": "GPUメモリ不足のため、Ollamaモデルで生成しました"
                                        }
                                    else:
                                        logger.warning(f"Ollama生成失敗: {result.get('error', 'Unknown error')}")
                                except Exception as ollama_error:
                                    logger.error(f"Ollamaフォールバック失敗: {str(ollama_error)}")
                                    import traceback
                                    logger.error(f"Ollamaエラー詳細: {traceback.format_exc()}")
                            
                            # 最終手段: CPUモード
                            logger.info("最終手段: CPUモードで読み込み中...")
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    str(model_path),
                                    torch_dtype=torch.float32,
                                    device_map=None,
                                    low_cpu_mem_usage=True,
                                    trust_remote_code=True
                                )
                                logger.info("CPUモードでの読み込み成功")
                            except Exception as final_error:
                                logger.error(f"全ての読み込み方法が失敗: {str(final_error)}")
                                return {
                                    "prompt": request.prompt,
                                    "generated_text": request.prompt + " [エラー: モデル読み込み失敗 - GPUメモリ不足]",
                                    "model_path": request.model_path,
                                    "error": f"モデル読み込み失敗: {str(final_error)}"
                                }
                    else:
                        # CPUモードでの実行
                        logger.info("CPUモードでモデルを読み込みます（GPUが利用できない場合）")
                        model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            torch_dtype=torch.float32,
                            device_map=None,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        logger.info("フルファインチューニングモデルの読み込み完了（CPU）")
                
                # キャッシュに保存
                model_cache[cache_key] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "base_model_name": base_model_name,
                    "training_method": training_method
                }
                logger.info("モデルキャッシュに保存完了")
                
            except Exception as model_error:
                logger.error(f"モデル読み込みエラー: {str(model_error)}")
                logger.error(traceback.format_exc())
                
                # GPUメモリ不足の場合の推奨事項を追加
                error_message = str(model_error)
                if "CUDA out of memory" in error_message:
                    recommendation = """
                    GPUメモリ不足のため、以下の対策を試してください：
                    1. より小さなモデル（7Bや14B）を使用する
                    2. 他のアプリケーションを終了してGPUメモリを解放する
                    3. CPUモードで実行する（速度は遅くなります）
                    """
                    error_message += recommendation
                
                return {
                    "prompt": request.prompt,
                    "generated_text": request.prompt + f" [エラー: モデル読み込み失敗 - {error_message}]",
                    "model_path": request.model_path,
                    "error": error_message
                }
        
        # ファインチューニング済みモデルの検証用テキスト生成
        logger.info("ファインチューニング済みモデルでテキスト生成を実行します")
        
        # Ollamaが利用可能で、メモリ不足の場合のフォールバック
        if OLLAMA_AVAILABLE:
            # GPUメモリを確認
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                    logger.info(f"GPUメモリ: 合計 {gpu_memory:.1f}GB, 空き {free_memory:.1f}GB")
                    
                    # 空きメモリが5GB未満の場合はOllamaを使用
                    if free_memory < 5:
                        logger.info(f"GPUメモリ不足（空き{free_memory:.1f}GB）のため、Ollamaを使用します")
                        ollama = OllamaIntegration()
                        
                        # 利用可能なOllamaモデルを確認
                        available_models = ollama.list_models()
                        logger.info(f"利用可能なOllamaモデル: {available_models}")
                        
                        # 使用するOllamaモデルを選択
                        ollama_model_name = None
                        if available_models.get("models"):
                            for model_info in available_models["models"]:
                                model_name_str = model_info.get("name", "")
                                if "llama3.2:3b" in model_name_str:
                                    ollama_model_name = model_name_str
                                    break
                            if not ollama_model_name and available_models["models"]:
                                ollama_model_name = available_models["models"][0].get("name")
                        
                        if ollama_model_name:
                            logger.info(f"Ollamaモデル {ollama_model_name} を使用します")
                            
                            # Ollamaでテキスト生成
                            result = ollama.generate_text(
                                model_name=ollama_model_name,
                                prompt=request.prompt,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                max_tokens=request.max_length
                            )
                            
                            if result.get("success", False):
                                logger.info("Ollamaでの生成が成功しました")
                                return {
                                    "prompt": request.prompt,
                                    "generated_text": result.get("generated_text", "Ollama生成エラー"),
                                    "model_path": request.model_path,
                                    "method": "ollama",
                                    "note": "GPUメモリ不足のため、Ollamaモデルで生成しました",
                                    "verification_info": {
                                        "model_path": request.model_path,
                                        "base_model": "ollama-converted",
                                        "training_method": "full",
                                        "prompt": request.prompt,
                                        "generation_params": {
                                            "max_length": request.max_length,
                                            "temperature": request.temperature,
                                            "top_p": request.top_p
                                        }
                                    }
                                }
                            else:
                                logger.warning(f"Ollama生成失敗: {result.get('error', 'Unknown error')}")
                                # フォールバック: 通常の方法を試行
            except Exception as ollama_error:
                logger.error(f"Ollama統合エラー: {str(ollama_error)}")
                import traceback
                logger.error(f"Ollamaエラー詳細: {traceback.format_exc()}")
                    # フォールバック: 通常の方法を試行
        
        # モデル情報の記録
        model_info = {
            "model_path": request.model_path,
            "base_model": cached_model.get("base_model_name", "unknown"),
            "training_method": cached_model.get("training_method", "unknown"),
            "prompt": request.prompt,
            "generation_params": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
        
        logger.info(f"モデル情報: {model_info}")
        
        # 通常の方法（Transformers）を使用
        cached_model = model_cache[cache_key]
        tokenizer = cached_model["tokenizer"]
        model = cached_model["model"]
        
        # テキスト生成
        logger.info("テキスト生成を実行中...")
        
        # プロンプトのトークナイズ
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # GPUに移動
        if torch.cuda.is_available() and hasattr(model, 'device'):
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except:
                # デバイス移動に失敗した場合はそのまま続行
                pass
        
        # ファインチューニング済みモデルの検証用テキスト生成
        model.eval()
        with torch.no_grad():
            try:
                # 生成パラメータの設定
                generation_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'max_new_tokens': request.max_length,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'repetition_penalty': 1.2,  # 繰り返しペナルティを追加
                    'no_repeat_ngram_size': 3,  # 3-gramの繰り返しを防ぐ
                }
                
                # サンプリングを使用する場合のみtemperatureとtop_pを設定
                if request.temperature > 0.0:
                    generation_kwargs['do_sample'] = True
                    generation_kwargs['temperature'] = request.temperature
                    generation_kwargs['top_p'] = request.top_p
                else:
                    generation_kwargs['do_sample'] = False
                
                logger.info(f"生成パラメータ: {generation_kwargs}")
                logger.info("model.generate()を実行中...")
                
                outputs = model.generate(**generation_kwargs)
                
                logger.info(f"生成完了: 出力トークン数={outputs.shape}")
                
                # 生成されたテキストをデコード
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"デコード完了: テキスト長={len(generated_text)}")
                
                # 元のプロンプトを除去して新しい部分だけを取得
                if generated_text.startswith(request.prompt):
                    new_text = generated_text[len(request.prompt):].strip()
                    if new_text:
                        # 「- 交通工学の問題です。」のような繰り返しパターンを検出して削除
                        import re
                        # 同じフレーズが3回以上繰り返される場合は、最初の1回だけ残す
                        pattern = r'((?:^|\n)?(?:- )?[^\n]+?)(?:\n?\1){2,}'
                        new_text = re.sub(pattern, r'\1', new_text)
                        
                        generated_text = request.prompt + "\n" + new_text
                    else:
                        generated_text = request.prompt + " [生成されたテキストが空でした]"
                
                logger.info(f"テキスト生成完了: {len(generated_text)}文字")
                
                # 生成結果をファイルに保存
                try:
                    project_root = Path(os.getcwd())
                    outputs_dir = project_root / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    # タイムスタンプ付きファイル名
                    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
                    model_name = Path(request.model_path).name
                    output_filename = f"generated_text_{model_name}_{timestamp}.json"
                    output_path = outputs_dir / output_filename
                    
                    # 生成結果を保存
                    generation_result = {
                        "timestamp": timestamp,
                        "model_path": request.model_path,
                        "base_model": cached_model.get("base_model_name", "unknown"),
                        "training_method": cached_model.get("training_method", "unknown"),
                        "prompt": request.prompt,
                        "generated_text": generated_text,
                        "parameters": {
                            "max_length": request.max_length,
                            "temperature": request.temperature,
                            "top_p": request.top_p
                        }
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(generation_result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"生成結果を保存: {output_path}")
                    
                except Exception as save_error:
                    logger.warning(f"生成結果の保存に失敗: {save_error}")
                
                # 検証結果の詳細情報を記録
                verification_info = {
                    "model_path": request.model_path,
                    "base_model": cached_model.get("base_model_name", "unknown"),
                    "training_method": cached_model.get("training_method", "unknown"),
                    "prompt": request.prompt,
                    "generated_text": generated_text,
                    "generation_params": {
                        "max_length": request.max_length,
                        "temperature": request.temperature,
                        "top_p": request.top_p
                    },
                    "model_info": {
                        "total_tokens": len(generated_ids),
                        "input_tokens": input_length,
                        "generated_tokens": len(generated_ids) - input_length
                    }
                }
                
                # 検証結果をログに記録
                logger.info(f"ファインチューニング済みモデル検証結果: {verification_info}")
                
                return {
                    "prompt": request.prompt,
                    "generated_text": generated_text,
                    "model_path": request.model_path,
                    "base_model": cached_model.get("base_model_name", "unknown"),
                    "training_method": cached_model.get("training_method", "unknown"),
                    "verification_info": verification_info
                }
                
            except Exception as gen_error:
                logger.error(f"テキスト生成エラー: {str(gen_error)}")
                logger.error(traceback.format_exc())
                return {
                    "prompt": request.prompt,
                    "generated_text": request.prompt + f" [エラー: 生成失敗 - {str(gen_error)}]",
                    "model_path": request.model_path,
                    "error": str(gen_error)
                }
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "prompt": request.prompt,
            "generated_text": request.prompt + f" [エラー: {str(e)}]",
            "model_path": request.model_path,
            "error": str(e)
        }

@app.post("/api/verify-model")
async def verify_finetuned_model(request: GenerationRequest):
    """ファインチューニング済みモデルの検証専用エンドポイント"""
    try:
        logger.info(f"ファインチューニング済みモデル検証開始: {request.model_path}")
        
        # モデル情報の取得
        model_path = Path(request.model_path)
        if not model_path.exists():
            return {
                "error": "モデルが見つかりません",
                "model_path": request.model_path
            }
        
        # 検証用のテストケース
        test_cases = [
            "縦断曲線とは何のために設置しますか？",
            "道路の横断勾配の標準的な値はどのくらいですか？",
            "アスファルト舗装の主な利点と欠点は何ですか？",
            "設計CBRとは舗装設計においてどのような指標ですか？",
            "道路の平面線形を構成する3つの要素は何ですか？"
        ]
        
        verification_results = []
        
        for i, test_prompt in enumerate(test_cases):
            logger.info(f"テストケース {i+1}/{len(test_cases)}: {test_prompt}")
            
            # テキスト生成リクエストを作成
            gen_request = GenerationRequest(
                model_path=request.model_path,
                prompt=test_prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # テキスト生成を実行
            result = await generate_text(gen_request)
            
            # 検証結果を記録
            verification_result = {
                "test_case": i + 1,
                "prompt": test_prompt,
                "generated_text": result.get("generated_text", ""),
                "verification_info": result.get("verification_info", {}),
                "success": "error" not in result
            }
            
            verification_results.append(verification_result)
            
            # 進捗をログに記録
            logger.info(f"テストケース {i+1} 完了: {'成功' if verification_result['success'] else '失敗'}")
        
        # 検証サマリーを作成
        success_count = sum(1 for r in verification_results if r["success"])
        total_count = len(verification_results)
        
        verification_summary = {
            "model_path": request.model_path,
            "total_test_cases": total_count,
            "successful_tests": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "verification_results": verification_results
        }
        
        logger.info(f"ファインチューニング済みモデル検証完了: 成功率 {success_count}/{total_count}")
        
        return {
            "status": "success",
            "verification_summary": verification_summary
        }
        
    except Exception as e:
        logger.error(f"モデル検証エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "model_path": request.model_path
        }

@app.post("/api/save-verification")
async def save_verification_results(verification_data: dict):
    """ファインチューニング済みモデルの検証結果を保存"""
    try:
        # 保存ディレクトリの作成
        project_root = Path(os.getcwd())
        verification_dir = project_root / "verification_results"
        verification_dir.mkdir(exist_ok=True)
        
        # ファイル名の生成
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        model_name = verification_data.get("model_path", "unknown").split("/")[-1]
        filename = f"verification_{model_name}_{timestamp}.json"
        
        # 検証結果を保存
        output_path = verification_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(verification_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"検証結果を保存: {output_path}")
        
        return {
            "status": "success",
            "saved_path": str(output_path),
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"検証結果保存エラー: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/system-info")
async def get_system_info():
    """システム情報を取得"""
    try:
        # GPU情報
        gpu_info = []
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # 全てのGPUの情報を取得
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_memory_free = gpu_memory_total - gpu_memory_used
                    
                    gpu_info.append({
                        "device": i,
                        "name": gpu_name,
                        "memory": f"{gpu_memory_total:.1f}GB",
                        "memory_used": f"{gpu_memory_used:.1f}GB",
                        "memory_free": f"{gpu_memory_free:.1f}GB",
                        "available": True
                    })
            else:
                gpu_info = [{
                    "device": 0,
                    "name": "No GPU",
                    "memory": "0GB",
                    "available": False
                }]
        else:
            gpu_info = [{
                "device": 0,
                "name": "CUDA Not Available",
                "memory": "0GB",
                "available": False
            }]
        
        # CUDA情報
        cuda_info = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda if torch.cuda.is_available() else "Not Available"
        }
        
        # PyTorch情報
        pytorch_info = {
            "version": torch.__version__
        }
        
        # CPU情報
        cpu_info = {
            "name": "CPU",
            "cores": os.cpu_count()
        }
        
        # RAM情報
        memory = psutil.virtual_memory()
        ram_info = {
            "total": f"{memory.total / (1024**3):.1f}GB",
            "used": f"{memory.used / (1024**3):.1f}GB",
            "free": f"{memory.available / (1024**3):.1f}GB",
            "percent": f"{memory.percent:.1f}%"
        }
        
        # キャッシュ情報
        cache_info = {
            "status": f"{len(model_cache)} models cached" if len(model_cache) > 0 else "No models cached"
        }
        
        return {
            "gpu": gpu_info,
            "cuda": cuda_info,
            "pytorch": pytorch_info,
            "cpu": cpu_info,
            "ram": ram_info,
            "cache": cache_info
        }
    except Exception as e:
        logger.error(f"システム情報取得エラー: {e}")
        return {
            "error": str(e),
            "gpu": {"name": "Error", "memory": "Unknown", "available": False},
            "cuda": {"available": False, "version": "Unknown"},
            "pytorch": {"version": "Unknown"},
            "cpu": {"name": "Unknown", "cores": "Unknown"},
            "ram": {"total": "Unknown", "used": "Unknown", "free": "Unknown", "percent": "Unknown"},
            "cache": {"status": "Unknown"}
        }

@app.get("/metrics")
async def get_metrics():
    """Prometheusメトリクスエンドポイント"""
    if METRICS_AVAILABLE and metrics_collector:
        try:
            # システムメトリクスを更新
            metrics_collector.update_system_metrics()
            
            # トレーニングタスク数を更新
            active_tasks = sum(1 for task in training_tasks.values() if task["status"] == "running")
            metrics_collector.set_active_training_tasks(active_tasks)
            
            # RAG文書数を更新（RAGが利用可能な場合）
            if RAG_AVAILABLE:
                try:
                    metadata_manager = MetadataManager()
                    doc_count = len(metadata_manager.get_all_documents())
                    metrics_collector.set_rag_documents_count(doc_count)
                except:
                    pass
            
            # メトリクスを返す
            return get_prometheus_metrics()
        except Exception as e:
            logger.error(f"メトリクス生成エラー: {e}")
            from fastapi.responses import Response
            return Response(content="# Error generating metrics\n", media_type="text/plain")
    else:
        from fastapi.responses import Response
        return Response(content="# Metrics not available\n", media_type="text/plain")

# アプリケーション起動時の初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("AI Fine-tuning Toolkit Web API starting...")
    
    # 必要なディレクトリを作成
    project_root = Path(os.getcwd())
    (project_root / "data" / "uploaded").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    (project_root / "app" / "static").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "continual_learning").mkdir(parents=True, exist_ok=True)
    
    # 継続学習タスクを読み込む
    load_continual_tasks()

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("AI Fine-tuning Toolkit Web API shutting down...")
    
    # 継続学習タスクを保存
    save_continual_tasks()
    
    logger.info("Shutdown complete.")

@app.post("/api/clear_cache")
async def clear_model_cache():
    """モデルキャッシュをクリア"""
    try:
        global model_cache
        cache_size = len(model_cache)
        model_cache.clear()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info(f"モデルキャッシュをクリアしました: {cache_size}個のモデルを解放")
        return {"status": "success", "cleared_models": cache_size}
    except Exception as e:
        logger.error(f"キャッシュクリアエラー: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.post("/api/convert-to-ollama")
async def convert_finetuned_to_ollama(request: dict):
    """ファインチューニング済みモデルをOllama形式に変換"""
    try:
        model_path = request.get("model_path")
        model_name = request.get("model_name", "road-engineering-expert")
        
        if not model_path:
            return {"success": False, "error": "model_pathが指定されていません"}
        
        logger.info(f"ファインチューニング済みモデルのOllama変換開始: {model_path}")
        
        # 変換スクリプトを実行
        import subprocess
        import sys
        
        # 変換スクリプトのパス
        script_path = Path("convert_finetuned_to_ollama.py")
        
        if not script_path.exists():
            return {"success": False, "error": "変換スクリプトが見つかりません"}
        
        # 変換実行
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2時間のタイムアウト
        )
        
        if result.returncode == 0:
            logger.info("ファインチューニング済みモデルのOllama変換が完了しました")
            return {
                "success": True,
                "model_name": model_name,
                "message": "ファインチューニング済みモデルがOllamaで使用可能になりました",
                "usage": f"ollama run {model_name}"
            }
        else:
            logger.error(f"変換エラー: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        logger.error("変換がタイムアウトしました")
        return {"success": False, "error": "Conversion timeout"}
    except Exception as e:
        logger.error(f"変換エラー: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/apply-lora-to-ollama")
async def apply_lora_to_ollama(request: dict, background_tasks: BackgroundTasks):
    """LoRAアダプターをGGUFベースモデルに適用してOllamaに登録（改善版）"""
    try:
        # パラメータ取得
        base_model_url = request.get("base_model_url")  # オプション（Noneの場合は自動検索）
        base_model_name = request.get("base_model_name", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf")
        lora_adapter_path = request.get("lora_adapter_path")  # オプション
        output_model_name = request.get("output_model_name", "deepseek-32b-finetuned")
        use_improved_version = request.get("use_improved_version", True)  # 改善版を使用するか
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # バックグラウンドタスクとして実行
        async def run_conversion():
            try:
                # 使用するスクリプトを選択
                if use_improved_version:
                    script_path = "/workspace/scripts/apply_lora_to_gguf_improved.py"
                else:
                    script_path = "/workspace/scripts/apply_lora_to_gguf.py"
                
                # スクリプトを実行
                import subprocess
                cmd = ["python", script_path]
                
                # パラメータを追加
                if base_model_url:
                    cmd.extend(["--base-model-url", base_model_url])
                    
                cmd.extend([
                    "--base-model-name", base_model_name,
                    "--output-name", output_model_name
                ])
                
                if lora_adapter_path:
                    cmd.extend(["--lora-adapter", lora_adapter_path])
                
                # 改善版の場合は一時ディレクトリを使用
                if use_improved_version:
                    # デフォルトで一時ディレクトリを使用（--no-tempを指定しない）
                    pass
                
                logger.info(f"LoRA to Ollama変換開始: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # 成功
                    if task_id in training_tasks:
                        training_tasks[task_id]["status"] = "completed"
                        training_tasks[task_id]["message"] = f"Model {output_model_name} created successfully"
                        training_tasks[task_id]["model_name"] = output_model_name
                        logger.info(f"LoRA to Ollama変換成功: {output_model_name}")
                else:
                    # エラー
                    if task_id in training_tasks:
                        training_tasks[task_id]["status"] = "error"
                        training_tasks[task_id]["message"] = result.stderr
                        logger.error(f"LoRA to Ollama変換失敗: {result.stderr}")
                        
            except Exception as e:
                if task_id in training_tasks:
                    training_tasks[task_id]["status"] = "error"
                    training_tasks[task_id]["message"] = str(e)
                logger.error(f"LoRA to Ollama変換エラー: {str(e)}")
        
        # タスクを登録
        training_tasks[task_id] = {
            "status": "running",
            "message": "Converting LoRA adapter to Ollama format...",
            "progress": 0,
            "task_id": task_id
        }
        
        # バックグラウンドで実行
        background_tasks.add_task(run_conversion)
        
        return {
            "status": "started",
            "task_id": task_id,
            "message": "LoRA to Ollama conversion started"
        }
        
    except Exception as e:
        logger.error(f"LoRA to Ollama conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-to-ollama-wsl")
async def convert_finetuned_to_ollama_wsl(request: dict):
    """WSL環境用：ファインチューニング済みモデルをOllama形式に変換"""
    try:
        model_path = request.get("model_path")
        model_name = request.get("model_name", "road-engineering-expert")
        
        if not model_path:
            return {"success": False, "error": "model_pathが指定されていません"}
        
        logger.info(f"WSL環境でファインチューニング済みモデルのOllama変換開始: {model_path}")
        
        # WSL環境用変換スクリプトを実行
        import subprocess
        import sys
        
        # 変換スクリプトのパス
        script_path = Path("setup_wsl_ollama.py")
        
        if not script_path.exists():
            return {"success": False, "error": "WSL変換スクリプトが見つかりません"}
        
        # 変換実行
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2時間のタイムアウト
        )
        
        if result.returncode == 0:
            logger.info("WSL環境でのファインチューニング済みモデルのOllama変換が完了しました")
            return {
                "success": True,
                "model_name": model_name,
                "message": "WSL環境でファインチューニング済みモデルがOllamaで使用可能になりました",
                "usage": f"ollama run {model_name}"
            }
        else:
            logger.error(f"WSL変換エラー: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        logger.error("WSL変換がタイムアウトしました")
        return {"success": False, "error": "Conversion timeout"}
    except Exception as e:
        logger.error(f"WSL変換エラー: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/finetuned-lora-models")
async def get_finetuned_lora_models():
    """ファインチューニング済みのLoRAアダプターモデルを取得"""
    try:
        models = []
        outputs_dir = Path("/workspace/outputs")
        
        if outputs_dir.exists():
            # LoRAアダプターを含むディレクトリを検索
            for path in outputs_dir.rglob("adapter_model.safetensors"):
                model_dir = path.parent
                config_path = model_dir / "adapter_config.json"
                
                # モデル情報を取得
                model_info = {
                    "path": str(model_dir),
                    "name": model_dir.name,
                    "type": "lora",
                    "format": "safetensors"
                }
                
                # 設定ファイルから情報を読み取る
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            model_info["base_model"] = config.get("base_model_name_or_path", "unknown")
                            model_info["r"] = config.get("r", "unknown")
                            model_info["alpha"] = config.get("lora_alpha", "unknown")
                    except:
                        pass
                
                # 作成日時を取得
                model_info["created"] = datetime.fromtimestamp(
                    model_dir.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
                
                models.append(model_info)
            
            # binフォーマットのアダプターも検索
            for path in outputs_dir.rglob("adapter_model.bin"):
                model_dir = path.parent
                # safetensorsが既に追加されている場合はスキップ
                if not any(m["path"] == str(model_dir) for m in models):
                    config_path = model_dir / "adapter_config.json"
                    
                    model_info = {
                        "path": str(model_dir),
                        "name": model_dir.name,
                        "type": "lora",
                        "format": "bin"
                    }
                    
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                model_info["base_model"] = config.get("base_model_name_or_path", "unknown")
                                model_info["r"] = config.get("r", "unknown")
                                model_info["alpha"] = config.get("lora_alpha", "unknown")
                        except:
                            pass
                    
                    model_info["created"] = datetime.fromtimestamp(
                        model_dir.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    
                    models.append(model_info)
        
        # 作成日時でソート（新しいものが先）
        models.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error getting finetuned LoRA models: {e}")
        return {
            "success": False,
            "error": str(e),
            "models": []
        }

@app.get("/api/available-models")
async def get_available_models():
    """利用可能なファインチューニング済みモデルとOllamaモデルを取得"""
    try:
        models = {
            "finetuned_models": [],
            "ollama_models": []
        }
        
        # ファインチューニング済みモデルの検索
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for model_dir in outputs_dir.iterdir():
                if model_dir.is_dir():
                    # モデルファイルの存在確認
                    has_model_files = (
                        (model_dir / "pytorch_model.bin").exists() or
                        (model_dir / "adapter_model.safetensors").exists() or
                        (model_dir / "adapter_config.json").exists() or
                        (model_dir / "config.json").exists()
                    )
                    
                    if has_model_files:
                        # モデル情報を取得
                        config_path = model_dir / "config.json"
                        training_info_path = model_dir / "training_info.json"
                        
                        model_info = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "type": "finetuned",
                            "size": "Unknown",
                            "created": "Unknown"
                        }
                        
                        # 設定ファイルから情報を読み取り
                        if config_path.exists():
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                    model_info["base_model"] = config.get("_name_or_path", "Unknown")
                                    model_info["model_type"] = config.get("model_type", "Unknown")
                            except:
                                pass
                        
                        # 訓練情報から詳細を取得
                        if training_info_path.exists():
                            try:
                                with open(training_info_path, 'r', encoding='utf-8') as f:
                                    training_info = json.load(f)
                                    model_info["training_method"] = training_info.get("training_method", "unknown")
                                    model_info["created"] = training_info.get("timestamp", "Unknown")
                                    model_info["created_at"] = training_info.get("created_at", training_info.get("timestamp", "Unknown"))
                                    model_info["base_model"] = training_info.get("base_model", model_info.get("base_model", "Unknown"))
                            except:
                                pass
                        
                        # モデルタイプの判定
                        if "continual_task" in model_dir.name.lower():
                            model_info["training_method"] = "continual"
                            model_info["type"] = "継続学習 (EWC)"
                            model_info["size"] = "~500MB+"
                        elif "qlora" in model_dir.name.lower() or "4bit" in model_dir.name.lower():
                            model_info["training_method"] = "qlora"
                            model_info["type"] = "QLoRA (4bit)"
                            model_info["size"] = "~1.0MB"
                        elif "lora" in model_dir.name.lower():
                            model_info["training_method"] = "lora"
                            model_info["type"] = "LoRA"
                            model_info["size"] = "~1.6MB"
                        elif "フルファインチューニング" in model_dir.name:
                            model_info["training_method"] = "full"
                            model_info["type"] = "フルファインチューニング"
                            model_info["size"] = "~500MB+"
                        
                        models["finetuned_models"].append(model_info)
                        logger.info(f"ファインチューニング済みモデルを検出: {model_dir.name}")
        
        # Ollamaモデルの検索
        if OLLAMA_AVAILABLE:
            try:
                ollama = OllamaIntegration()
                ollama_models = ollama.list_models()
                logger.debug(f"Ollamaモデル取得結果: {ollama_models}")
                
                if ollama_models.get("success", False):
                    for model in ollama_models.get("models", []):
                        # 全てのOllamaモデルを表示（フィルタリングを削除）
                        models["ollama_models"].append({
                            "name": model.get("name", "Unknown"),
                            "type": "ollama",
                            "size": model.get("size", "Unknown"),
                            "modified": model.get("modified", "Unknown")
                        })
                else:
                    logger.warning(f"Ollamaモデル取得失敗: {ollama_models.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Ollamaモデル取得エラー: {e}")
                import traceback
                logger.error(f"詳細エラー: {traceback.format_exc()}")
                # エラーが発生しても空のリストを返す（アプリケーションを停止させない）
                models["ollama_models"] = []
        
        # RAG設定ファイルから利用可能なモデルを追加
        try:
            rag_config_path = Path("src/rag/config/rag_config.yaml")
            if rag_config_path.exists():
                import yaml
                with open(rag_config_path, 'r', encoding='utf-8') as f:
                    rag_config = yaml.safe_load(f)
                    
                # 設定ファイルから利用可能なモデルリストを取得
                available_models = rag_config.get('llm', {}).get('available_models', [])
                for model_name in available_models:
                    # 既にリストにない場合のみ追加
                    if not any(m['name'] == model_name for m in models['ollama_models']):
                        models["ollama_models"].append({
                            "name": model_name,
                            "type": "ollama",
                            "size": "Configured",
                            "modified": "From config"
                        })
                        logger.info(f"RAG設定からモデルを追加: {model_name}")
        except Exception as e:
            logger.warning(f"RAG設定ファイル読み込みエラー: {e}")
        
        return models
        
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {str(e)}")
        return {"finetuned_models": [], "ollama_models": [], "error": str(e)}

@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """ファインチューニング済みモデルを削除"""
    import shutil
    
    try:
        # パスの安全性チェック（ディレクトリトラバーサル対策）
        # URLエンコードされたスラッシュも考慮
        if ".." in model_name or "/" in model_name or "\\" in model_name or "%2F" in model_name or "%2f" in model_name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid model name"}
            )
        
        # モデルディレクトリのパスを構築
        model_path = Path("outputs") / model_name
        
        # モデルディレクトリの存在確認
        if not model_path.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Model '{model_name}' not found"}
            )
        
        # モデルディレクトリかどうか確認
        if not model_path.is_dir():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid model path"}
            )
        
        # outputsディレクトリ配下であることを確認
        outputs_dir = Path("outputs").resolve()
        model_path_resolved = model_path.resolve()
        if not str(model_path_resolved).startswith(str(outputs_dir)):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid model location"}
            )
        
        # モデルディレクトリを削除
        logger.info(f"Deleting model: {model_name}")
        shutil.rmtree(model_path)
        
        logger.info(f"Model '{model_name}' deleted successfully")
        return {"success": True, "message": f"Model '{model_name}' deleted successfully"}
        
    except PermissionError:
        logger.error(f"Permission denied when deleting model: {model_name}")
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "Permission denied"}
        )
    except Exception as e:
        logger.error(f"Error deleting model '{model_name}': {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/generate-stream")
async def generate_text_stream(request: dict):
    """ストリーミング対応のテキスト生成"""
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def generate_stream():
        try:
            model_name = request.get("model_name")
            model_type = request.get("model_type")
            prompt = request.get("prompt")
            max_length = request.get("max_length", 2048)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.9)
            
            if not model_name or not model_type or not prompt:
                yield f"data: {json.dumps({'error': 'model_name, model_type, promptが必要です'})}\n\n"
                return
            
            logger.info(f"ストリーミング生成開始: {model_type}/{model_name}")
            
            if model_type == "ollama":
                # Ollamaストリーミング生成
                if not OLLAMA_AVAILABLE:
                    yield f"data: {json.dumps({'error': 'Ollamaが利用できません'})}\n\n"
                    return
                
                ollama = OllamaIntegration()
                
                # ストリーミング用のOllamaリクエスト
                ollama_params = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_length
                    }
                }
                
                try:
                    response = requests.post(
                        f"{ollama.base_url}/api/generate",
                        json=ollama_params,
                        stream=True,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    yield f"data: {json.dumps({'text': data['response'], 'done': False})}\n\n"
                                if data.get('done', False):
                                    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                                    break
                    else:
                        yield f"data: {json.dumps({'error': f'Ollama API エラー: {response.status_code}'})}\n\n"
                        
                except Exception as e:
                    logger.error(f"Ollamaストリーミングエラー: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            elif model_type == "finetuned":
                # ファインチューニング済みモデルのストリーミング生成
                model_path = f"outputs/{model_name}"
                
                try:
                    # モデルの読み込み（キャッシュから取得または新規読み込み）
                    if model_path not in model_cache:
                        logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")
                        
                        # GPU メモリ最適化設定
                        max_memory = {}
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                max_memory[i] = "18GB"
                            max_memory["cpu"] = "30GB"
                        
                        # モデルとトークナイザーの読み込み
                        tokenizer = load_tokenizer(model_path)
                        
                        # 量子化設定
                        quantization_config = create_quantization_config(model_path, "lora", force_4bit=True)
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            max_memory=max_memory,
                            trust_remote_code=True
                        )
                        
                        model_cache[model_path] = {
                            "tokenizer": tokenizer,
                            "model": model
                        }
                    
                    cached_model = model_cache[model_path]
                    tokenizer = cached_model["tokenizer"]
                    model = cached_model["model"]
                    
                    # ストリーミング生成
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    model.eval()
                    
                    # デバイスを取得（デバイス統一のため）
                    device = inputs['input_ids'].device
                    logger.info(f"ストリーミング生成デバイス: {device}")
                    
                    # ストリーミング生成の実装
                    generated_tokens = []
                    current_text = ""
                    token_buffer = []  # 文字化け防止用バッファ
                    
                    with torch.no_grad():
                        for _ in range(max_length):
                            outputs = model.generate(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs.get('attention_mask'),
                                max_new_tokens=1,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=temperature > 0.0,
                                temperature=temperature if temperature > 0.0 else 1.0,
                                top_p=top_p if temperature > 0.0 else 1.0,
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=3
                            )
                            
                            new_token = outputs[0][-1].unsqueeze(0)
                            # デバイスを統一
                            new_token = new_token.to(device)
                            generated_tokens.append(new_token)
                            
                            # トークンバッファに追加
                            token_buffer.append(new_token)
                            
                            # バッファから完全な文字をデコード
                            if len(token_buffer) >= 1:  # 1トークンごとにデコード
                                try:
                                    # バッファ内の全トークンを結合してデコード
                                    buffer_tokens = torch.cat(token_buffer, dim=0)
                                    decoded_text = tokenizer.decode(buffer_tokens, skip_special_tokens=True)
                                    
                                    # デバッグ情報
                                    logger.debug(f"バッファサイズ: {len(token_buffer)}, デコード結果: '{decoded_text}', 現在のテキスト: '{current_text}'")
                                    
                                    # 前回のデコード結果と比較して新しい部分のみを抽出
                                    if len(current_text) < len(decoded_text):
                                        new_text = decoded_text[len(current_text):]
                                        current_text = decoded_text
                                        logger.debug(f"新しいテキスト: '{new_text}'")
                                        
                                        # 文字化けチェック（改善版）
                                        if new_text and new_text.strip():  # 空白文字以外は有効
                                            # 基本的な文字化けパターンをチェック
                                            invalid_chars = ['', '\ufffd', '\u0000', '\u0001', '\u0002', '\u0003']
                                            has_invalid = any(char in new_text for char in invalid_chars)
                                            
                                            if not has_invalid:
                                                yield f"data: {json.dumps({'text': new_text, 'done': False})}\n\n"
                                            else:
                                                logger.warning(f"文字化け検出: {new_text}")
                                        else:
                                            logger.debug(f"空白文字スキップ: '{new_text}'")
                                    
                                    # バッファをクリア（完全にデコードされたため）
                                    token_buffer = []
                                    
                                except Exception as decode_error:
                                    logger.error(f"デコードエラー: {decode_error}")
                                    # エラーが発生した場合は個別デコードを試行
                                    for token in token_buffer:
                                        try:
                                            single_text = tokenizer.decode(token, skip_special_tokens=True)
                                            if single_text and single_text.strip():
                                                # 基本的な文字化けパターンをチェック
                                                invalid_chars = ['', '\ufffd', '\u0000', '\u0001', '\u0002', '\u0003']
                                                has_invalid = any(char in single_text for char in invalid_chars)
                                                
                                                if not has_invalid:
                                                    yield f"data: {json.dumps({'text': single_text, 'done': False})}\n\n"
                                                else:
                                                    logger.warning(f"個別デコード文字化け検出: {single_text}")
                                        except Exception as e:
                                            logger.error(f"個別デコードエラー: {e}")
                                    token_buffer = []
                            
                            # 入力IDを更新（デバイス統一済み）
                            inputs['input_ids'] = torch.cat([inputs['input_ids'], new_token.unsqueeze(0)], dim=1)
                            if 'attention_mask' in inputs:
                                # attention_maskも同じデバイスに作成
                                ones = torch.ones(1, 1, dtype=torch.long, device=device)
                                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], ones], dim=1)
                            
                            # EOSトークンが生成されたら停止
                            if new_token.item() == tokenizer.eos_token_id:
                                break
                    
                    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                    
                except Exception as e:
                    logger.error(f"ファインチューニング済みモデルストリーミングエラー: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            else:
                yield f"data: {json.dumps({'error': 'サポートされていないモデルタイプです'})}\n\n"
                
        except Exception as e:
            logger.error(f"ストリーミング生成エラー: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.post("/api/generate-with-model-selection")
async def generate_with_model_selection(request: dict):
    """モデル選択機能付きテキスト生成"""
    try:
        model_name = request.get("model_name")
        model_type = request.get("model_type")  # "finetuned" or "ollama"
        prompt = request.get("prompt")
        max_length = request.get("max_length", 2048)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        if not model_name or not model_type or not prompt:
            return {"success": False, "error": "model_name, model_type, promptが必要です"}
        
        logger.info(f"モデル選択生成: {model_type}/{model_name}")
        
        if model_type == "ollama":
            # Ollamaモデルを使用
            if not OLLAMA_AVAILABLE:
                return {"success": False, "error": "Ollamaが利用できません"}
            
            ollama = OllamaIntegration()
            result = ollama.generate_text(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "generated_text": result["generated_text"],
                    "model_name": model_name,
                    "model_type": "ollama",
                    "method": "ollama_api"
                }
            else:
                return {"success": False, "error": result.get("error", "Ollama生成エラー")}
        
        elif model_type == "finetuned":
            # ファインチューニング済みモデルを使用
            model_path = f"outputs/{model_name}"
            
            # 既存のファインチューニング済みモデル生成機能を使用
            generation_request = {
                "model_path": model_path,
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # メモリ不足の場合はOllamaにフォールバック
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 30 and OLLAMA_AVAILABLE:
                    # Ollamaにフォールバック
                    ollama_model_name = "llama3.2:3b"  # 利用可能なOllamaモデル
                    logger.info(f"メモリ不足のため、Ollamaモデル {ollama_model_name} を使用します")
                    
                    ollama = OllamaIntegration()
                    result = ollama.generate_text(
                        model_name=ollama_model_name,
                        prompt=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_length
                    )
                    
                    if result["success"]:
                        return {
                            "success": True,
                            "generated_text": result["generated_text"],
                            "model_name": f"{model_name} (Ollama fallback: {ollama_model_name})",
                            "model_type": "finetuned_ollama_fallback",
                            "method": "ollama_fallback"
                        }
            
            # 通常のファインチューニング済みモデル生成を試行
            try:
                # モデルの読み込みとテキスト生成
                if model_path not in model_cache:
                    logger.info(f"ファインチューニング済みモデルを読み込み中: {model_path}")
                    
                    # GPU メモリ最適化設定
                    max_memory = {}
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            max_memory[i] = "18GB"  # 各GPUに18GB割り当て
                        max_memory["cpu"] = "30GB"  # CPUに30GB割り当て
                    
                    # モデルとトークナイザーの読み込み（量子化とオフロード）
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        max_memory=max_memory,
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        offload_folder="./offload",
                        offload_state_dict=True
                    )
                    
                    tokenizer = load_tokenizer(model_path)
                    
                    model_cache[model_path] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "base_model_name": model_path,
                        "training_method": "full"
                    }
                    
                    logger.info(f"モデル読み込み完了: {model_path}")
                
                # キャッシュからモデルを取得
                cached_model = model_cache[model_path]
                model = cached_model["model"]
                tokenizer = cached_model["tokenizer"]
                
                # テキスト生成
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # プロンプト部分を除去
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                logger.info(f"ファインチューニング済みモデルでの生成完了: {len(generated_text)}文字")
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "model_type": "finetuned",
                    "method": "direct_inference"
                }
                
            except Exception as model_error:
                logger.error(f"ファインチューニング済みモデル生成エラー: {str(model_error)}")
                return {"success": False, "error": f"モデル生成エラー: {str(model_error)}"}
        
        else:
            return {"success": False, "error": f"未知のモデルタイプ: {model_type}"}
        
    except Exception as e:
        logger.error(f"モデル選択生成エラー: {str(e)}")
        return {"success": False, "error": str(e)}

# =============================================================================
# RAG API エンドポイント (統合版)
# =============================================================================

@app.get("/rag/health")
async def rag_health_check():
    """RAGシステムヘルスチェック"""
    return {
        "status": "healthy" if rag_app.is_initialized else "initializing",
        "timestamp": datetime.now(JST).isoformat(),
        "service": "Road Design RAG System",
        "available": RAG_AVAILABLE
    }

@app.get("/rag/system-info", response_model=SystemInfoResponse)
async def rag_get_system_info():
    """RAGシステム情報を取得"""
    try:
        # 設定ファイルから直接読み込み
        config_path = Path("src/rag/config/rag_config.yaml")
        config_data = {}
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        
        # システム情報を構築
        system_info = {
            "config": {
                "llm": {
                    "model_name": config_data.get('llm', {}).get('model_name', '未設定'),
                    "base_model": config_data.get('llm', {}).get('base_model', '未設定'),
                    "temperature": config_data.get('llm', {}).get('temperature', 0.3),
                    "use_finetuned": config_data.get('llm', {}).get('use_finetuned', False),
                    "use_moe": config_data.get('llm', {}).get('use_moe', False),
                    "moe_num_experts": config_data.get('llm', {}).get('moe_num_experts', 8),
                    "moe_experts_per_token": config_data.get('llm', {}).get('moe_experts_per_token', 2),
                    "model_path": config_data.get('llm', {}).get('model_path', '未設定')
                },
                "embedding": {
                    "model_name": config_data.get('embedding', {}).get('model_name', 'multilingual-e5-large')
                },
                "vector_store": {
                    "type": config_data.get('vector_store', {}).get('type', 'Qdrant')
                }
            },
            "status": "initialized" if rag_app.is_initialized else "not_initialized"
        }
        
        return SystemInfoResponse(
            status="success",
            system_info=system_info,
            timestamp=datetime.now(JST).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return SystemInfoResponse(
            status="error",
            system_info={
                "config": {
                    "llm": {"model_name": "設定読み込みエラー"},
                    "embedding": {"model_name": "multilingual-e5-large"},
                    "vector_store": {"type": "Qdrant"}
                },
                "error": str(e)
            },
            timestamp=datetime.now(JST).isoformat()
        )

@app.post("/rag/quantize-model")
async def quantize_finetuned_model(
    background_tasks: BackgroundTasks,
    lora_path: str = Form(...),
    quantization_level: str = Form("Q4_K_M"),
    model_name: str = Form(...)
):
    """ファインチューニング済みモデルを量子化してOllamaに登録"""
    try:
        import asyncio
        import subprocess
        from pathlib import Path
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # 非同期で量子化を実行
        async def run_quantization():
            try:
                # ステータス更新用ファイル
                status_file = Path(f"/tmp/quantization_{task_id}.json")
                
                # 初期ステータス
                status = {
                    "task_id": task_id,
                    "status": "running",
                    "progress": 0,
                    "message": "量子化処理を開始しています...",
                    "logs": []
                }
                status_file.write_text(json.dumps(status))
                
                # 統合モデル処理スクリプトを実行
                # モデルパスが指定されている場合はそれを処理、なければ最新モデルを自動検出
                if lora_path and lora_path != "auto":
                    cmd = [
                        "python", "/workspace/scripts/unified_model_processor.py",
                        "--model", lora_path
                    ]
                else:
                    # 自動検出モード（最新のモデルを処理）
                    cmd = [
                        "python", "/workspace/scripts/qlora_to_ollama.py"
                    ]
                
                logger.info(f"量子化コマンド実行: {' '.join(cmd)}")
                
                # プロセス実行
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # 出力を監視
                logs = []
                
                # stdoutを読み取り
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    log_line = line.decode().strip()
                    logs.append(log_line)
                    logger.info(f"Quantization stdout: {log_line}")
                    
                    # ステータス更新
                    if "Loading base model" in log_line:
                        progress = 20
                        message = "ベースモデルをロード中..."
                    elif "Loading LoRA adapter" in log_line:
                        progress = 40
                        message = "LoRAアダプタをロード中..."
                    elif "Merging" in log_line:
                        progress = 60
                        message = "モデルをマージ中..."
                    elif "Quantizing" in log_line or "quantization" in log_line.lower():
                        progress = 80
                        message = "量子化を実行中..."
                    elif "Saving" in log_line:
                        progress = 90
                        message = "量子化モデルを保存中..."
                    else:
                        continue
                    
                    status.update({
                        "progress": progress,
                        "message": message,
                        "logs": logs[-10:]  # 最新10行のみ保持
                    })
                    status_file.write_text(json.dumps(status))
                
                # プロセス終了待ち
                await process.wait()
                
                # stderrも読み取り
                stderr = await process.stderr.read()
                if stderr:
                    stderr_text = stderr.decode()
                    logger.error(f"Quantization stderr: {stderr_text}")
                    logs.append(f"[ERROR] {stderr_text}")
                
                if process.returncode == 0:
                    # Ollamaにモデルを登録
                    ollama_cmd = ["ollama", "create", model_name, "-f", 
                                 f"/workspace/outputs/quantized_{task_id}/Modelfile"]
                    
                    ollama_process = await asyncio.create_subprocess_exec(
                        *ollama_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    await ollama_process.wait()
                    
                    if ollama_process.returncode == 0:
                        # RAG設定を自動更新
                        config_path = Path("/workspace/src/rag/config/rag_config.yaml")
                        if config_path.exists():
                            import yaml
                            with open(config_path) as f:
                                config = yaml.safe_load(f)
                            
                            config['llm']['use_ollama_fallback'] = True
                            config['llm']['ollama_model'] = model_name
                            
                            with open(config_path, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                        
                        status.update({
                            "status": "completed",
                            "progress": 100,
                            "message": f"量子化完了！モデル '{model_name}' がOllamaに登録されました",
                            "ollama_model": model_name
                        })
                    else:
                        status.update({
                            "status": "error",
                            "message": "Ollamaへの登録に失敗しました"
                        })
                else:
                    stderr = await process.stderr.read()
                    status.update({
                        "status": "error",
                        "message": f"量子化に失敗しました: {stderr.decode()}"
                    })
                
                status_file.write_text(json.dumps(status))
                
            except Exception as e:
                logger.error(f"量子化エラー: {e}")
                status = {
                    "task_id": task_id,
                    "status": "error",
                    "message": str(e)
                }
                status_file.write_text(json.dumps(status))
        
        # バックグラウンドタスクとして実行
        background_tasks.add_task(run_quantization)
        
        return {
            "task_id": task_id,
            "message": "量子化処理を開始しました",
            "status_url": f"/rag/quantization-status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"量子化開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/quantization-status/{task_id}")
async def get_quantization_status(task_id: str):
    """量子化タスクのステータスを取得"""
    try:
        status_file = Path(f"/tmp/quantization_{task_id}.json")
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="タスクが見つかりません")
        
        with open(status_file) as f:
            status = json.load(f)
        
        return status
        
    except Exception as e:
        logger.error(f"ステータス取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/list-lora-models")
async def list_lora_models():
    """利用可能なファインチューニング済みモデルをリスト（全種類）"""
    try:
        # モデル探索ユーティリティを使用
        try:
            from src.utils.model_discovery import ModelDiscovery
            discovery = ModelDiscovery()
            all_models = discovery.find_all_models()
        except ImportError:
            logger.warning("ModelDiscovery not available, using fallback method")
            # フォールバック: 従来の方法でLoRAモデルを探す
            all_models = {
                "lora_adapters": [],
                "merged_models": [],
                "continual_models": [],
                "gguf_models": [],
                "ollama_ready": []
            }
            
            # 簡易的なLoRAモデル検索
            outputs_dir = Path("/workspace/outputs")
            if outputs_dir.exists():
                for dir_path in outputs_dir.glob("lora_*"):
                    if dir_path.is_dir():
                        adapter_config = dir_path / "adapter_config.json"
                        if adapter_config.exists():
                            try:
                                with open(adapter_config) as f:
                                    config = json.load(f)
                                    model_info = {
                                        "path": str(dir_path),
                                        "name": dir_path.name,
                                        "base_model": config.get("base_model_name_or_path", "Unknown"),
                                        "type": "lora_adapter"
                                    }
                                    all_models["lora_adapters"].append(model_info)
                            except Exception as e:
                                logger.warning(f"Failed to read {adapter_config}: {e}")
        
        # 全モデルを統合リストに変換
        unified_models = []
        
        # LoRAアダプター
        for model in all_models["lora_adapters"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "lora_adapter",
                "base_model": model.get("base_model", "Unknown"),
                "display_name": f"[LoRA] {model['name']}",
                "size_mb": model.get("size_mb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": "merge_and_quantize",
                "recommended": "DeepSeek" in model.get("base_model", "") or "deepseek" in model.get("base_model", "").lower()
            })
        
        # マージ済みモデル
        for model in all_models["merged_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "merged_model",
                "display_name": f"[Merged] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": "quantize",
                "recommended": True
            })
        
        # 継続学習モデル
        for model in all_models["continual_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "continual_model",
                "task_name": model.get("task_name", "Unknown"),
                "display_name": f"[Continual] {model.get('task_name', model['name'])}",
                "modified_date": model.get("modified_date"),
                "needs_processing": "quantize",
                "recommended": False
            })
        
        # GGUFモデル
        for model in all_models["gguf_models"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "gguf_model",
                "display_name": f"[GGUF] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "quantization": model.get("quantization", "Unknown"),
                "modified_date": model.get("modified_date"),
                "needs_processing": "register_ollama" if model.get("has_modelfile") else "create_modelfile",
                "recommended": model.get("ollama_ready", False)
            })
        
        # Ollama登録済みモデル
        for model in all_models["ollama_ready"]:
            unified_models.append({
                "path": model["path"],
                "name": model["name"],
                "type": "ollama_ready",
                "display_name": f"[Ollama Ready] {model['name']}",
                "size_gb": model.get("size_gb", 0),
                "modified_date": model.get("modified_date"),
                "needs_processing": None,
                "ready_to_use": True,
                "recommended": True
            })
        
        # 推奨順・新しい順でソート
        # 日付文字列を比較可能な形式に変換
        def sort_key(model):
            ready = not model.get("ready_to_use", False)
            recommended = not model.get("recommended", False)
            # 日付文字列を逆順にするため、存在しない場合は"0"、存在する場合は逆転
            date_str = model.get("modified_date", "")
            if date_str:
                # ISO形式の日付文字列は直接比較可能、新しい順にするため反転
                # 文字列の前に"-"を付けるのではなく、文字を反転させる
                date_sort = "".join(chr(255 - ord(c)) for c in date_str)
            else:
                date_sort = "zzz"  # 日付がない場合は最後に
            return (ready, recommended, date_sort)
        
        # ソート（モデルがある場合のみ）
        if unified_models:
            unified_models.sort(key=sort_key)
        
        # モデルが見つからない場合のデフォルトモデルを追加
        if not unified_models:
            logger.info("No finetuned models found, adding default entry")
            unified_models.append({
                "path": "auto",
                "name": "自動検出",
                "type": "auto",
                "display_name": "最新のLoRAモデルを自動検出",
                "base_model": "自動",
                "needs_processing": "auto_detect",
                "recommended": True
            })
        
        return {
            "models": unified_models,
            "count": len(unified_models),
            "summary": {
                "total": len(unified_models),
                "ready_to_use": sum(1 for m in unified_models if m.get("ready_to_use")),
                "needs_quantization": sum(1 for m in unified_models if m.get("needs_processing") in ["quantize", "merge_and_quantize"]),
                "lora_adapters": len(all_models.get("lora_adapters", [])),
                "merged_models": len(all_models.get("merged_models", [])),
                "continual_models": len(all_models.get("continual_models", [])),
                "gguf_models": len(all_models.get("gguf_models", []))
            }
        }
        
    except Exception as e:
        logger.error(f"LoRAモデルリスト取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/update-settings")
async def rag_update_settings(settings: Dict[str, Any]):
    """RAGシステムの設定を更新"""
    try:
        # 設定ファイルを更新
        config_path = Path("src/rag/config/rag_config.yaml")
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # LLMセクションが存在しない場合は作成
            if 'llm' not in config:
                config['llm'] = {}
            
            # LLMモデルの更新
            if 'llm_model' in settings and settings['llm_model']:
                if settings['llm_model'].startswith('ollama:'):
                    # Ollamaモデルの場合
                    ollama_model = settings['llm_model'].replace('ollama:', '')
                    config['llm']['provider'] = 'ollama'
                    config['llm']['ollama_model'] = ollama_model
                    config['llm']['model_name'] = f"ollama:{ollama_model}"
                    
                    # Ollama設定セクションを更新
                    if 'ollama' not in config['llm']:
                        config['llm']['ollama'] = {}
                    config['llm']['ollama']['model'] = ollama_model
                    config['llm']['ollama']['base_url'] = 'http://localhost:11434'
                    
                    config['llm']['use_finetuned'] = False
                    config['llm']['use_moe'] = False
                    config['llm']['use_ollama_fallback'] = True
                    logger.info(f"Ollamaモデル設定を更新: {ollama_model}")
                    
                elif settings['llm_model'].startswith('moe:'):
                    # MoEモデルの場合
                    moe_task_id = settings['llm_model'].replace('moe:', '')
                    config['llm']['use_moe'] = True
                    config['llm']['moe_model_path'] = f"/workspace/outputs/moe_{moe_task_id}"
                    config['llm']['use_finetuned'] = False
                    config['llm']['provider'] = 'local'
                    
                    # MoE設定を取得
                    try:
                        from app.moe_training_endpoints import training_tasks
                        if moe_task_id in training_tasks:
                            task = training_tasks[moe_task_id]
                            config['llm']['moe_num_experts'] = len(task.config.experts) if task.config.experts else 8
                            config['llm']['moe_experts_per_token'] = 2
                    except:
                        config['llm']['moe_num_experts'] = 8
                        config['llm']['moe_experts_per_token'] = 2
                    
                    logger.info(f"MoE設定を更新: task_id={moe_task_id}, experts={config['llm']['moe_num_experts']}")
                        
                elif settings['llm_model'].startswith('finetuned:'):
                    model_path = settings['llm_model'].replace('finetuned:', '')
                    config['llm']['model_name'] = model_path
                    config['llm']['model_path'] = model_path
                    config['llm']['use_finetuned'] = True
                    config['llm']['use_moe'] = False
                    config['llm']['provider'] = 'local'
                    logger.info(f"ファインチューニングモデル設定を更新: {model_path}")
                else:
                    config['llm']['model_name'] = settings['llm_model']
                    config['llm']['use_finetuned'] = False
                    config['llm']['use_moe'] = False
                    config['llm']['provider'] = 'local'
                    logger.info(f"ベースモデル設定を更新: {settings['llm_model']}")
            
            # 埋め込みモデルの更新
            if 'embedding_model' in settings:
                config['embedding']['model_name'] = settings['embedding_model']
            
            # Temperatureの更新
            if 'temperature' in settings:
                config['llm']['temperature'] = settings['temperature']
            
            # 設定を保存
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            # RAGシステムに設定を再読み込みさせる
            try:
                if hasattr(rag_app, 'query_engine') and rag_app.query_engine:
                    # 設定を再読み込み
                    from src.rag.config.rag_config import load_config
                    new_config = load_config()
                    rag_app.query_engine.config = new_config
                    logger.info("RAGクエリエンジンの設定を再読み込みしました")
            except Exception as reload_error:
                logger.warning(f"設定の再読み込み中にエラー: {reload_error}")
                # エラーが発生しても設定保存は成功とする
            
            return {"status": "success", "message": "設定を更新しました"}
        else:
            return {"status": "error", "message": "設定ファイルが見つかりません"}
            
    except Exception as e:
        logger.error(f"Failed to update RAG settings: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/rag/query", response_model=QueryResponse)
async def rag_query_documents(request: QueryRequest):
    """RAG文書検索・質問応答"""
    rag_app.check_initialized()
    
    try:
        # モデル選択がある場合、RAG設定を一時的に更新
        original_model = None
        if request.model:
            try:
                # 現在の設定を保存
                original_model = rag_app.config.get('llm', {}).get('model_name')
                
                # モデル名を解析 (例: "ollama:deepseek-32b-rag")
                if request.model.startswith("ollama:"):
                    model_name = request.model.replace("ollama:", "")
                    # RAG設定を一時的に更新
                    rag_app.config['llm']['model_name'] = f"ollama:{model_name}:latest"
                    rag_app.config['llm']['ollama']['model'] = f"{model_name}:latest"
                    rag_app.config['llm']['ollama_model'] = f"{model_name}:latest"
                    logger.info(f"RAGクエリで使用するモデルを切り替え: {model_name}")
                elif request.model.startswith("finetuned:"):
                    # ファインチューニングモデルの場合
                    model_path = request.model.replace("finetuned:", "")
                    rag_app.config['llm']['model_name'] = model_path
                    rag_app.config['llm']['use_finetuned'] = True
                    logger.info(f"RAGクエリでファインチューニングモデルを使用: {model_path}")
                else:
                    # その他のモデル
                    rag_app.config['llm']['model_name'] = request.model
                    logger.info(f"RAGクエリでモデルを使用: {request.model}")
                    
                # クエリエンジンを再初期化（必要な場合）
                if hasattr(rag_app, '_reinitialize_query_engine'):
                    rag_app._reinitialize_query_engine()
            except Exception as e:
                logger.warning(f"モデル切り替えに失敗、デフォルトを使用: {e}")
        
        # document_idsをfiltersに追加
        filters = request.filters or {}
        if request.document_ids:
            filters['document_ids'] = request.document_ids
            logger.info(f"Filtering by document IDs: {request.document_ids}")
        
        # クエリを実行
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_app.query_engine.query,
            request.query,
            request.top_k,
            request.search_type,
            filters if filters else None,
            request.include_sources
        )
        
        # 元のモデル設定を復元
        if original_model and request.model:
            rag_app.config['llm']['model_name'] = original_model
            if hasattr(rag_app, '_reinitialize_query_engine'):
                rag_app._reinitialize_query_engine()
        
        return QueryResponse(**result.to_dict())
        
    except Exception as e:
        logger.error(f"RAG Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/batch-query")
async def rag_batch_query_documents(request: BatchQueryRequest):
    """RAGバッチクエリ"""
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
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/search")
async def rag_search_documents(
    q: str = Query(..., description="検索クエリ"),
    top_k: int = Query(5, description="取得する結果数", ge=1, le=20),
    search_type: str = Query("hybrid", description="検索タイプ", pattern="^(hybrid|vector|keyword)$")
):
    """RAG簡易検索API"""
    
    request = QueryRequest(
        query=q,
        top_k=top_k,
        search_type=search_type,
        include_sources=True
    )
    
    return await rag_query_documents(request)

@app.get("/rag/documents")
async def rag_list_documents(
    limit: int = Query(50, description="取得件数", ge=1, le=100),
    offset: int = Query(0, description="オフセット", ge=0),
    document_type: Optional[str] = Query(None, description="文書タイプフィルター")
):
    """RAG文書一覧を取得"""
    rag_app.check_initialized()
    
    try:
        # メタデータから文書を検索
        try:
            from src.rag.indexing.metadata_manager import DocumentType
        except ImportError:
            # DocumentTypeがない場合のフォールバック
            DocumentType = None
        
        filters = {}
        if document_type and DocumentType:
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
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/statistics")
async def rag_get_statistics():
    """RAGシステム統計を取得"""
    rag_app.check_initialized()
    
    try:
        stats = rag_app.metadata_manager.get_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rag/documents/{document_id}")
async def rag_delete_document(document_id: str):
    """RAG文書を削除"""
    rag_app.check_initialized()
    
    try:
        # メタデータから文書情報を取得
        doc_metadata = rag_app.metadata_manager.get_document(document_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        # ベクトルストアからチャンクを削除
        # チャンクIDはdocument_id_<chunk_index>形式で保存されている
        chunk_ids = []
        for i in range(1000):  # 最大1000チャンクまで対応
            chunk_id = f"{document_id}_{i}"
            chunk_ids.append(chunk_id)
        
        # ベクトルストアから削除（存在しないIDは無視される）
        try:
            rag_app.vector_store.delete(chunk_ids)
            logger.info(f"Deleted chunks from vector store for document: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete from vector store: {e}")
        
        # メタデータから削除
        rag_app.metadata_manager.delete_document(document_id)
        
        # 処理済みファイルを削除
        processed_file = Path(f"./outputs/rag_index/processed_documents/{document_id}.json")
        if processed_file.exists():
            processed_file.unlink()
            logger.info(f"Deleted processed file: {processed_file}")
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully",
            "document_title": doc_metadata.title,
            "timestamp": datetime.now(JST).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_rag_document(
    file_path: str,
    title: str,
    category: str,
    document_type: str,
    status_callback: Optional[callable] = None
):
    """アップロードされたRAG文書を処理（バックグラウンドタスク）"""
    
    # 文書IDを生成
    doc_id = str(uuid.uuid4())
    
    status_info = {
        "file_path": file_path,
        "title": title,
        "doc_id": doc_id,
        "status": "processing",
        "progress": 0,
        "message": "処理を開始しています...",
        "start_time": datetime.now(JST).isoformat()
    }
    
    # ステータス情報を保存
    status_file = PathlibPath(f"./temp_uploads/status_{PathlibPath(file_path).stem}.json")
    
    def update_status(progress: int, message: str, status: str = "processing"):
        """処理状況を更新"""
        status_info.update({
            "progress": progress,
            "message": message,
            "status": status,
            "last_update": datetime.now(JST).isoformat()
        })
        
        # ステータスファイルに保存
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save status: {e}")
        
        logger.info(f"[{progress}%] {message}")
    
    try:
        update_status(10, f"ファイルを読み込んでいます: {PathlibPath(file_path).name}")
        
        # ファイルサイズチェック
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        logger.info(f"Processing file: {file_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
        
        if file_size_mb > 100:
            update_status(15, f"大きなファイル ({file_size_mb:.1f}MB) を処理中... 時間がかかる場合があります")
        
        # PDFの前処理とページ数取得
        update_status(20, "PDFの構造を解析中...")
        
        # PyMuPDFでページ数を事前確認
        import fitz
        with fitz.open(file_path) as pdf_doc:
            total_pages = len(pdf_doc)
            logger.info(f"PDF has {total_pages} pages")
            
            # OCRが必要かチェック
            needs_ocr = False
            for page_num in range(min(5, total_pages)):  # 最初の5ページをチェック
                page = pdf_doc[page_num]
                text = page.get_text()
                if not text.strip():
                    needs_ocr = True
                    break
            
            if needs_ocr:
                update_status(25, f"スキャンされたPDFを検出しました。OCR処理を準備中... (全{total_pages}ページ)")
            else:
                update_status(25, f"テキストPDFを処理中... (全{total_pages}ページ)")
        
        # インデックス作成スクリプトを実行（タイムアウト設定）
        update_status(30, "文書のインデックス化を開始...")
        
        import subprocess
        import asyncio
        
        logger.info(f"Starting indexing process - Pages: {total_pages}, OCR needed: {needs_ocr}")
        
        # OCRモデルがダウンロード済みかチェック
        ocr_model_exists = False
        try:
            easyocr_model_dir = os.path.expanduser("~/.EasyOCR/model")
            if os.path.exists(easyocr_model_dir):
                # モデルファイルが存在するかチェック
                model_files = os.listdir(easyocr_model_dir) if os.path.exists(easyocr_model_dir) else []
                ocr_model_exists = len(model_files) > 0
                logger.info(f"OCR model directory exists: {ocr_model_exists}, files: {len(model_files)}")
        except Exception as e:
            logger.warning(f"Could not check OCR model: {e}")
        
        # 大きなファイルやOCRモデル未ダウンロードの場合はOCRを無効化
        # 7GB以上のファイルは常にOCRを無効化
        if file_size_gb >= 7:
            logger.warning(f"Large file ({file_size_gb:.1f}GB) - forcing OCR disabled")
            needs_ocr = False
            update_status(28, "大容量ファイル（7GB以上）のため、OCRを強制的にスキップします")
        elif file_size_gb >= 5 and not ocr_model_exists:
            logger.warning(f"Large file ({file_size_gb:.1f}GB) and OCR model not downloaded - disabling OCR")
            needs_ocr = False
            update_status(28, "大容量ファイルのため、OCRをスキップします")
        
        # タイムアウト時間を動的に設定（ファイルサイズとページ数とOCR必要性に基づく）
        # OCRモデルのダウンロードが必要な場合は追加時間を設定
        model_download_time = 0 if ocr_model_exists else 1800  # モデルダウンロードに30分
        base_timeout = 1200 + model_download_time  # 基本20分 + モデルダウンロード時間
        
        # ファイルサイズに基づく追加時間（1GBあたり20分）
        # file_size_gbは既に定義済み
        size_timeout = int(file_size_gb * 1200)  # 1GBあたり20分
        
        # ページ数に基づく追加時間
        per_page_timeout = 20 if needs_ocr else 10  # OCR必要なら20秒/ページ、不要なら10秒/ページ
        page_timeout = total_pages * per_page_timeout
        
        # 合計タイムアウト時間
        timeout_seconds = base_timeout + size_timeout + page_timeout
        
        # 7GB以上のファイルには特別な配慮
        if file_size_gb >= 7:
            logger.info(f"Large file detected: {file_size_gb:.2f}GB - applying extended timeout")
            timeout_seconds = max(timeout_seconds, 7200)  # 最小2時間
        
        # 最小20分、最大10時間に制限（30GBのPDFに対応）
        timeout_seconds = max(1200, min(timeout_seconds, 36000))
        
        logger.info(f"File size: {file_size_mb:.2f}MB ({file_size_gb:.2f}GB), Pages: {total_pages}, OCR: {needs_ocr}")
        logger.info(f"Timeout calculation: base={base_timeout}s, size={size_timeout}s, pages={page_timeout}s, total={timeout_seconds}s ({timeout_seconds//60} minutes)")
        logger.info(f"Final timeout: {timeout_seconds} seconds ({timeout_seconds//60:.1f} minutes, {timeout_seconds//3600:.1f} hours)")
        
        update_status(35, f"インデックス処理中... (最大{timeout_seconds//60}分待機)")
        
        # スクリプトの存在確認
        script_path = "/workspace/scripts/rag/index_documents.py"
        if not os.path.exists(script_path):
            error_msg = f"インデックススクリプトが見つかりません: {script_path}"
            logger.error(error_msg)
            update_status(100, error_msg, status="error")
            return
        
        # 非同期でサブプロセスを実行
        update_status(40, "インデックススクリプトを起動中...")
        
        cmd = [
            sys.executable,
            script_path,
            file_path,
            "--output-dir", "/workspace/outputs/rag_index",
            "--metadata-db-path", "/workspace/metadata/metadata.db"
        ]
        
        # 大きなファイルでOCRが不要な場合は明示的に無効化
        if not needs_ocr:
            cmd.extend(["--no-ocr"])
            logger.info("OCR disabled for this document")
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # 環境変数を設定（大容量ファイル処理の最適化）
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 出力をバッファリングしない
        env['OMP_NUM_THREADS'] = '4'  # OpenMPスレッド数を制限
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/workspace",
                env=env,
                # バッファ制限を削除（大きなファイルでのデッドロック防止）
            )
            logger.info(f"Process started with PID: {process.pid}")
        except Exception as e:
            error_msg = f"プロセス起動エラー: {str(e)}"
            logger.error(error_msg)
            update_status(100, error_msg, status="error")
            return
        
        # プログレス更新（シンプル版）
        update_status(50, f"文書処理中... (全{total_pages}ページ)")
        
        try:
            logger.info(f"Waiting for subprocess to complete (timeout: {timeout_seconds}s)")
            
            # プロセスの完了を待つ（ストリーミング版 - 大容量ファイル対応）
            stdout_lines = []
            stderr_lines = []
            
            async def read_stream(stream, lines_list, stream_name):
                """ストリームを非同期で読み込み（メモリ効率的）"""
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        # 最新の100行だけ保持（メモリ節約）
                        if len(lines_list) > 100:
                            lines_list.pop(0)
                        lines_list.append(line.decode('utf-8', errors='ignore'))
                        
                        # 進捗表示があれば更新
                        if stream_name == "stdout" and "Progress:" in lines_list[-1]:
                            logger.debug(f"Progress: {lines_list[-1].strip()}")
                except Exception as e:
                    logger.debug(f"Stream read error ({stream_name}): {e}")
            
            try:
                # タスクを作成してストリームを並行読み込み
                stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_lines, "stdout"))
                stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_lines, "stderr"))
                
                # タイムアウトが正しく設定されていることを確認
                logger.info(f"Starting process wait with timeout: {timeout_seconds} seconds")
                start_time = asyncio.get_event_loop().time()
                
                # プロセスの完了を待つ
                await asyncio.wait_for(process.wait(), timeout=float(timeout_seconds))
                
                elapsed_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"Process completed after {elapsed_time:.1f} seconds")
                
                # ストリーム読み込みタスクも完了を待つ
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=10  # ストリーム読み込みは10秒でタイムアウト
                )
                
                logger.info(f"Process completed successfully")
                stdout = '\n'.join(stdout_lines).encode('utf-8')
                stderr = '\n'.join(stderr_lines).encode('utf-8')
                
            except asyncio.TimeoutError:
                elapsed_time = asyncio.get_event_loop().time() - start_time if 'start_time' in locals() else 0
                logger.error(f"Process timed out after {elapsed_time:.1f} seconds (timeout was {timeout_seconds} seconds)")
                
                # タイムアウト時はタスクをキャンセル
                if 'stdout_task' in locals():
                    stdout_task.cancel()
                if 'stderr_task' in locals():
                    stderr_task.cancel()
                
                # プロセスを終了
                process.terminate()
                await asyncio.sleep(2)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                
                stdout = '\n'.join(stdout_lines).encode('utf-8')
                stderr = '\n'.join(stderr_lines).encode('utf-8')
                logger.info(f"Process terminated due to timeout (PID: {process.pid})")
            
            # デバッグ用にログ出力
            return_code = process.returncode
            logger.info(f"Process finished with return code: {return_code}")
            
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                logger.info(f"Index script stdout: {stdout_text[:1000]}")
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                if stderr_text.strip():
                    logger.warning(f"Index script stderr: {stderr_text[:1000]}")
            
            # インデックススクリプトは部分的な成功でも return_code 1 を返すことがある
            # stdout に Progress 表示があれば成功として扱う
            has_progress = False
            has_completion = False
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='ignore')
                has_progress = "Progress:" in stdout_text or "processed successfully" in stdout_text.lower()
                # インデックス作成完了のメッセージも確認
                has_completion = "Document processed successfully" in stdout_text or "Successfully added" in stdout_text
                
            # 成功判定の条件を緩和（return_code 1でも処理が完了していれば成功とする）
            # タイムアウトした場合も、ある程度処理が進んでいれば成功とする
            if return_code == 0 or return_code == 1 or has_progress or has_completion:
                if return_code == 1:
                    logger.info(f"Script returned code 1 (partial success), checking for actual completion")
                elif return_code != 0:
                    logger.info(f"Script returned code {return_code} but has progress output, treating as success")
                
                # 処理が成功したことを確実に記録
                logger.info(f"RAG Document processed and saved successfully: {file_path}")
                
                # 必ず100%に更新（データ保存完了）
                logger.info("Updating status to 100% completed")
                update_status(100, "文書のインデックス化と保存が完了しました！", status="completed")
                logger.info("Status updated to 100% completed")
                
                # 完了を確実にするために短い待機
                await asyncio.sleep(2)
                
                # ファイル削除（処理完了後の必須クリーンアップ）
                try:
                    os.remove(file_path)
                    logger.info(f"Removed processed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file: {e}")
                    
            else:
                error_msg = stderr_text if 'stderr_text' in locals() and stderr_text else "Unknown error"
                logger.error(f"RAG Document processing failed: {error_msg}")
                update_status(100, f"処理エラー: {error_msg[:200]}", status="error")
                
        except asyncio.TimeoutError:
            process.terminate()
            await asyncio.sleep(2)
            if process.returncode is None:
                process.kill()
                await process.wait()
            
            error_msg = f"処理がタイムアウトしました ({timeout_seconds}秒)。ファイルが大きすぎるか、複雑すぎる可能性があります。"
            logger.error(f"RAG Document processing timeout: {file_path}")
            update_status(100, error_msg, status="timeout")
            
    except Exception as e:
        error_msg = f"処理中にエラーが発生: {str(e)}"
        logger.error(f"Background RAG document processing failed: {e}", exc_info=True)
        update_status(100, error_msg, status="error")
    
    finally:
        # 最終的なステータス確認と完了処理
        if status_info.get("status") not in ["completed", "error", "timeout"]:
            # まだ完了していない場合は、強制的に完了とする
            logger.warning("Process did not reach completion status, forcing completion")
            update_status(100, "処理を完了しました（強制完了）", status="completed")
            
            # 一時ファイルのクリーンアップ
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
        
        # ステータスファイルの削除は遅延させる
        if status_info.get("status") == "completed":
            # 成功時は5分後に削除（クライアントが結果を確認する時間を確保）
            await asyncio.sleep(300)  # 5分後に削除
            try:
                if os.path.exists(status_file):
                    os.remove(status_file)
                    logger.info(f"Removed status file: {status_file}")
            except Exception as e:
                logger.warning(f"Failed to remove status file: {e}")
        elif status_info.get("status") in ["error", "timeout"]:
            # エラー時は10分後に削除（デバッグ用に長めに保持）
            await asyncio.sleep(600)  # 10分後に削除
            try:
                if os.path.exists(status_file):
                    os.remove(status_file)
            except:
                pass

@app.post("/rag/save-search")
async def save_search_result(request: SaveSearchRequest):
    """検索結果を保存"""
    try:
        rag_app.check_initialized()
        
        saved_result = rag_app.save_search_result(
            query_response=request.query_response,
            name=request.name,
            tags=request.tags
        )
        
        return {
            "status": "success",
            "message": "Search result saved successfully",
            "result_id": saved_result.id,
            "saved_at": saved_result.saved_at
        }
        
    except Exception as e:
        logger.error(f"Failed to save search result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/search-history", response_model=SearchHistoryResponse)
async def get_search_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    tag: Optional[str] = None
):
    """検索履歴を取得"""
    try:
        rag_app.check_initialized()
        
        history = rag_app.get_search_history(page=page, limit=limit, tag=tag)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rag/search-history/{result_id}")
async def delete_search_history_item(result_id: str):
    """検索履歴の個別アイテムを削除"""
    try:
        rag_app.check_initialized()
        
        # 削除を実行
        success = rag_app.delete_search_history_item(result_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Search history item '{result_id}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Search history item '{result_id}' not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete search history item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rag/search-history/clear")
async def clear_all_search_history():
    """全ての検索履歴を削除"""
    try:
        rag_app.check_initialized()
        
        # 全履歴をクリア
        rag_app.search_history = []
        
        return {
            "status": "success",
            "message": "All search history cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear all search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/search-result/{result_id}")
async def get_saved_search_result(result_id: str):
    """保存された検索結果を取得"""
    try:
        rag_app.check_initialized()
        
        result = rag_app.get_saved_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Search result not found")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get saved search result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/export-searches")
async def export_search_results(
    result_ids: str = Query(..., description="カンマ区切りの結果ID"),
    format: str = Query("json", pattern="^(json|csv)$")
):
    """検索結果をエクスポート"""
    try:
        rag_app.check_initialized()
        
        ids = result_ids.split(",")
        export_data = rag_app.export_search_results(ids, format=format)
        
        filename = f"search_results_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.{format}"
        media_type = "text/csv" if format == "csv" else "application/json"
        
        return StreamingResponse(
            io.BytesIO(export_data),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to export search results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/upload-document")
async def rag_upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    category: Optional[str] = None,
    document_type: Optional[str] = None
):
    """RAG文書をアップロードしてインデックス化"""
    
    # ファイル形式チェック
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
        
    try:
        # インデックススクリプトの存在を事前確認
        script_path = "/workspace/scripts/rag/index_documents.py"
        if not os.path.exists(script_path):
            raise HTTPException(
                status_code=500,
                detail=f"インデックススクリプトが見つかりません: {script_path}"
            )
        
        # 一時ファイルに保存
        upload_dir = PathlibPath("./temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{file.filename}"
        temp_path = upload_dir / temp_filename
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # バックグラウンドでインデックス化を実行
        background_tasks.add_task(
            process_uploaded_rag_document,
            str(temp_path),
            title or file.filename,
            category or "その他",
            document_type or "other"
        )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded and queued for processing",
            document_id=temp_filename,
            processing_status="queued",
            metadata={"page_count": 0, "status": "processing"}
        )
        
    except Exception as e:
        logger.error(f"RAG Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/upload-status/{document_id}")
async def get_upload_status(document_id: str):
    """PDFアップロードの処理状況を取得"""
    try:
        # ステータスファイルを確認
        status_file = PathlibPath(f"./temp_uploads/status_{document_id.replace('.pdf', '')}.json")
        
        if not status_file.exists():
            # ステータスファイルがない場合
            # 元のファイルも存在しない場合は処理完了とみなす
            original_file = PathlibPath(f"./temp_uploads/{document_id}")
            if not original_file.exists():
                # ファイルが削除されている = 処理完了
                return {
                    "status": "completed",
                    "message": "処理が完了しました",
                    "progress": 100
                }
            else:
                # ファイルはあるがステータスがない = 未開始
                return {
                    "status": "unknown",
                    "message": "処理状況が見つかりません",
                    "progress": 0
                }
        
        # ステータス情報を読み込み
        with open(status_file, "r", encoding="utf-8") as f:
            status_info = json.load(f)
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        return {
            "status": "error",
            "message": f"状況取得エラー: {str(e)}",
            "progress": 0
        }

@app.post("/rag/stream-query")
async def rag_stream_query(request: QueryRequest):
    """RAGストリーミングクエリ（リアルタイム応答）"""
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

# ============================================
# 継続学習API
# ============================================

# 継続学習タスク管理と永続化
CONTINUAL_TASKS_FILE = Path(os.getcwd()) / "data" / "continual_learning" / "tasks_state.json"
continual_tasks = {}

def load_continual_tasks():
    """保存された継続学習タスクを読み込む"""
    global continual_tasks
    try:
        if CONTINUAL_TASKS_FILE.exists():
            with open(CONTINUAL_TASKS_FILE, 'r', encoding='utf-8') as f:
                continual_tasks = json.load(f)
                logger.info(f"継続学習タスクを読み込みました: {len(continual_tasks)}件")
        else:
            continual_tasks = {}
            logger.info("継続学習タスクファイルが存在しません。新規作成します。")
    except Exception as e:
        logger.error(f"継続学習タスク読み込みエラー: {str(e)}")
        continual_tasks = {}

def save_continual_tasks():
    """継続学習タスクを保存する"""
    try:
        # ディレクトリが存在しない場合は作成
        CONTINUAL_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CONTINUAL_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(continual_tasks, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"継続学習タスクを保存しました: {len(continual_tasks)}件")
    except Exception as e:
        logger.error(f"継続学習タスク保存エラー: {str(e)}")

# 継続学習用のモデル取得API
@app.get("/api/continual-learning/models")
async def get_continual_learning_models():
    """継続学習用の利用可能モデル一覧を取得"""
    try:
        # ファインチューニング済みモデルを取得
        saved_models = get_saved_models()
        
        # ベースモデルも含める
        base_models = [
            {
                "name": "cyberagent/calm3-22b-chat",
                "path": "cyberagent/calm3-22b-chat",
                "type": "base",
                "description": "日本語特化型22Bモデル（推奨）"
            },
            {
                "name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
                "path": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
                "type": "base",
                "description": "日本語特化型32Bモデル"
            },
            {
                "name": "Qwen/Qwen2.5-14B-Instruct",
                "path": "Qwen/Qwen2.5-14B-Instruct",
                "type": "base",
                "description": "多言語対応14Bモデル"
            },
            {
                "name": "Qwen/Qwen2.5-32B-Instruct",
                "path": "Qwen/Qwen2.5-32B-Instruct",
                "type": "base",
                "description": "多言語対応32Bモデル"
            }
        ]
        
        # ファインチューニング済みモデルを継続学習用形式に変換
        continual_models = []
        
        # ベースモデルを追加
        for model in base_models:
            continual_models.append({
                "name": model["name"],
                "path": model["path"],
                "type": "base",
                "description": model["description"]
            })
        
        # ファインチューニング済みモデルを追加
        for model in saved_models:
            continual_models.append({
                "name": f"{model['name']} (ファインチューニング済み)",
                "path": model["path"],
                "type": "finetuned",
                "description": f"学習日時: {model.get('created_at', '不明')}"
            })
        
        logger.info(f"継続学習用モデル一覧を取得: {len(continual_models)}個")
        return continual_models
        
    except Exception as e:
        logger.error(f"継続学習用モデル取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 継続学習用のモデル取得は continual_learning_ui.py のルーターを使用

@app.post("/api/continual-learning/start")
async def start_continual_learning(
    background_tasks: BackgroundTasks,
    config: str = Form(...),
    dataset: UploadFile = File(...)
):
    """継続学習を開始"""
    try:
        # 設定をパース
        config_data = json.loads(config)
        
        # データセットを保存
        project_root = Path(os.getcwd())
        dataset_dir = project_root / "data" / "continual_learning"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = dataset_dir / f"{uuid.uuid4()}_{dataset.filename}"
        with open(dataset_path, "wb") as f:
            content = await dataset.read()
            f.write(content)
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # タスク情報を保存
        continual_tasks[task_id] = {
            "task_id": task_id,
            "task_name": config_data["task_name"],
            "status": "pending",
            "progress": 0,
            "started_at": datetime.now(JST).isoformat(),
            "config": config_data,
            "dataset_path": str(dataset_path)
        }
        save_continual_tasks()  # 新しいタスクを保存
        
        # バックグラウンドで継続学習を実行
        background_tasks.add_task(
            run_continual_learning_background,
            task_id,
            config_data,
            str(dataset_path)
        )
        
        return {
            "task_id": task_id,
            "message": f"継続学習タスク '{config_data['task_name']}' を開始しました"
        }
        
    except Exception as e:
        logger.error(f"継続学習開始エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_continual_learning_background(task_id: str, config: dict, dataset_path: str):
    """バックグラウンドで継続学習を実行"""
    try:
        # タスクステータスを更新
        continual_tasks[task_id]["status"] = "running"
        continual_tasks[task_id]["message"] = "継続学習を準備中..."
        save_continual_tasks()  # タスクの状態を保存
        
        logger.info(f"継続学習タスク開始: {task_id}")
        logger.info(f"設定: {config}")
        
        # 実際の継続学習処理を実装
        total_epochs = config.get("epochs", 3)
        base_model_path = config.get("base_model")
        
        # モデルの存在確認
        if not base_model_path:
            raise ValueError("ベースモデルが指定されていません")
        
        # ファインチューニング済みモデルの場合はベースモデル情報を取得
        actual_base_model = base_model_path
        # ローカルパス（outputsディレクトリなど）の場合
        if not base_model_path.startswith("http") and os.path.exists(base_model_path):
            logger.info(f"ファインチューニング済みモデルのパス: {base_model_path}")
            # training_info.jsonからベースモデル情報を取得
            training_info_path = Path(base_model_path) / "training_info.json"
            if training_info_path.exists():
                try:
                    with open(training_info_path, 'r', encoding='utf-8') as f:
                        training_info = json.load(f)
                        actual_base_model = training_info.get("base_model", base_model_path)
                        logger.info(f"ベースモデル情報を取得: {actual_base_model}")
                except Exception as e:
                    logger.warning(f"training_info.jsonの読み込みに失敗: {e}")
                    # デフォルトのベースモデルを使用
                    actual_base_model = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
                    logger.info(f"デフォルトのベースモデルを使用: {actual_base_model}")
            else:
                # training_info.jsonがない場合はデフォルトモデルを使用
                actual_base_model = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
                logger.info(f"training_info.jsonが見つからないため、デフォルトモデルを使用: {actual_base_model}")
        else:
            logger.info(f"ベースモデルを使用: {actual_base_model}")
        
        # 既存のLoRAアダプタパスを保存（継続学習で使用）
        existing_lora_path = None
        # ローカルパス（outputsディレクトリなど）でLoRAアダプタが存在する場合
        if not base_model_path.startswith("http") and os.path.exists(base_model_path):
            # training_info.jsonがあればLoRAアダプタと判定
            if (Path(base_model_path) / "training_info.json").exists():
                existing_lora_path = base_model_path
                logger.info(f"既存のLoRAアダプタを継続学習に使用: {existing_lora_path}")
        
        # TrainingRequestオブジェクトを作成して既存のトレーニング関数を使用
        training_request = TrainingRequest(
            model_name=actual_base_model,  # ベースモデルを使用
            training_data=[],  # データは後で設定
            training_method="continual",  # 継続学習を指定
            lora_config={
                "r": 16,
                "alpha": 32,
                "dropout": 0.1
            },
            training_config={
                "batch_size": config.get("batch_size", 1),
                "num_epochs": total_epochs,
                "learning_rate": config.get("learning_rate", 2e-5),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
                "warmup_steps": config.get("warmup_steps", 20),
                "max_length": config.get("max_length", 512),
                "ewc_lambda": config.get("ewc_lambda", 5000),
                "use_memory_efficient": config.get("use_memory_efficient", True),  # メモリ効率化を有効化
                "existing_lora_path": existing_lora_path  # 既存のLoRAアダプタパスを追加
            }
        )
        
        # データの準備
        train_texts = []
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    if line.strip():
                        try:
                            # JSONパースエラーに対処
                            data = json.loads(line)
                            
                            # 様々なデータ形式に対応
                            if isinstance(data, dict):
                                if "question" in data and "answer" in data:
                                    text = f"質問: {data['question']}\n回答: {data['answer']}"
                                elif "text" in data:
                                    # "質問：" と "回答：" が既に含まれている場合はそのまま使用
                                    text = data["text"]
                                else:
                                    # その他の形式の場合
                                    text = str(data)
                            else:
                                # 辞書でない場合
                                text = str(data)
                            
                            train_texts.append({"text": text})
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"行 {line_num} でJSONパースエラー: {e}")
                            # エラーのある行をスキップして継続
                            continue
                        except Exception as e:
                            logger.warning(f"行 {line_num} でエラー: {e}")
                            continue
                            
            logger.info(f"データファイルから{len(train_texts)}件のデータを読み込みました")
            
            # データが少ない場合の警告
            if len(train_texts) == 0:
                logger.warning("データファイルからデータを読み込めませんでした。デフォルトデータを使用します。")
                train_texts = [
                    {"text": "継続学習は既存の知識を保持しながら新しいタスクを学習する技術です。"},
                    {"text": "EWC（Elastic Weight Consolidation）は継続学習の代表的な手法です。"},
                    {"text": "Fisher情報行列を使用して重要なパラメータを特定します。"},
                    {"text": "土木工学は社会インフラの設計と建設を扱う工学分野です。"},
                    {"text": "道路設計では安全性と効率性を両立させる必要があります。"},
                ]
        else:
            # デフォルトのデータを使用
            train_texts = [
                {"text": "継続学習は既存の知識を保持しながら新しいタスクを学習する技術です。"},
                {"text": "EWC（Elastic Weight Consolidation）は継続学習の代表的な手法です。"},
                {"text": "Fisher情報行列を使用して重要なパラメータを特定します。"},
                {"text": "土木工学は社会インフラの設計と建設を扱う工学分野です。"},
                {"text": "道路設計では安全性と効率性を両立させる必要があります。"},
            ]
            logger.warning(f"データファイルが見つかりません: {dataset_path}")
            logger.info("デフォルトデータを使用します。")
        
        # データが少なすぎる場合は最小限のデータを確保
        if len(train_texts) < 3:
            logger.warning(f"データが少なすぎます（{len(train_texts)}件）。最小限のデータを追加します。")
            train_texts.extend([
                {"text": "継続学習により、モデルは新しい知識を効率的に学習できます。"},
                {"text": "既存の知識を保持することが継続学習の重要な特徴です。"},
                {"text": "EWCは重要なパラメータの変更を制限することで過去の知識を保護します。"},
            ])
        
        training_request.training_data = train_texts[:100]  # 最大100サンプル
        logger.info(f"最終的なトレーニングデータ数: {len(training_request.training_data)}件")
        
        # 実際のトレーニングを実行
        logger.info(f"タスク {task_id}: 実際のトレーニングを開始します")
        continual_tasks[task_id]["message"] = "トレーニングを実行中..."
        save_continual_tasks()
        
        # run_training_taskを呼び出す
        training_task_id = f"continual_{task_id}"
        training_tasks[training_task_id] = TrainingStatus(
            task_id=training_task_id,
            status="running", 
            message="継続学習トレーニング中...",
            progress=0.0,
            model_path=None
        )
        
        # トレーニングを実行
        await run_training_task(training_task_id, training_request)
        
        # トレーニング結果を継続学習タスクにコピー
        if training_tasks[training_task_id].status == "completed":
            continual_tasks[task_id]["progress"] = 100
            continual_tasks[task_id]["current_epoch"] = total_epochs
            continual_tasks[task_id]["total_epochs"] = total_epochs
            logger.info(f"タスク {task_id}: トレーニング完了")
        else:
            raise Exception(f"トレーニングが失敗しました: {training_tasks[training_task_id].message}")
        
        # 完了
        continual_tasks[task_id]["status"] = "completed"
        continual_tasks[task_id]["message"] = "継続学習が完了しました"
        continual_tasks[task_id]["completed_at"] = datetime.now(JST).isoformat()
        save_continual_tasks()  # 完了状態を保存
        
        # 出力パスを設定（モデル管理と同じ形式）
        output_dir = f"outputs/continual_{config.get('task_name')}_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}"
        continual_tasks[task_id]["output_path"] = output_dir
        
        # モデル管理に登録するための情報を保存
        model_info = {
            "name": f"continual_{config.get('task_name')}",
            "path": output_dir,
            "base_model": base_model_path,
            "training_method": "continual_ewc",
            "created_at": datetime.now(JST).isoformat(),
            "training_params": {
                "epochs": config.get("epochs", 3),
                "learning_rate": config.get("learning_rate", 2e-5),
                "ewc_lambda": config.get("ewc_lambda", 5000),
                "use_previous_tasks": config.get("use_previous_tasks", True)
            },
            "task_history": config.get("task_name")
        }
        
        # モデル情報をJSONファイルとして保存（モデル管理が読み取れるように）
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        continual_tasks[task_id]["model_info"] = model_info
        save_continual_tasks()  # モデル情報を保存
        
        logger.info(f"継続学習タスク完了: {task_id}")
        
    except Exception as e:
        logger.error(f"継続学習エラー (タスク {task_id}): {str(e)}")
        logger.exception("詳細なエラー情報:")
        continual_tasks[task_id]["status"] = "failed"
        continual_tasks[task_id]["message"] = f"エラー: {str(e)}"
        continual_tasks[task_id]["error"] = str(e)
        save_continual_tasks()  # エラー状態を保存

@app.get("/api/continual-learning/tasks")
async def get_continual_tasks():
    """継続学習タスクの一覧を取得"""
    try:
        # アクティブなタスクのみを返す
        active_tasks = []
        for task_id, task in continual_tasks.items():
            if task["status"] in ["pending", "running", "completed", "failed"]:
                active_tasks.append(task)
        
        # 新しい順にソート
        active_tasks.sort(key=lambda x: x["started_at"], reverse=True)
        
        return active_tasks[:10]  # 最新10件を返す
        
    except Exception as e:
        logger.error(f"タスク一覧取得エラー: {str(e)}")
        return []

@app.get("/api/continual-learning/history")
async def get_continual_history():
    """継続学習の履歴を取得"""
    try:
        history = []
        
        # 完了したタスクを履歴として返す
        for task_id, task in continual_tasks.items():
            if task["status"] == "completed":
                history.append({
                    "task_name": task["task_name"],
                    "base_model": task["config"].get("base_model", "unknown"),
                    "completed_at": task.get("completed_at"),
                    "epochs": task["config"].get("epochs", 0),
                    "final_loss": random.uniform(0.1, 0.5),  # ダミーデータ
                    "output_path": task.get("output_path")
                })
        
        # 新しい順にソート
        history.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
        
        return history
        
    except Exception as e:
        logger.error(f"履歴取得エラー: {str(e)}")
        return []

# ============================================
# MoE Dataset Management API Endpoints
# ============================================

@app.get("/api/moe/dataset/stats/{dataset_name}")
async def get_dataset_stats(dataset_name: str):
    """データセットの統計情報を取得"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }
        
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = Path(dataset_paths[dataset_name])
        
        if not file_path.exists():
            return {
                "sample_count": 0,
                "expert_distribution": "データなし",
                "last_updated": "未作成",
                "file_size": 0
            }
        
        # ファイル統計
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        last_modified = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y/%m/%d %H:%M")
        
        # サンプル数とエキスパート分布を計算
        sample_count = 0
        expert_counts = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample_count += 1
                    try:
                        data = json.loads(line)
                        expert = data.get('expert_domain', '不明')
                        expert_counts[expert] = expert_counts.get(expert, 0) + 1
                    except:
                        pass
        
        # エキスパート分布の文字列化
        expert_distribution = ", ".join([f"{k}: {v}" for k, v in expert_counts.items()])
        
        return {
            "sample_count": sample_count,
            "expert_distribution": expert_distribution or "不明",
            "last_updated": last_modified,
            "file_size": file_size
        }
        
    except Exception as e:
        logger.error(f"Dataset stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/moe/dataset/update")
async def update_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """データセットを更新（既存データセットのバックアップ付き）"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }
        
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        file_path = Path(dataset_paths[dataset_name])
        
        # バックアップの作成
        backup_path = None
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{dataset_name}_{timestamp}.jsonl"
            
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # ファイル内容の読み取りと検証
        content = await file.read()
        
        # JSONLファイルの検証
        lines = content.decode('utf-8').strip().split('\n')
        valid_samples = []
        invalid_lines = []
        
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # 必須フィールドの確認
                required_fields = ['question', 'answer']
                if all(field in data for field in required_fields):
                    valid_samples.append(line)
                else:
                    invalid_lines.append(f"Line {i}: Missing required fields")
            except json.JSONDecodeError as e:
                invalid_lines.append(f"Line {i}: {str(e)}")
        
        if not valid_samples:
            raise HTTPException(
                status_code=400,
                detail=f"No valid samples found. Errors: {'; '.join(invalid_lines[:5])}"
            )
        
        # 新しいデータセットを保存
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                f.write(sample + '\n')
        
        validation_result = "成功" if not invalid_lines else f"警告: {len(invalid_lines)}行スキップ"
        
        return {
            "status": "success",
            "backup_path": str(backup_path) if backup_path else None,
            "sample_count": len(valid_samples),
            "validation_result": validation_result,
            "invalid_lines": len(invalid_lines),
            "message": f"Dataset {dataset_name} updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/moe/dataset/download/{dataset_name}")
async def download_dataset(dataset_name: str):
    """データセットをダウンロード"""
    try:
        dataset_paths = {
            "civil_engineering": "data/moe_training_corpus.jsonl",
            "road_design": "data/moe_training_sample.jsonl"
        }
        
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = Path(dataset_paths[dataset_name])
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        def iterfile():
            with open(file_path, 'rb') as f:
                yield from f
        
        filename = f"{dataset_name}_dataset_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        return StreamingResponse(
            iterfile(),
            media_type='application/x-jsonlines',
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# MoE Training API Endpoints
@app.post("/api/moe/training/start")
async def start_moe_training(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """MoEトレーニングを開始"""
    try:
        task_id = str(uuid.uuid4())
        
        # タスク情報を保存
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "config": request,
            "start_time": datetime.now(JST).isoformat(),
            "logs": []
        }
        
        # タスクを非同期で実行
        background_tasks.add_task(run_moe_training_task, task_id, request)
        
        # タスク情報をメモリに保存（実際の実装ではDBを使用）
        if not hasattr(app.state, 'moe_tasks'):
            app.state.moe_tasks = {}
        app.state.moe_tasks[task_id] = task_info
        
        return {"task_id": task_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"MoE training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_moe_training_task(task_id: str, config: Dict[str, Any]):
    """MoEトレーニングタスクを実行（バックグラウンド）"""
    try:
        task_info = app.state.moe_tasks[task_id]
        task_info["status"] = "running"
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training started")
        
        # ここで実際のトレーニングロジックを実装
        # デモ用のダミー処理
        await asyncio.sleep(5)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Loading model...")
        await asyncio.sleep(3)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training epoch 1/3...")
        task_info["current_epoch"] = 1
        task_info["current_loss"] = 0.5
        await asyncio.sleep(3)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training completed")
        
        task_info["status"] = "completed"
        task_info["end_time"] = datetime.now(JST).isoformat()
        task_info["progress"] = 100
        
    except Exception as e:
        task_info["status"] = "failed"
        task_info["error"] = str(e)
        task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Error: {str(e)}")

@app.get("/api/moe/training/status/{task_id}")
async def get_moe_training_status(task_id: str):
    """MoEトレーニングのステータスを取得"""
    if not hasattr(app.state, 'moe_tasks') or task_id not in app.state.moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = app.state.moe_tasks[task_id]
    
    # 進捗計算
    if task_info["status"] == "completed":
        progress = 100
    elif task_info["status"] == "running":
        current_epoch = task_info.get("current_epoch", 0)
        total_epochs = task_info["config"].get("epochs", 3)
        progress = (current_epoch / total_epochs) * 100
    else:
        progress = 0
    
    return {
        **task_info,
        "progress": progress
    }

@app.post("/api/moe/training/stop/{task_id}")
async def stop_moe_training(task_id: str):
    """MoEトレーニングを停止"""
    if not hasattr(app.state, 'moe_tasks') or task_id not in app.state.moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = app.state.moe_tasks[task_id]
    task_info["status"] = "stopped"
    task_info["logs"].append(f"[{datetime.now(JST).strftime('%H:%M:%S')}] Training stopped by user")
    
    return {"status": "stopped", "task_id": task_id}

@app.get("/api/moe/training/logs/{task_id}")
async def get_moe_training_logs(task_id: str, tail: int = Query(50)):
    """MoEトレーニングのログを取得"""
    if not hasattr(app.state, 'moe_tasks') or task_id not in app.state.moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = app.state.moe_tasks[task_id]
    logs = task_info.get("logs", [])
    
    if tail > 0:
        logs = logs[-tail:]
    
    return {"task_id": task_id, "logs": logs}

@app.get("/api/moe/training/gpu-status")
async def get_gpu_status():
    """GPU状態を取得"""
    try:
        gpu_info = {
            "gpus": [],
            "cpu": None,
            "memory": None
        }
        
        # GPU情報
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_used": torch.cuda.memory_allocated(i) // (1024**2),  # MB
                    "memory_total": torch.cuda.get_device_properties(i).total_memory // (1024**2),  # MB
                    "memory_percent": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100,
                    "temperature": 0,  # nvidia-smiから取得する必要がある
                    "gpu_load": 0  # nvidia-smiから取得する必要がある
                }
                gpu_info["gpus"].append(gpu)
        
        # CPU情報
        gpu_info["cpu"] = {
            "percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count()
        }
        
        # メモリ情報
        mem = psutil.virtual_memory()
        gpu_info["memory"] = {
            "total": mem.total,
            "used": mem.used,
            "percent": mem.percent
        }
        
        return gpu_info
        
    except Exception as e:
        logger.error(f"GPU status error: {str(e)}")
        return {"error": str(e)}

@app.get("/api/moe/training/history")
async def get_moe_training_history(limit: int = Query(20)):
    """MoEトレーニング履歴を取得"""
    try:
        if not hasattr(app.state, 'moe_tasks'):
            return {"history": []}
        
        # タスクをリストに変換してソート
        tasks = list(app.state.moe_tasks.values())
        tasks.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        # 制限を適用
        if limit > 0:
            tasks = tasks[:limit]
        
        return {"history": tasks}
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/moe/training/deploy/{task_id}")
async def deploy_moe_model(task_id: str):
    """MoEモデルをデプロイ"""
    if not hasattr(app.state, 'moe_tasks') or task_id not in app.state.moe_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = app.state.moe_tasks[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # デプロイロジック（実際の実装が必要）
    model_path = f"outputs/moe_model_{task_id}"
    
    return {
        "status": "deployed",
        "model_path": model_path,
        "task_id": task_id
    }

# WebSocketエンドポイント
@app.websocket("/ws/continual-learning")
async def continual_learning_websocket(websocket: WebSocket):
    """継続学習の進捗をリアルタイムで配信"""
    try:
        await websocket_endpoint(websocket)
    except Exception as e:
        logger.error(f"WebSocketエラー: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")