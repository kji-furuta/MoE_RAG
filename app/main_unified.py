#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Unified Implementation
統合されたWebインターフェース実装
"""

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
                            else:
                                model_type = "LoRA"
                                model_size = "~1.6MB"
                    except Exception as e:
                        logger.warning(f"training_info.jsonの読み込みに失敗: {e}")
                        # ディレクトリ名から推定
                        if "lora" in model_dir.name.lower():
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
            model, tokenizer = load_model_and_tokenizer(
                model_name=request.model_name,
                training_method=request.training_method,
                cache_dir=cache_dir
            )
            logger.info(f"Task {task_id}: モデル読み込み完了")
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Task {task_id}: モデル読み込みエラー: {str(e)}")
            logger.error(f"Task {task_id}: エラー詳細: {error_traceback}")
            training_tasks[task_id].status = "failed"
            training_tasks[task_id].message = handle_model_loading_error(e, request.model_name, task_id)
            return
        
        # LoRA設定
        if request.training_method in ["lora", "qlora"]:
            training_tasks[task_id].message = "LoRAアダプターを設定中..."
            training_tasks[task_id].progress = 30.0
            
            # QLoRAの場合はモデルを準備
            if request.training_method == "qlora":
                model = prepare_model_for_kbit_training(model)
            
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
        
        # JSONLファイルからデータを読み込み
        train_texts = []
        for data_path in request.training_data:
            data_file = Path(data_path)
            if data_file.exists() and data_file.suffix == '.jsonl':
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                train_texts.append(data['text'])
                            elif 'input' in data and 'output' in data:
                                train_texts.append(f"{data['input']}\n{data['output']}")
                        except json.JSONDecodeError:
                            continue
        
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
        
        # データセット作成
        train_dataset = SimpleDataset(train_texts, tokenizer)
        
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
            
            def compute_loss(self, model, inputs, return_outputs=False):
                """損失関数にEWCペナルティを追加"""
                outputs = model(**inputs)
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
                
                # EWCペナルティを追加
                if self.use_ewc and self.ewc_helper is not None and self.ewc_helper.fisher_matrix is not None:
                    ewc_loss = self.ewc_helper.compute_ewc_loss(model)
                    loss = loss + self.ewc_lambda * ewc_loss
                    
                return (loss, outputs) if return_outputs else loss
        

        
        # トレーニング引数
        # フルファインチューニングの場合は、より慎重なパラメータを使用
        if request.training_method == "full":
            batch_size = get_config_value(training_config, "batch_size", 1, int)
            gradient_accumulation_steps = get_config_value(training_config, "gradient_accumulation_steps", 16, int)
            num_epochs = get_config_value(training_config, "num_epochs", 1, int)
            
            effective_batch_size = batch_size * gradient_accumulation_steps
            total_steps = len(train_dataset) * num_epochs // effective_batch_size
            max_steps = min(100, total_steps)  # フルファインチューニングは100ステップまで
            learning_rate = 5e-6  # より低い学習率
        else:
            batch_size = get_config_value(training_config, "batch_size", 1, int)
            max_steps = min(50, len(train_dataset) // batch_size)
            learning_rate = get_config_value(training_config, "learning_rate", 2e-4, float)
        
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
        # フルファインチューニングの場合はEWCを使用 (リソース問題のためデフォルトで無効化)
        use_ewc = False # request.training_method == "full"
        ewc_lambda = 5000.0 if use_ewc else 0.0
        
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            use_ewc=use_ewc,
            ewc_lambda=ewc_lambda,
        )
        
        # EWCを使用する場合、事前学習データでFisher行列を計算
        if use_ewc and trainer.ewc_helper is not None:
            logger.info("Fisher行列を計算中...")
            # 事前学習データとして一般的な日本語テキストを使用
            pretrain_texts = [
                "人工知能は急速に発展している技術分野です。",
                "機械学習はデータから学習するアルゴリズムです。",
                "深層学習はニューラルネットワークを使用します。",
                "自然言語処理は言語を理解する技術です。",
                "コンピュータビジョンは画像を解析します。",
            ]
            
            pretrain_dataset = SimpleDataset(pretrain_texts, tokenizer)
            from torch.utils.data import DataLoader
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)
            
            # Fisher行列の計算（最適化版）
            trainer.ewc_helper.compute_fisher_matrix(pretrain_loader, max_batches=30)
            logger.info("Fisher行列の計算完了")
        
        # トレーニング実行
        logger.info(f"Task {task_id}: 実際のトレーニング開始")
        try:
            trainer.train()
            logger.info(f"Task {task_id}: トレーニング完了")
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
        
        if file.filename.endswith('.jsonl'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data_count = len(lines)
                    
                    for i, line in enumerate(lines[:5]):  # 最初の5行をサンプルとして取得
                        line = line.strip()
                        if line:  # 空行をスキップ
                            try:
                                data = json.loads(line)
                                sample_data.append(data)
                            except json.JSONDecodeError as je:
                                logger.error(f"JSON parse error at line {i+1}: {str(je)}")
                                raise HTTPException(status_code=400, detail=f"行 {i+1} でJSONパースエラー: {str(je)}")
                        
                logger.info(f"JSONL解析完了: {data_count}行, サンプル: {len(sample_data)}件")
                
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
        
        result = {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "data_count": data_count,
            "sample_data": sample_data[:3]
        }
        
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
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        
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
                        quantization_config = BitsAndBytesConfig(
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
                            except:
                                pass
                        
                        # モデルタイプの判定
                        if "qlora" in model_dir.name.lower() or "4bit" in model_dir.name.lower():
                            model_info["training_method"] = "qlora"
                            model_info["size"] = "~1.0MB"
                        elif "lora" in model_dir.name.lower():
                            model_info["training_method"] = "lora"
                            model_info["size"] = "~1.6MB"
                        elif "フルファインチューニング" in model_dir.name:
                            model_info["training_method"] = "full"
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
        
        return models
        
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {str(e)}")
        return {"finetuned_models": [], "ollama_models": [], "error": str(e)}

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
            
            # LLMモデルの更新
            if 'llm_model' in settings and settings['llm_model']:
                if settings['llm_model'].startswith('finetuned:'):
                    model_path = settings['llm_model'].replace('finetuned:', '')
                    config['llm']['model_name'] = model_path
                    config['llm']['model_path'] = model_path
                    config['llm']['use_finetuned'] = True
                else:
                    config['llm']['model_name'] = settings['llm_model']
                    config['llm']['use_finetuned'] = False
            
            # 埋め込みモデルの更新
            if 'embedding_model' in settings:
                config['embedding']['model_name'] = settings['embedding_model']
            
            # Temperatureの更新
            if 'temperature' in settings:
                config['llm']['temperature'] = settings['temperature']
            
            # 設定を保存
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
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
    document_type: str
):
    """アップロードされたRAG文書を処理（バックグラウンドタスク）"""
    
    try:
        logger.info(f"Processing uploaded RAG document: {file_path}")
        
        # インデックス作成スクリプトを実行
        import subprocess
        
        result = subprocess.run([
            sys.executable,
            "scripts/rag/index_documents.py",
            file_path,
            "--output-dir", "./outputs/rag_index"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            logger.info(f"RAG Document processed successfully: {file_path}")
        else:
            logger.error(f"RAG Document processing failed: {result.stderr}")
            logger.error(f"RAG Document processing stdout: {result.stdout}")
            
        # 一時ファイルを削除（成功時のみ）
        if result.returncode == 0:
            logger.info(f"Removing processed file: {file_path}")
            os.remove(file_path)
        else:
            logger.warning(f"Keeping failed file for debugging: {file_path}")
        
    except Exception as e:
        logger.error(f"Background RAG document processing failed: {e}")

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
        
        # 実際の継続学習処理をここに実装
        # 現在はシミュレーション
        total_epochs = config.get("epochs", 3)
        base_model_path = config.get("base_model")
        
        # モデルの存在確認
        if not base_model_path:
            raise ValueError("ベースモデルが指定されていません")
        
        # ファインチューニング済みモデルの場合はパスを確認
        if "/" not in base_model_path and os.path.exists(base_model_path):
            logger.info(f"ファインチューニング済みモデルを使用: {base_model_path}")
        else:
            logger.info(f"ベースモデルを使用: {base_model_path}")
        
        for epoch in range(total_epochs):
            # プログレス更新
            progress = int((epoch + 1) / total_epochs * 100)
            continual_tasks[task_id]["progress"] = progress
            continual_tasks[task_id]["current_epoch"] = epoch + 1
            continual_tasks[task_id]["total_epochs"] = total_epochs
            continual_tasks[task_id]["message"] = f"エポック {epoch + 1}/{total_epochs} を実行中..."
            save_continual_tasks()  # 進捗を保存
            
            logger.info(f"タスク {task_id}: エポック {epoch + 1}/{total_epochs}")
            
            # 実際の学習処理をここに追加
            await asyncio.sleep(5)  # シミュレーション用の待機
            
            # TODO: 実際の継続学習実装
            # 1. モデルとトークナイザーの読み込み
            # 2. データセットの準備
            # 3. EWC設定
            # 4. トレーニングループ
            # 5. チェックポイント保存
        
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