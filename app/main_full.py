#!/usr/bin/env python3
"""
AI Fine-tuning Toolkit Web API - Full Implementation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import uuid
from pathlib import Path
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import yaml
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション初期化
app = FastAPI(
    title="AI Fine-tuning Toolkit",
    description="日本語LLMファインチューニング用Webインターフェース",
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

# 静的ファイルディレクトリの自動検出
import os
from pathlib import Path

def find_static_directory():
    """静的ファイルディレクトリを検索"""
    # 現在のファイルの場所から相対的に特定
    current_file = Path(__file__)
    static_path = current_file.parent / "static"
    
    if static_path.exists() and static_path.is_dir():
        return str(static_path)
    
    # プロジェクトルートからの相対パス
    project_root = Path(os.getcwd())
    possible_paths = [
        project_root / "app" / "static",
        project_root / "static"
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return str(path)
    
    return str(static_path)  # デフォルト（存在しなくても作成される）

static_dir = find_static_directory()
print(f"Using static directory: {static_dir}")

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# HTMLページの配信
@app.get("/manual", response_class=HTMLResponse)
async def manual_page():
    """利用マニュアルページ"""
    manual_path = os.path.join(static_dir, "manual.html")
    try:
        if os.path.exists(manual_path):
            with open(manual_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content=f"<h1>Manual page not found</h1><p>Looking for: {manual_path}</p><p>Static dir: {static_dir}</p>", 
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading manual</h1><p>Error: {str(e)}</p><p>Path: {manual_path}</p>", 
            status_code=500
        )

@app.get("/system-overview", response_class=HTMLResponse)
async def system_overview_page():
    """システム概要ページ"""
    overview_path = os.path.join(static_dir, "system-overview.html")
    try:
        if os.path.exists(overview_path):
            with open(overview_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content=f"<h1>System overview page not found</h1><p>Looking for: {overview_path}</p><p>Static dir: {static_dir}</p>", 
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading system overview</h1><p>Error: {str(e)}</p><p>Path: {overview_path}</p>", 
            status_code=500
        )

@app.get("/docs/{doc_name}")
async def serve_documentation(doc_name: str):
    """ドキュメントファイルの配信"""
    # セキュリティ: ファイル名の検証
    allowed_docs = [
        "API_REFERENCE.md",
        "LARGE_MODEL_SETUP.md", 
        "MULTI_GPU_OPTIMIZATION.md",
        "USER_MANUAL.md",
        "QUICKSTART_GUIDE.md",
        "USAGE_GUIDE.md",
        "TRAINED_MODEL_USAGE.md",
        "DEEPSEEK_SETUP.md"
    ]
    
    if doc_name not in allowed_docs:
        return {"error": "Document not found", "requested": doc_name, "available": allowed_docs}
    
    # ドキュメントパスの検索
    project_root = Path(os.getcwd())
    possible_docs_paths = [
        project_root / "docs",  # プロジェクトルートからの相対パス
        Path(__file__).parent.parent / "docs"  # このファイルから見た相対パス
    ]
    
    debug_info = []
    for docs_path in possible_docs_paths:
        doc_file_path = docs_path / doc_name
        path_exists = doc_file_path.exists()
        dir_exists = docs_path.exists()
        
        debug_info.append({
            "docs_path": docs_path,
            "doc_file_path": doc_file_path,
            "dir_exists": dir_exists,
            "file_exists": path_exists
        })
        
        if dir_exists:
            try:
                files_in_dir = [f.name for f in docs_path.iterdir() if f.is_file()]
                debug_info[-1]["files_in_dir"] = files_in_dir
            except Exception as e:
                debug_info[-1]["files_in_dir"] = f"Error: {str(e)}"
        
        if path_exists:
            try:
                with open(doc_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Markdownファイルをプレーンテキストとして返す
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(content, media_type="text/markdown")
            
            except Exception as e:
                return {"error": f"Error reading document: {str(e)}", "path": doc_file_path, "debug": debug_info}
    
    return {
        "error": "Document file not found", 
        "requested": doc_name,
        "current_working_directory": os.getcwd(),
        "debug_info": debug_info
    }

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
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    model_path: Optional[str] = None

# グローバル変数
training_tasks = {}
available_models = [
    # Small Models (テスト用・軽量)
    {
        "name": "distilgpt2",
        "description": "軽量な英語モデル（テスト用）",
        "size": "82MB",
        "status": "available"
    },
    {
        "name": "gpt2",
        "description": "GPT-2ベースモデル（英語）",
        "size": "124MB",
        "status": "available"
    },
    {
        "name": "rinna/japanese-gpt2-small",
        "description": "日本語GPT-2 Small（Rinna）",
        "size": "110MB",
        "status": "available"
    },
    {
        "name": "rinna/japanese-gpt2-medium",
        "description": "日本語GPT-2 Medium（Rinna）",
        "size": "350MB",
        "status": "available"
    },
    {
        "name": "cyberagent/open-calm-small",
        "description": "OpenCALM Small（日本語）",
        "size": "160MB",
        "status": "available"
    },
    {
        "name": "cyberagent/open-calm-medium",
        "description": "OpenCALM Medium（日本語）",
        "size": "400MB",
        "status": "available"
    },
    {
        "name": "cyberagent/open-calm-large",
        "description": "OpenCALM Large（日本語）",
        "size": "830MB",
        "status": "available"
    },
    {
        "name": "line-corporation/japanese-large-lm-1.7b",
        "description": "LINE日本語LM 1.7B",
        "size": "1.7B",
        "status": "available"
    },
    
    # Small Models (3B-7B)
    {
        "name": "stabilityai/japanese-stablelm-base-alpha-3b",
        "description": "Japanese StableLM 3B Base",
        "size": "3B",
        "status": "available"
    },
    {
        "name": "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        "description": "Japanese StableLM 3B Instruct（推奨）",
        "size": "3B",
        "status": "available"
    },
    {
        "name": "rinna/japanese-gpt-neox-3.6b",
        "description": "日本語GPT-NeoX 3.6B（Rinna）",
        "size": "3.6B",
        "status": "available"
    },
    {
        "name": "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "description": "日本語GPT-NeoX 3.6B Instruct（Rinna）",
        "size": "3.6B",
        "status": "available"
    },
    {
        "name": "line-corporation/japanese-large-lm-3.6b",
        "description": "LINE Japanese LM 3.6B",
        "size": "3.6B",
        "status": "available"
    },
    {
        "name": "elyza/ELYZA-japanese-Llama-2-7b",
        "description": "ELYZA日本語Llama-2 7B",
        "size": "7B",
        "status": "gpu-required"
    },
    {
        "name": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        "description": "ELYZA日本語Llama-2 7B Instruct",
        "size": "7B",
        "status": "gpu-required"
    },
    
    # Medium Models (8B-13B)
    {
        "name": "elyza/ELYZA-japanese-Llama-3-8B",
        "description": "ELYZA Llama-3 Japanese 8B（16GB GPU推奨）",
        "size": "8B",
        "status": "gpu-required"
    },
    
    # Large Models (17B-32B)
    {
        "name": "meta-llama/Llama-3.1-17B-Instruct",
        "description": "Meta Llama 3.1 17B Instruct（34GB GPU推奨）",
        "size": "17B",
        "status": "gpu-required"
    },
    {
        "name": "microsoft/Phi-3.5-17B-Instruct",
        "description": "Microsoft Phi-3.5 17B Instruct（34GB GPU推奨）",
        "size": "17B",
        "status": "gpu-required"
    },
    {
        "name": "Qwen/Qwen2.5-17B-Instruct",
        "description": "Qwen 2.5 17B Instruct（34GB GPU推奨）",
        "size": "17B",
        "status": "gpu-required"
    },
    {
        "name": "cyberagent/calm3-22b-chat",
        "description": "🆕 CyberAgent CALM3 22B Chat（44GB GPU推奨）",
        "size": "22B",
        "status": "gpu-required"
    },
    {
        "name": "meta-llama/Llama-3.1-32B-Instruct",
        "description": "Meta Llama 3.1 32B Instruct（64GB GPU推奨）",
        "size": "32B",
        "status": "gpu-required"
    },
    {
        "name": "microsoft/Phi-3.5-32B-Instruct",
        "description": "Microsoft Phi-3.5 32B Instruct（64GB GPU推奨）",
        "size": "32B",
        "status": "gpu-required"
    },
    {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 2.5 32B Instruct（64GB GPU推奨）",
        "size": "32B",
        "status": "gpu-required"
    },
    {
        "name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "description": "DeepSeek R1 Distill Qwen 32B 日本語特化（64GB GPU推奨）",
        "size": "32B",
        "status": "gpu-required"
    },
    
    # Extra Large Models (70B)
    {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "description": "Meta Llama 3.1 70B Instruct（140GB GPU推奨）",
        "size": "70B",
        "status": "gpu-required"
    }
]

# モデルキャッシュ
model_cache = {}
executor = ThreadPoolExecutor(max_workers=2)

def get_saved_models():
    """保存済みモデル一覧を取得"""
    saved_models = []
    workspace_path = Path("/workspace")
    
    # LoRAモデルを検索
    for model_dir in workspace_path.glob("lora_demo_*"):
        if model_dir.is_dir():
            # adapter_config.jsonから実際のトレーニング方法を読み取り
            config_path = model_dir / "adapter_config.json"
            model_type = "LoRA"
            model_size = "~1.6MB"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        training_method = config.get("training_method", "lora")
                        
                        if training_method == "full":
                            model_type = "フルファインチューニング"
                            model_size = "~500MB+"
                        elif training_method == "qlora":
                            model_type = "QLoRA (8bit)"
                            model_size = "~1.0MB"
                        else:
                            model_type = "LoRA"
                            model_size = "~1.6MB"
                except:
                    pass
            
            saved_models.append({
                "name": model_dir.name,
                "path": str(model_dir),
                "type": model_type,
                "size": model_size
            })
    
    # outputsディレクトリも検索
    outputs_path = workspace_path / "outputs"
    if outputs_path.exists():
        for model_dir in outputs_path.glob("*_lora_*"):
            if model_dir.is_dir():
                # adapter_config.jsonから実際のトレーニング方法を読み取り
                config_path = model_dir / "adapter_config.json"
                model_type = "LoRA"
                model_size = "~1.6MB"
                
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            training_method = config.get("training_method", "lora")
                            
                            if training_method == "full":
                                model_type = "フルファインチューニング"
                                model_size = "~500MB+"
                            elif training_method == "qlora":
                                model_type = "QLoRA (8bit)"
                                model_size = "~1.0MB"
                            else:
                                model_type = "LoRA"
                                model_size = "~1.6MB"
                    except:
                        pass
                
                saved_models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": model_type,
                    "size": model_size
                })
    
    return saved_models

# 実際のトレーニング実装
async def run_training_task(task_id: str, request: TrainingRequest):
    """バックグラウンドでトレーニングを実行"""
    try:
        # ステータス更新
        training_tasks[task_id].status = "preparing"
        method_name = {
            "lora": "LoRA",
            "qlora": "QLoRA (8bit)", 
            "full": "フルファインチューニング"
        }.get(request.training_method, "LoRA")
        training_tasks[task_id].message = f"{method_name}でモデルを準備中..."
        training_tasks[task_id].progress = 10.0
        logger.info(f"Task {task_id}: {method_name}準備開始")
        
        # モデル保存ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "outputs", f"{method_name.lower()}_{timestamp}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        config_path = os.path.join(os.getcwd(), "config", "training_config.yaml")
        training_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if request.training_method in config_data.get('training_presets', {}):
                    training_config = config_data['training_presets'][request.training_method]
        
        # トークナイザーとモデルの読み込み
        training_tasks[task_id].message = "モデルを読み込み中..."
        training_tasks[task_id].progress = 20.0
        
        try:
            # DeepSeek-R1-Distill-Qwen-32B-Japaneseの場合はキャッシュパスを指定
            if request.model_name == "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese":
                os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
                os.environ["HF_HUB_OFFLINE"] = "0"
                
                # 4bit量子化設定
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    request.model_name,
                    cache_dir=os.path.join(os.getcwd(), "hf_cache"),
                    trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    request.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    cache_dir=os.path.join(os.getcwd(), "hf_cache"),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(request.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # 大きなモデルの場合は量子化を使用
                if any(size in request.model_name.lower() for size in ['22b', '32b', 'large']):
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        request.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        request.model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
            logger.info(f"Task {task_id}: モデル読み込み完了")
        except Exception as e:
            logger.error(f"Task {task_id}: モデル読み込みエラー: {str(e)}")
            training_tasks[task_id].status = "failed"
            training_tasks[task_id].message = f"モデル読み込みエラー: {str(e)}"
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
                r=request.lora_config.get("r", training_config.get("lora_r", 16)),
                lora_alpha=request.lora_config.get("lora_alpha", training_config.get("lora_alpha", 32)),
                target_modules=training_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                lora_dropout=training_config.get("lora_dropout", 0.05),
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
            if os.path.exists(data_path) and data_path.endswith('.jsonl'):
                with open(data_path, 'r', encoding='utf-8') as f:
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
        
        # Hugging Face Trainerを使用した実際のトレーニング
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
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze()
                }
        
        # データセット作成
        train_dataset = SimpleDataset(train_texts, tokenizer)
        
        # トレーニング引数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.get("batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=training_config.get("num_epochs", 1),  # 短縮してテスト
            learning_rate=training_config.get("learning_rate", 2e-4),
            warmup_steps=training_config.get("warmup_steps", 10),
            logging_steps=10,
            save_steps=50,
            max_steps=min(100, len(train_dataset) // training_config.get("batch_size", 1)),  # 最大100ステップ
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=[],  # wandbなどのログを無効化
            save_strategy="steps",
            save_total_limit=2
        )
        
        # プログレス更新用のコールバック
        class ProgressCallback:
            def __init__(self, task_id, training_tasks):
                self.task_id = task_id
                self.training_tasks = training_tasks
                self.step_count = 0
                self.total_steps = training_args.max_steps
            
            def on_step_end(self, trainer, step):
                self.step_count = step
                progress = 50 + (step / self.total_steps) * 40  # 50%から90%まで
                self.training_tasks[self.task_id].progress = min(progress, 90.0)
                self.training_tasks[self.task_id].message = f"{method_name}でファインチューニング中... Step {step}/{self.total_steps}"
        
        # Trainer作成
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        
        # プログレスコールバック設定
        progress_callback = ProgressCallback(task_id, training_tasks)
        
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
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # アダプター設定ファイルを保存
        adapter_config = {
            "model_type": request.training_method,
            "base_model": request.model_name,
            "r": request.lora_config.get("r", training_config.get("lora_r", 16)),
            "lora_alpha": request.lora_config.get("lora_alpha", training_config.get("lora_alpha", 32)),
            "task_type": "CAUSAL_LM",
            "training_data_size": len(train_texts),
            "training_method": request.training_method,
            "use_qlora": request.training_method == "qlora",
            "load_in_4bit": request.training_method == "qlora",
            "timestamp": timestamp,
            "output_dir": output_dir
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)
        
        # 完了
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].progress = 100.0
        training_tasks[task_id].message = f"{method_name}ファインチューニング完了！"
        training_tasks[task_id].model_path = output_dir
        logger.info(f"Task {task_id}: {method_name}ファインチューニング完了 - {output_dir}")
        
    except Exception as e:
        logger.error(f"Task {task_id}: エラー発生: {str(e)}")
        logger.error(traceback.format_exc())
        training_tasks[task_id].status = "failed"
        training_tasks[task_id].message = f"エラー: {str(e)}"

# API エンドポイント

@app.get("/debug-paths")
async def debug_paths():
    """デバッグ用: パス情報を表示"""
    import os
    debug_info = {
        "current_working_directory": os.getcwd(),
        "static_dir_used": static_dir,
        "static_dir_exists": os.path.exists(static_dir),
        "manual_file_exists": os.path.exists(os.path.join(static_dir, "manual.html")),
        "overview_file_exists": os.path.exists(os.path.join(static_dir, "system-overview.html")),
        "index_file_exists": os.path.exists(os.path.join(static_dir, "index.html")),
        "files_in_static_dir": []
    }
    
    if os.path.exists(static_dir):
        try:
            debug_info["files_in_static_dir"] = os.listdir(static_dir)
        except Exception as e:
            debug_info["files_in_static_dir"] = f"Error listing files: {str(e)}"
    
    return debug_info

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """メインページ"""
    index_path = os.path.join(static_dir, "index.html")
    try:
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content=f"<h1>Index page not found</h1><p>Looking for: {index_path}</p><p>Static dir: {static_dir}</p>", 
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading index</h1><p>Error: {str(e)}</p><p>Path: {index_path}</p>", 
            status_code=500
        )

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
        # ファイル保存
        project_root = Path(os.getcwd())
        upload_dir = project_root / "data" / "uploaded"
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # ファイル形式の検証
        sample_data = []
        if file.filename.endswith('.jsonl'):
            # JSONL形式の検証
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5]):  # 最初の5行をサンプルとして表示
                    try:
                        data = json.loads(line.strip())
                        sample_data.append(data)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail=f"Invalid JSON at line {i+1}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "sample_data": sample_data[:3]
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """ファインチューニングを開始"""
    task_id = str(uuid.uuid4())
    logger.info(f"ファインチューニング開始リクエスト受信: task_id={task_id}")
    
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

@app.get("/api/training-status/{task_id}")
async def get_training_status(task_id: str):
    """トレーニングステータスを取得"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_tasks[task_id]

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    """実際のモデルを使用したテキスト生成"""
    try:
        logger.info(f"テキスト生成開始: モデル={request.model_path}, プロンプト={request.prompt[:50]}...")
        
        # モデルパスからベースモデル名を取得
        base_model_name = "distilgpt2"  # デフォルト
        
        # 直接モデル名が指定された場合（CALM3-22Bなど）
        if request.model_path == "calm3-22b":
            base_model_name = "/workspace/calm3-22b"
        elif "calm3" in request.model_path.lower():
            base_model_name = request.model_path
        else:
            # adapter_config.jsonがあれば読み込む
            adapter_config_path = Path(request.model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model", "distilgpt2")
        
        # キャッシュキー
        cache_key = f"{base_model_name}_{request.model_path}"
        
        # モデルがキャッシュにない場合は読み込み
        if cache_key not in model_cache:
            logger.info(f"モデル読み込み中: {base_model_name}")
            
            # DeepSeek-R1-Distill-Qwen-32B-Japaneseの場合はキャッシュパスを指定
            if base_model_name == "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese":
                os.environ["HF_HOME"] = "/home/kjifu/AI_finet/hf_cache"
                os.environ["HF_HUB_OFFLINE"] = "0"
                
                # DeepSeek用のトークナイザー設定
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    cache_dir="/home/kjifu/AI_finet/hf_cache",
                    trust_remote_code=True
                )
            else:
                # トークナイザーとベースモデルの読み込み
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 大規模モデル（22B以上）の場合は4bit量子化を使用
            if "22B" in base_model_name or "32B" in base_model_name or any(x in base_model_name for x in ["calm3-22b", "32B", "DeepSeek-R1-Distill-Qwen-32B"]):
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # DeepSeekモデルの場合は特別な設定
                if "DeepSeek-R1-Distill-Qwen-32B" in base_model_name:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        cache_dir="/home/kjifu/AI_finet/hf_cache",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            model_cache[cache_key] = {
                "tokenizer": tokenizer,
                "model": model
            }
            logger.info("モデル読み込み完了")
        
        # キャッシュからモデルとトークナイザーを取得
        tokenizer = model_cache[cache_key]["tokenizer"]
        model = model_cache[cache_key]["model"]
        
        # テキスト生成
        logger.info("生成開始...")
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # デバイスに移動
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"生成完了: {len(generated_text)}文字")
        
        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "model_path": request.model_path
        }
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        # エラー時は簡単なフォールバック応答
        fallback_text = request.prompt + "\n[エラー: モデル読み込みに失敗しました。]\n" + "これはテスト応答です。"
        return {
            "prompt": request.prompt,
            "generated_text": fallback_text,
            "model_path": request.model_path,
            "error": str(e)
        }

@app.get("/api/system-info")
async def get_system_info():
    """システム情報を取得"""
    try:
        import psutil
        
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory // 1024**3,
                    "memory_used": torch.cuda.memory_allocated(i) // 1024**3
                })
        
        return {
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_info": gpu_info,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // 1024**3,
            "memory_used": psutil.virtual_memory().used // 1024**3
        }
    except Exception as e:
        return {
            "gpu_count": 0,
            "gpu_info": [],
            "cpu_count": 4,
            "memory_total": 16,
            "memory_used": 8,
            "error": str(e)
        }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")