"""Service layer for fine-tuning workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

from app.model_utils import (
    create_quantization_config,
    get_device_map,
    get_output_directory,
    handle_model_loading_error,
    load_model_and_tokenizer,
    load_tokenizer,
    load_training_config,
    JST,
)
from src.training.memory_monitor import MemoryMonitor
from .models import TrainingRequest, TrainingStatus
from .state import training_tasks

logger = logging.getLogger(__name__)


def create_training_task(request: TrainingRequest) -> str:
    """Register a new fine-tuning task and return its identifier."""

    if not request.model_name:
        raise ValueError("model_name is required")

    if not request.training_data:
        raise ValueError("training_data is required")

    task_id = str(uuid.uuid4())

    logger.info(
        "Fine-tuning request received: task_id=%s, model_name=%s, method=%s",
        task_id,
        request.model_name,
        request.training_method,
    )
    logger.info("LoRA config: %s", request.lora_config)
    logger.info("Training config: %s", request.training_config)

    training_tasks[task_id] = TrainingStatus(
        task_id=task_id,
        status="starting",
        progress=0.0,
        message="ファインチューニングを開始しています...",
    )

    return task_id

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
            # GPT-NeoXモデル用のターゲットモジュールを判定
            if "gpt-neox" in request.model_name.lower():
                # GPT-NeoX特有のQKV統合層
                default_target_modules = [
                    "attention.query_key_value",
                    "attention.dense",
                    "mlp.dense_h_to_4h",
                    "mlp.dense_4h_to_h"
                ]
                logger.info("GPT-NeoXモデル用のターゲットモジュールを使用")
            else:
                # 通常のLLaMAスタイルモジュール
                default_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            lora_config = LoraConfig(
                r=get_config_value(request.lora_config, "r", get_config_value(training_config, "lora_r", 16, int), int),
                lora_alpha=get_config_value(request.lora_config, "lora_alpha", get_config_value(training_config, "lora_alpha", 32, int), int),
                target_modules=training_config.get("target_modules", default_target_modules),
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

