#!/usr/bin/env python3
"""
Large Model Fine-tuning Script
Supports models up to 32B parameters with memory-efficient techniques
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset
import deepspeed

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.japanese_model import JapaneseModel
from src.utils.training_utils import setup_logging, save_training_info
from src.utils.gpu_utils import get_gpu_memory_info, log_gpu_usage

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune large language models (up to 32B)")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B",
        help="Model name from HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/large_model",
        help="Directory to save the fine-tuned model"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training dataset (JSONL format)"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--qlora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--qlora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        help="Path to DeepSpeed config file"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offloading for optimizer and parameters"
    )
    parser.add_argument(
        "--disk_offload_dir",
        type=str,
        default="./offload",
        help="Directory for disk offloading"
    )
    
    # Other arguments
    parser.add_argument(
        "--auth_token",
        type=str,
        help="HuggingFace authentication token"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    
    return parser.parse_args()


def load_and_prepare_dataset(args, tokenizer):
    """Load and prepare dataset for training"""
    logger.info(f"Loading dataset from {args.dataset_path}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    
    def tokenize_function(examples):
        # Format as instruction-response pairs
        texts = []
        for instruction, response in zip(examples["instruction"], examples["response"]):
            if "input" in examples and examples["input"]:
                text = f"指示: {instruction}\n入力: {examples['input']}\n回答: {response}"
            else:
                text = f"指示: {instruction}\n回答: {response}"
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Tokenize datasets
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing training data"
    )
    
    tokenized_val = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing validation data"
    )
    
    return tokenized_train, tokenized_val


def get_training_args(args, model_config):
    """Get training arguments with optimizations for large models"""
    
    # Base training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_dir=f"{args.output_dir}/logs",
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # DeepSpeed configuration
    if args.use_deepspeed:
        if args.deepspeed_config:
            training_args.deepspeed = args.deepspeed_config
        else:
            # Use auto-generated config from model
            deepspeed_config = model_config.get("deepspeed", {})
            if deepspeed_config:
                # Save config to file
                config_path = f"{args.output_dir}/deepspeed_config.json"
                os.makedirs(args.output_dir, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(deepspeed_config, f, indent=2)
                training_args.deepspeed = config_path
    
    # Memory optimizations for large models
    if model_config.get("model_parallel", False):
        training_args.fsdp = "full_shard auto_wrap"
        training_args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"
    
    return training_args


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger.info(f"Starting large model fine-tuning with args: {args}")
    
    # Set seed
    set_seed(args.seed)
    
    # Log GPU info
    gpu_info = get_gpu_memory_info()
    logger.info(f"GPU info: {gpu_info}")
    
    # Initialize model
    logger.info(f"Loading model: {args.model_name}")
    model_wrapper = JapaneseModel(
        model_name=args.model_name,
        load_in_8bit=args.use_8bit,
        load_in_4bit=args.use_4bit,
        use_flash_attention=True,
        gradient_checkpointing=args.gradient_checkpointing,
        use_auth_token=args.auth_token,
        use_qlora=args.use_qlora,
        qlora_r=args.qlora_r,
        qlora_alpha=args.qlora_alpha,
        enable_deepspeed=args.use_deepspeed,
        cpu_offload=args.cpu_offload,
        disk_offload_dir=args.disk_offload_dir
    )
    
    # Get recommended config
    model_config = model_wrapper.get_recommended_training_config()
    logger.info(f"Recommended training config: {json.dumps(model_config, indent=2)}")
    
    # Load model and tokenizer
    model = model_wrapper.load_model()
    tokenizer = model_wrapper.load_tokenizer()
    
    # Log model info
    if hasattr(model, "get_memory_footprint"):
        memory_footprint = model.get_memory_footprint() / 1024**3
        logger.info(f"Model memory footprint: {memory_footprint:.2f} GB")
    
    # Prepare dataset
    train_dataset, val_dataset = load_and_prepare_dataset(args, tokenizer)
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Get training arguments
    training_args = get_training_args(args, model_config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Log initial GPU usage
    log_gpu_usage("Before training")
    
    # Train
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Save model
    logger.info("Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    save_training_info(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_result=train_result,
        args=args
    )
    
    # Log final GPU usage
    log_gpu_usage("After training")
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")
    
    # Evaluate final model
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(f"{args.output_dir}/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()