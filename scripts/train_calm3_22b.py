#!/usr/bin/env python3
"""
CALM3-22B Model Fine-tuning Script
Optimized for CyberAgent's CALM3 22B parameter model
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
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.training_utils import (
    setup_logging,
    save_training_info,
    get_gpu_memory_info,
    log_gpu_usage
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CALM3-22B model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="./calm3-22b",
        help="Path to CALM3-22B model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/calm3-22b-finetuned",
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
        default=2048,
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
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    
    # LoRA/QLoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization + LoRA)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Memory optimization arguments
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/deepspeed/ds_config_large.json",
        help="Path to DeepSpeed config file"
    )
    
    # Other arguments
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
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load CALM3-22B model and tokenizer with optimizations"""
    logger.info(f"Loading CALM3-22B model from {args.model_path}")
    
    # Load model configuration
    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading configuration
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    # QLoRA configuration
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["quantization_config"] = bnb_config
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare model for training
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    if args.use_lora or args.use_qlora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    """Load and prepare dataset for training"""
    logger.info(f"Loading dataset from {args.dataset_path}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    
    def tokenize_function(examples):
        # Format as chat conversation
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            response = examples["response"][i]
            
            # CALM3 chat format
            text = f"USER: {instruction}\nASSISTANT: {response}"
            
            # Add input context if available
            if "input" in examples and examples["input"][i]:
                text = f"USER: {instruction}\n{examples['input'][i]}\nASSISTANT: {response}"
            
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


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logger.info(f"Starting CALM3-22B fine-tuning with args: {args}")
    
    # Set seed
    set_seed(args.seed)
    
    # Log GPU info
    gpu_info = get_gpu_memory_info()
    logger.info(f"GPU info: {gpu_info}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Prepare dataset
    train_dataset, val_dataset = load_and_prepare_dataset(args, tokenizer)
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
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
        optim="adamw_torch",
        logging_dir=f"{args.output_dir}/logs",
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
    )
    
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
    train_result = trainer.train()
    
    # Save model
    logger.info("Saving fine-tuned model...")
    trainer.save_model()
    
    # Save full model if using LoRA
    if args.use_lora or args.use_qlora:
        logger.info("Merging and saving full model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(f"{args.output_dir}/merged")
        tokenizer.save_pretrained(f"{args.output_dir}/merged")
    else:
        tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    save_training_info(
        output_dir=args.output_dir,
        model_name="cyberagent/calm3-22b-chat",
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
        json.dump(eval_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()