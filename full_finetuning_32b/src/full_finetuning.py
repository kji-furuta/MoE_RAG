#!/usr/bin/env python3
"""
Full Fine-tuning Script for 32B Models
Optimized for large language models with memory-efficient techniques
"""

import os
import sys
import argparse
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset as HFDataset
import deepspeed
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from peft import prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
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


class MemoryEfficientTrainer:
    """Memory-efficient trainer for large models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            cpu=args.cpu_offload
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with memory optimization"""
        logger.info(f"Loading model: {self.args.model_name_or_path}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for memory efficiency
        quantization_config = None
        if self.args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.args.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load model with memory optimization
        if self.args.use_deepspeed:
            # DeepSpeed ZeRO-3 configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto" if not self.args.cpu_offload else None,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
                trust_remote_code=True
            )
        else:
            # Standard loading with device map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Prepare for training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
            
        logger.info(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def load_dataset(self):
        """Load and prepare dataset"""
        logger.info(f"Loading dataset from: {self.args.dataset_path}")
        
        if self.args.dataset_path.endswith('.jsonl'):
            # Load JSONL file
            with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            # Extract text field
            texts = []
            for item in data:
                if 'text' in item:
                    texts.append(item['text'])
                elif 'instruction' in item and 'output' in item:
                    text = f"### 指示:\\n{item['instruction']}\\n\\n### 回答:\\n{item['output']}"
                    texts.append(text)
                else:
                    logger.warning(f"Skipping item without 'text' field: {item}")
            
            # Split into train and validation
            split_idx = int(len(texts) * (1 - self.args.validation_split))
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            
        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(self.args.dataset_path)
            train_texts = dataset['train']['text']
            val_texts = dataset['validation']['text'] if 'validation' in dataset else []
        
        # Create datasets
        self.train_dataset = TextDataset(train_texts, self.tokenizer, self.args.max_seq_length)
        self.val_dataset = TextDataset(val_texts, self.tokenizer, self.args.max_seq_length) if val_texts else None
        
        logger.info(f"Dataset loaded. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset) if self.val_dataset else 0}")
        
    def setup_training_args(self):
        """Setup training arguments"""
        
        # Calculate total training steps
        num_training_steps = len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps) * self.args.num_epochs
        
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=self.args.gradient_checkpointing,
            warmup_steps=self.args.warmup_steps,
            learning_rate=self.args.learning_rate,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            logging_dir=f"{self.args.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps" if self.val_dataset else "no",
            eval_steps=100 if self.val_dataset else None,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="loss" if self.val_dataset else None,
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"],
            deepspeed=self.args.deepspeed_config if self.args.use_deepspeed else None,
            optim=self.args.optimizer,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=self.args.weight_decay,
            max_grad_norm=self.args.max_grad_norm,
            lr_scheduler_type=self.args.lr_scheduler_type,
            dataloader_num_workers=self.args.num_workers,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        return training_args
        
    def train(self):
        """Main training loop"""
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Callbacks
        callbacks = []
        if self.args.early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.args.early_stopping_patience
            ))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # Log initial GPU memory
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.args.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        # Save training metrics
        metrics_file = Path(self.args.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        
        # Cleanup
        del trainer
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
        return train_result


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Full fine-tuning for 32B models")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="outputs/full_finetuned",
                       help="The output directory where the model predictions and checkpoints will be written")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the training dataset (JSONL format)")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per GPU/CPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Linear warmup over warmup_steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                       help="The scheduler type to use")
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                       help="Optimizer to use")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X updates steps")
    
    # Memory optimization arguments
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Enable CPU offloading")
    parser.add_argument("--fp16", action="store_true",
                       help="Use fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bf16 mixed precision")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    
    # DeepSpeed arguments
    parser.add_argument("--use_deepspeed", action="store_true",
                       help="Use DeepSpeed for distributed training")
    parser.add_argument("--deepspeed_config", type=str,
                       help="Path to DeepSpeed config file")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for initialization")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="For distributed training: local_rank")
    
    return parser.parse_args()


def main():
    """Main function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set seed
    set_seed(args.seed)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = Path(args.output_dir) / "config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize trainer and start training
    trainer = MemoryEfficientTrainer(args)
    result = trainer.train()
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()