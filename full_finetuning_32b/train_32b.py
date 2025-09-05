#!/usr/bin/env python3
"""
32B Model Full Fine-tuning Launcher
Command-line interface for training large language models
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description="32B Model Full Fine-tuning Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train_32b.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path data/training_data.jsonl

  # Training with 4-bit quantization for memory efficiency
  python train_32b.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path data/training_data.jsonl --use_4bit

  # Multi-GPU training with DeepSpeed
  python train_32b.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path data/training_data.jsonl --use_deepspeed --deepspeed_config configs/deepspeed_config.json

  # Training with custom hyperparameters
  python train_32b.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path data/training_data.jsonl --num_epochs 5 --learning_rate 1e-5 --batch_size 2
        """
    )

    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Model name or path (HuggingFace model ID or local path)"
    )
    
    # Dataset configuration
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
        help="Validation split ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    # Memory optimization
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for memory efficiency"
    )
    
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (default: True)"
    )
    
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offloading for large models"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision (recommended for A100)"
    )
    
    # DeepSpeed configuration
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training"
    )
    
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/deepspeed_zero3.json",
        help="Path to DeepSpeed configuration file"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/model_name_timestamp)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        model_name = args.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/{model_name}_{timestamp}"
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "src/full_finetuning.py",
        "--model_name_or_path", args.model_name,
        "--dataset_path", args.dataset_path,
        "--output_dir", args.output_dir,
        "--validation_split", str(args.validation_split),
        "--max_seq_length", str(args.max_seq_length),
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--warmup_steps", str(args.warmup_steps),
        "--weight_decay", str(args.weight_decay),
        "--save_steps", str(args.save_steps),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
    ]
    
    # Add boolean flags
    if args.use_4bit:
        cmd.append("--use_4bit")
    if args.use_8bit:
        cmd.append("--use_8bit")
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.cpu_offload:
        cmd.append("--cpu_offload")
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    if args.early_stopping:
        cmd.append("--early_stopping")
        cmd.extend(["--early_stopping_patience", str(args.early_stopping_patience)])
    
    # Add DeepSpeed configuration
    if args.use_deepspeed:
        cmd.append("--use_deepspeed")
        cmd.extend(["--deepspeed_config", args.deepspeed_config])
        
        # For multi-GPU training with DeepSpeed
        if torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs. Using DeepSpeed launcher...")
            deepspeed_cmd = [
                "deepspeed",
                "--num_gpus", str(torch.cuda.device_count()),
            ] + cmd[1:]  # Remove python executable
            cmd = deepspeed_cmd
    
    # Add local rank for distributed training
    if args.local_rank != -1:
        cmd.extend(["--local_rank", str(args.local_rank)])
    
    # Print configuration
    print("="*60)
    print("32B Model Full Fine-tuning Configuration")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    
    if args.use_4bit:
        print("Memory Optimization: 4-bit Quantization")
    elif args.use_8bit:
        print("Memory Optimization: 8-bit Quantization")
    
    if args.use_deepspeed:
        print(f"DeepSpeed: Enabled (Config: {args.deepspeed_config})")
    
    print("="*60)
    print()
    
    # Execute training
    print("Starting training...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        import torch
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        print(f"Model saved to: {args.output_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return 1
    except ImportError:
        print("\nError: PyTorch not found. Please install required dependencies:")
        print("pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())