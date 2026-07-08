# 32B Model Full Fine-tuning Standalone Package

This directory contains a standalone implementation for full fine-tuning of 32B parameter language models, designed to run on separate hardware with high memory capacity.

## Directory Structure

```
full_finetuning_32b/
├── src/
│   └── full_finetuning.py      # Core training implementation
├── configs/
│   ├── deepspeed_zero3.json    # DeepSpeed ZeRO-3 configuration
│   └── training_config.yaml    # Training hyperparameters
├── data/                        # Training datasets
├── outputs/                     # Trained models and checkpoints
├── logs/                        # Training logs
├── scripts/                     # Utility scripts
├── train_32b.py                 # Main launcher script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

```bash
# Navigate to the directory
cd full_finetuning_32b

# Install dependencies
pip install -r requirements.txt

# For DeepSpeed support (optional but recommended)
pip install deepspeed
```

## Usage

### Basic Training

```bash
python train_32b.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dataset_path data/training_data.jsonl \
  --num_epochs 3 \
  --learning_rate 2e-5
```

### Memory-Efficient Training (4-bit Quantization)

```bash
python train_32b.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dataset_path data/training_data.jsonl \
  --use_4bit \
  --gradient_checkpointing
```

### Multi-GPU Training with DeepSpeed

```bash
python train_32b.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dataset_path data/training_data.jsonl \
  --use_deepspeed \
  --deepspeed_config configs/deepspeed_zero3.json
```

### CPU Offloading for Limited GPU Memory

```bash
python train_32b.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dataset_path data/training_data.jsonl \
  --cpu_offload \
  --gradient_checkpointing \
  --use_4bit
```

## Dataset Format

The training data should be in JSONL format with the following structure:

### Option 1: Simple text format
```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

### Option 2: Instruction-Response format
```json
{"instruction": "質問や指示", "output": "回答"}
{"instruction": "別の質問", "output": "別の回答"}
```

## Command-Line Options

### Essential Parameters
- `--model_name`: HuggingFace model ID or local path
- `--dataset_path`: Path to training data (JSONL format)
- `--output_dir`: Directory for saving trained model
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization

### Memory Optimization
- `--use_4bit`: Enable 4-bit quantization
- `--use_8bit`: Enable 8-bit quantization
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--cpu_offload`: Offload model parameters to CPU
- `--fp16/--bf16`: Use mixed precision training

### Training Configuration
- `--batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--warmup_steps`: Number of warmup steps
- `--weight_decay`: Weight decay coefficient
- `--save_steps`: Save checkpoint every N steps

### Distributed Training
- `--use_deepspeed`: Enable DeepSpeed
- `--deepspeed_config`: Path to DeepSpeed configuration

## Hardware Requirements

### Minimum Requirements
- GPU: 1x NVIDIA A100 80GB or equivalent
- RAM: 256GB system memory
- Storage: 500GB free space

### Recommended Setup
- GPU: 2x NVIDIA A100 80GB
- RAM: 512GB system memory
- Storage: 1TB NVMe SSD

### Memory-Constrained Options
With 4-bit quantization and CPU offloading:
- GPU: 1x NVIDIA A100 40GB or 2x RTX A6000
- RAM: 512GB system memory (for CPU offloading)

## Monitoring Training

Training logs are saved to:
- TensorBoard logs: `outputs/[model_name]/logs/`
- Text logs: `outputs/[model_name]/logs/training_[timestamp].log`

View with TensorBoard:
```bash
tensorboard --logdir outputs/[model_name]/logs/
```

## Output Files

After training, the following files are saved:
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `config.json`: Model configuration
- `tokenizer_config.json`: Tokenizer configuration
- `training_metrics.json`: Training metrics
- `training_args.bin`: Training arguments

## Troubleshooting

### Out of Memory (OOM) Errors
1. Enable 4-bit quantization: `--use_4bit`
2. Reduce batch size: `--batch_size 1`
3. Increase gradient accumulation: `--gradient_accumulation_steps 32`
4. Enable CPU offloading: `--cpu_offload`
5. Use DeepSpeed ZeRO-3: `--use_deepspeed`

### Slow Training
1. Disable CPU offloading if not needed
2. Use bf16 instead of fp16 on A100: `--bf16`
3. Reduce logging frequency
4. Use faster optimizer: AdamW with 8-bit states

### DeepSpeed Issues
1. Ensure DeepSpeed is installed: `pip install deepspeed`
2. Check CUDA compatibility
3. Verify configuration file path
4. Use appropriate ZeRO stage for your setup

## Advanced Configuration

Edit `configs/training_config.yaml` for detailed control over:
- Learning rate scheduling
- Optimizer parameters
- Evaluation strategies
- Checkpointing policies
- Monitoring settings

## Support

For issues specific to 32B model training, check:
1. GPU memory usage: `nvidia-smi`
2. System memory: `free -h`
3. Disk space: `df -h`
4. Training logs in `outputs/[model_name]/logs/`