#!/bin/bash
# Script to train 32B parameter models with optimal settings

# Model and paths
MODEL_NAME="cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_DIR="outputs/qwen_32b_finetuned"
DATASET_PATH="data/uploaded/test_training_data.jsonl"

# Create output directory
mkdir -p $OUTPUT_DIR

# Log system info
echo "=== System Information ==="
nvidia-smi
echo ""
echo "=== Starting 32B Model Fine-tuning ==="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training with optimal settings for 32B model
python3 scripts/train_large_model.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --validation_split 0.1 \
    --max_seq_length 512 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --use_4bit \
    --use_qlora \
    --qlora_r 128 \
    --qlora_alpha 32 \
    --use_deepspeed \
    --deepspeed_config configs/deepspeed/ds_config_large.json \
    --gradient_checkpointing \
    --cpu_offload \
    --disk_offload_dir "./offload" \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42

echo "=== Training Complete ==="