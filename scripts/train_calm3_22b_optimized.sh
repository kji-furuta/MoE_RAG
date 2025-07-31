#!/bin/bash
# CALM3-22B Optimized Training Script
# This script provides optimized training configurations for different hardware setups

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="./calm3-22b"
DATASET_PATH="data/processed/training_data.jsonl"
OUTPUT_DIR="outputs/calm3-22b-finetuned"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check GPU memory
check_gpu_memory() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. GPU training requires NVIDIA GPU and drivers."
        exit 1
    fi
    
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    print_info "GPU Memory: ${GPU_MEMORY}MB"
    
    if [ $GPU_MEMORY -lt 24000 ]; then
        print_warning "GPU memory is less than 24GB. QLoRA training is recommended."
        RECOMMENDED_MODE="qlora"
    elif [ $GPU_MEMORY -lt 40000 ]; then
        print_info "GPU memory is suitable for LoRA training."
        RECOMMENDED_MODE="lora"
    else
        print_info "GPU memory is sufficient for full fine-tuning."
        RECOMMENDED_MODE="full"
    fi
}

# Function to train with QLoRA (most memory efficient)
train_qlora() {
    print_info "Starting QLoRA training (4-bit quantization + LoRA)"
    
    python "$PROJECT_ROOT/scripts/train_calm3_22b.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${OUTPUT_DIR}_qlora" \
        --use_qlora \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --num_epochs 3 \
        --max_seq_length 2048 \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 100 \
        --gradient_checkpointing \
        --use_flash_attention \
        --seed 42
}

# Function to train with LoRA
train_lora() {
    print_info "Starting LoRA training"
    
    python "$PROJECT_ROOT/scripts/train_calm3_22b.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${OUTPUT_DIR}_lora" \
        --use_lora \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 3e-4 \
        --num_epochs 3 \
        --max_seq_length 2048 \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 100 \
        --gradient_checkpointing \
        --use_flash_attention \
        --seed 42
}

# Function to train with full fine-tuning
train_full() {
    print_info "Starting full fine-tuning"
    
    python "$PROJECT_ROOT/scripts/train_calm3_22b.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${OUTPUT_DIR}_full" \
        --batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-5 \
        --num_epochs 3 \
        --max_seq_length 2048 \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 100 \
        --gradient_checkpointing \
        --use_flash_attention \
        --use_deepspeed \
        --deepspeed_config "configs/deepspeed/ds_config_large.json" \
        --seed 42
}

# Function to train with DeepSpeed (multi-GPU)
train_deepspeed() {
    print_info "Starting DeepSpeed multi-GPU training"
    
    deepspeed --num_gpus=$1 "$PROJECT_ROOT/scripts/train_calm3_22b.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${OUTPUT_DIR}_deepspeed" \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_epochs 3 \
        --max_seq_length 2048 \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 100 \
        --gradient_checkpointing \
        --use_flash_attention \
        --use_deepspeed \
        --deepspeed_config "configs/deepspeed/ds_config_large.json" \
        --seed 42
}

# Main function
main() {
    print_info "CALM3-22B Fine-tuning Script"
    print_info "=============================="
    
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model not found at $MODEL_PATH"
        print_info "Please run the download script first:"
        print_info "python download_models.py"
        exit 1
    fi
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        print_error "Dataset not found at $DATASET_PATH"
        print_info "Please prepare your training data first:"
        print_info "python scripts/prepare_training_data.py"
        exit 1
    fi
    
    # Check GPU memory
    check_gpu_memory
    
    # Parse command line arguments
    case "${1:-auto}" in
        "qlora")
            train_qlora
            ;;
        "lora")
            train_lora
            ;;
        "full")
            train_full
            ;;
        "deepspeed")
            NUM_GPUS=${2:-2}
            train_deepspeed $NUM_GPUS
            ;;
        "auto")
            print_info "Auto-selecting training mode based on GPU memory: $RECOMMENDED_MODE"
            case $RECOMMENDED_MODE in
                "qlora")
                    train_qlora
                    ;;
                "lora")
                    train_lora
                    ;;
                "full")
                    train_full
                    ;;
            esac
            ;;
        *)
            echo "Usage: $0 [qlora|lora|full|deepspeed|auto] [num_gpus_for_deepspeed]"
            echo ""
            echo "Training modes:"
            echo "  qlora     - QLoRA training (4-bit quantization + LoRA) - Most memory efficient"
            echo "  lora      - LoRA training - Balanced efficiency and quality"
            echo "  full      - Full fine-tuning - Best quality, high memory usage"
            echo "  deepspeed - Multi-GPU training with DeepSpeed"
            echo "  auto      - Automatically select based on available GPU memory"
            echo ""
            echo "Examples:"
            echo "  $0 qlora"
            echo "  $0 lora"
            echo "  $0 full"
            echo "  $0 deepspeed 2"
            echo "  $0 auto"
            exit 1
            ;;
    esac
    
    print_success "Training completed successfully!"
    print_info "Model saved to: $OUTPUT_DIR"
    print_info "To use the trained model, see: docs/TRAINED_MODEL_USAGE.md"
}

# Run main function
main "$@"