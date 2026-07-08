# DeepSpeed Configuration Files

This directory contains optimized DeepSpeed configurations for different model sizes:

## Configuration Files

- **ds_config_ultra_large.json**: For 50B+ parameter models
  - Uses ZeRO Stage 3 with CPU+NVMe offloading
  - Includes weight quantization for memory efficiency
  - Aggressive memory optimization settings

- **ds_config_large.json**: For 30B-50B parameter models
  - Uses ZeRO Stage 3 with CPU offloading
  - NVMe offloading for parameters
  - Balanced performance and memory usage

- **ds_config_medium.json**: For 17B-30B parameter models
  - Uses ZeRO Stage 2 with optimizer offloading
  - Good balance of speed and memory efficiency

- **ds_config_small.json**: For models under 17B parameters
  - Uses ZeRO Stage 1 for basic optimization
  - Minimal overhead for smaller models

## Usage

Select the appropriate configuration based on your model size:

```bash
# For 32B model training
python scripts/train_large_model.py \
    --model_name cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B \
    --use_deepspeed \
    --deepspeed_config configs/deepspeed/ds_config_large.json \
    --dataset_path data/training_data.jsonl
```

## Memory Requirements

- Ultra Large (50B+): 8x A100 80GB or equivalent
- Large (30B-50B): 4x A100 80GB or equivalent
- Medium (17B-30B): 2x A100 80GB or equivalent
- Small (<17B): 1x A100 80GB or equivalent

Note: Actual requirements depend on batch size, sequence length, and other factors.
