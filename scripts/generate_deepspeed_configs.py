#!/usr/bin/env python3
"""
Generate DeepSpeed configurations for different model sizes
"""

import json
import os
from pathlib import Path


def generate_ultra_large_config():
    """Generate DeepSpeed config for 50B+ models"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        
        "wall_clock_breakdown": False,
        "compression_training": {
            "weight_quantization": {
                "shared_parameters": {
                    "enabled": True,
                    "quantizer_kernel": True,
                    "schedule_offset": 0,
                    "quantize_groups": 1,
                    "quantize_verbose": False,
                    "quantization_type": "symmetric",
                    "quantize_weight_in_forward": True,
                    "rounding": "nearest",
                    "fp16_mixed_quantize": False,
                    "quantize_change_ratio": 0.001
                },
                "different_groups": {
                    "wq1": {
                        "params": {
                            "start_bits": 12,
                            "target_bits": 8,
                            "quantization_period": 1000
                        },
                        "modules": ["attention", "self_attention"]
                    }
                }
            }
        }
    }


def generate_large_config():
    """Generate DeepSpeed config for 30B-50B models"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "./offload_nvme",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "gradient_checkpointing": True,
        "wall_clock_breakdown": False
    }


def generate_medium_config():
    """Generate DeepSpeed config for 17B-30B models"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "gradient_checkpointing": True,
        "wall_clock_breakdown": False
    }


def generate_small_config():
    """Generate DeepSpeed config for models under 17B"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "gradient_checkpointing": False,
        "wall_clock_breakdown": False
    }


def main():
    """Generate all DeepSpeed configurations"""
    configs_dir = Path("configs/deepspeed")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        "ds_config_ultra_large.json": generate_ultra_large_config(),
        "ds_config_large.json": generate_large_config(),
        "ds_config_medium.json": generate_medium_config(),
        "ds_config_small.json": generate_small_config()
    }
    
    for filename, config in configs.items():
        filepath = configs_dir / filename
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Generated: {filepath}")
    
    # Generate a README
    readme_content = """# DeepSpeed Configuration Files

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
python scripts/train_large_model.py \\
    --model_name cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B \\
    --use_deepspeed \\
    --deepspeed_config configs/deepspeed/ds_config_large.json \\
    --dataset_path data/training_data.jsonl
```

## Memory Requirements

- Ultra Large (50B+): 8x A100 80GB or equivalent
- Large (30B-50B): 4x A100 80GB or equivalent
- Medium (17B-30B): 2x A100 80GB or equivalent
- Small (<17B): 1x A100 80GB or equivalent

Note: Actual requirements depend on batch size, sequence length, and other factors.
"""
    
    readme_path = configs_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Generated: {readme_path}")


if __name__ == "__main__":
    main()