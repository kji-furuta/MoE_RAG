# 7B Model Fine-tuning Error: "Some modules are dispatched on the CPU or the disk"

## Error Description

When attempting to fine-tune 7B models (such as `elyza/ELYZA-japanese-Llama-2-7b-instruct`), users encounter the following error:

```
Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. 
If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, 
you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to `from_pretrained`.
```

## Root Cause

This error occurs when:
1. The model is too large to fit entirely in GPU memory
2. The automatic device mapping (`device_map="auto"`) attempts to split the model between GPU and CPU
3. Quantization (4-bit or 8-bit) is enabled but doesn't support CPU offloading by default

## Current Implementation Analysis

Looking at the code in `/home/kjifu/AI_finet/app/main_unified.py` (lines 319-334):

```python
# Large models automatically use 4-bit quantization
if any(size in request.model_name.lower() for size in ['22b', '32b', 'large', '7b']):
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        request.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
```

## Solutions

### Solution 1: Enable CPU Offloading for 8-bit Models (Recommended for 7B)

For 7B models, use 8-bit quantization with CPU offloading instead of 4-bit:

```python
from transformers import BitsAndBytesConfig

# Use 8-bit quantization with CPU offloading enabled
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

### Solution 2: Custom Device Map

Create a custom device map that keeps the entire model on GPU:

```python
# Check available GPU memory first
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 16:  # 16GB+ GPU
        # Keep entire model on GPU
        device_map = {"": 0}
    else:
        # Use balanced split
        device_map = "balanced"
```

### Solution 3: Use Pure 4-bit Quantization Without Split

For GPUs with at least 8GB memory, 7B models should fit with 4-bit quantization:

```python
# Ensure model fits on single GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Force single GPU usage
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": 0},  # Force all on GPU 0
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

### Solution 4: Model-Specific Configuration

Different model sizes require different approaches:

```python
def get_model_config(model_name, gpu_memory_gb):
    """Get optimal configuration based on model size and GPU memory"""
    
    # 7B models (14-16GB required for full precision)
    if "7b" in model_name.lower():
        if gpu_memory_gb >= 16:
            # Use 8-bit quantization for better quality
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": {"": 0}
            }
        elif gpu_memory_gb >= 8:
            # Use 4-bit quantization to fit
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ),
                "device_map": {"": 0}
            }
        else:
            # Enable CPU offloading
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_8bit_compute_dtype=torch.float16
                ),
                "device_map": "auto"
            }
```

## Recommended Fix for main_unified.py

Update the model loading logic to handle 7B models differently:

```python
# Around line 319 in main_unified.py
if any(size in request.model_name.lower() for size in ['22b', '32b', 'large']):
    # Keep existing 4-bit config for very large models
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    device_map = "auto"
elif "7b" in request.model_name.lower():
    # Special handling for 7B models
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    
    if gpu_memory >= 16:
        # 8-bit for better quality
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        device_map = {"": 0}
    else:
        # 4-bit with single GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        device_map = {"": 0}
else:
    # Small models don't need quantization
    quantization_config = None
    device_map = "auto" if torch.cuda.is_available() else None

# Load model with appropriate config
model_kwargs = {
    "torch_dtype": torch.float16,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "device_map": device_map
}

if quantization_config:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(
    request.model_name,
    **model_kwargs
)
```

## Environment-Specific Recommendations

### For RTX A5000 (24GB) x2:
- 7B models: Use 8-bit quantization on single GPU for best quality
- 13B models: Use 8-bit quantization with model parallelism
- 22B-32B models: Use 4-bit quantization with model parallelism

### For Single GPU Systems:
- 8GB GPU: 7B models with 4-bit quantization
- 16GB GPU: 7B models with 8-bit quantization
- 24GB GPU: 13B models with 8-bit quantization

## Testing the Fix

After implementing the fix, test with:

```python
# Test script
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

# Check GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.1f}GB")

# Load with 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Model loaded successfully!")
```

## Summary

The error occurs because 7B models with 4-bit quantization and `device_map="auto"` can trigger unwanted CPU/disk offloading. The solution is to:

1. Use 8-bit quantization instead of 4-bit for 7B models
2. Force the model to stay on a single GPU with `device_map={"": 0}`
3. Only use `device_map="auto"` with CPU offloading explicitly enabled
4. Implement model-size-specific loading strategies

This approach ensures stable fine-tuning for 7B models while maintaining efficiency for larger models.