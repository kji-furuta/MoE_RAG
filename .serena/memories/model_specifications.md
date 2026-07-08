# モデル仕様と設定詳細

## サポート済みモデル一覧

### 日本語特化モデル

#### Qwen2.5シリーズ
- **Qwen2.5-14B-Instruct**
  - パス: `Qwen/Qwen2.5-14B-Instruct`
  - VRAM要件: 28GB (float16), 14GB (8bit), 7GB (4bit)
  - 推奨用途: 汎用対話、コード生成、翻訳
  - 量子化: 8bit推奨

- **Qwen2.5-32B-Instruct**
  - パス: `Qwen/Qwen2.5-32B-Instruct`
  - VRAM要件: 64GB (float16), 32GB (8bit), 16GB (4bit)
  - 推奨用途: 高度な推論、長文生成
  - 量子化: 4bit必須

#### CyberAgentモデル
- **CALM3-22B-Chat**
  - パス: `cyberagent/calm3-22b-chat`
  - VRAM要件: 44GB (float16), 22GB (8bit), 11GB (4bit)
  - 特徴: 日本語に最適化、高品質な対話生成
  - 量子化: 4bit推奨

- **DeepSeek-R1-Distill-Qwen-32B-Japanese**
  - パス: `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese`
  - VRAM要件: 64GB (float16), 32GB (8bit)
  - 特徴: DeepSeekの推論能力を日本語に蒸留
  - 量子化: 8bit推奨

#### その他の日本語モデル
- **ELYZA-japanese-Llama-2-7b-instruct**
  - パス: `elyza/ELYZA-japanese-Llama-2-7b-instruct`
  - VRAM要件: 14GB (float16), 7GB (8bit)
  - 特徴: Llama2ベース、日本語特化

- **Rinna japanese-gpt2-small**
  - パス: `rinna/japanese-gpt2-small`
  - VRAM要件: 1GB
  - 用途: テスト、軽量処理

- **StabilityAI japanese-stablelm-3b-4e1t-instruct**
  - パス: `stabilityai/japanese-stablelm-3b-4e1t-instruct`
  - VRAM要件: 6GB (float16), 3GB (8bit)
  - 特徴: 効率的な小規模モデル

### 多言語モデル

#### Meta Llama 3.1
- **Llama-3.1-70B-Instruct**
  - パス: `meta-llama/Meta-Llama-3.1-70B-Instruct`
  - VRAM要件: 140GB (float16), 35GB (4bit)
  - 量子化: 4bit必須
  - CPUオフロード: 必須

#### Microsoft Phi
- **Phi-3.5-MoE-instruct**
  - パス: `microsoft/Phi-3.5-MoE-instruct`
  - VRAM要件: 32GB (float16), 16GB (8bit)
  - 特徴: MoE アーキテクチャ、効率的

## 量子化設定

### 自動量子化ルール
```python
# app/memory_optimized_loader.py の設定
QUANTIZATION_RULES = {
    "32B+": {
        "bits": 4,
        "method": "bnb",  # bitsandbytes
        "compute_dtype": "float16"
    },
    "20B-32B": {
        "bits": 8,
        "method": "bnb",
        "compute_dtype": "bfloat16"
    },
    "7B-20B": {
        "bits": 8,
        "method": "auto-gptq",
        "compute_dtype": "float16"
    },
    "<7B": {
        "bits": None,  # 量子化なし
        "method": None,
        "compute_dtype": "float32"
    }
}
```

### 手動量子化設定
```python
from transformers import BitsAndBytesConfig

# 4bit量子化
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 8bit量子化
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=False
)
```

## LoRA設定パラメータ

### 標準LoRA設定
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # ランク（4-64）
    lora_alpha=32,          # スケーリング係数
    target_modules=[        # 対象モジュール
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,       # ドロップアウト率
    bias="none",            # バイアス設定
    task_type="CAUSAL_LM"   # タスクタイプ
)
```

### モデル別推奨設定

#### 大規模モデル（20B+）
```python
{
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 8
}
```

#### 中規模モデル（7B-20B）
```python
{
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 4
}
```

#### 小規模モデル（<7B）
```python
{
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 2
}
```

## 訓練設定

### 推奨バッチサイズ
| モデルサイズ | float16 | 8bit | 4bit |
|------------|---------|------|------|
| 70B | 1 | 2 | 4 |
| 32B | 2 | 4 | 8 |
| 22B | 4 | 8 | 16 |
| 14B | 8 | 16 | 32 |
| 7B | 16 | 32 | 64 |
| 3B | 32 | 64 | 128 |

### 学習率設定
```python
LEARNING_RATE_CONFIG = {
    "full_finetuning": {
        "base_lr": 5e-5,
        "scheduler": "cosine",
        "warmup_ratio": 0.1
    },
    "lora_finetuning": {
        "base_lr": 2e-4,
        "scheduler": "linear",
        "warmup_steps": 500
    },
    "qlora_finetuning": {
        "base_lr": 1e-4,
        "scheduler": "constant_with_warmup",
        "warmup_steps": 100
    }
}
```

## CPUオフロード設定

### 自動オフロード
```python
from accelerate import cpu_offload_with_hook

# モデルサイズに基づく自動設定
def get_offload_config(model_size_gb):
    if model_size_gb > 40:
        return {
            "offload_layers": True,
            "offload_optimizer": True,
            "offload_gradients": True
        }
    elif model_size_gb > 20:
        return {
            "offload_layers": False,
            "offload_optimizer": True,
            "offload_gradients": True
        }
    else:
        return {
            "offload_layers": False,
            "offload_optimizer": False,
            "offload_gradients": False
        }
```

## マルチGPU設定

### DataParallel設定
```python
# 単純な並列化
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### DistributedDataParallel設定
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 分散訓練の初期化
dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

### DeepSpeed設定
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e6
  }
}
```

## Ollama統合設定

### モデル変換
```bash
# Ollama形式への変換
python scripts/convert_to_ollama.py \
    --model_path outputs/my_model \
    --output_name my-custom-model

# Ollamaへの登録
ollama create my-model -f Modelfile
ollama push my-model
```

### Modelfileテンプレート
```dockerfile
FROM outputs/gguf/model.gguf

TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "[/INST]"
PARAMETER stop "</s>"

SYSTEM """あなたは日本語の質問に答える親切なアシスタントです。"""
```