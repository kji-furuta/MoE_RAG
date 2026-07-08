# CALM3-22Bモデルの量子化とOllama統合方案

## 1. 現状の課題

### メモリ要件
- **CALM3-22B (FP16)**: 約44GB必要
- **利用可能GPUメモリ**: 約36GB（不足）
- **現在の回避策**: Ollama llama3.2:3b使用（精度低下の懸念）

## 2. 量子化によるソリューション

### 2.1 量子化オプション

| 量子化レベル | ビット数 | メモリ使用量 | 品質 | 推奨度 |
|------------|---------|------------|------|--------|
| Q4_K_M | 4-bit | ~12GB | 良好なバランス | ★★★ |
| Q5_K_M | 5-bit | ~15GB | より高品質 | ★★☆ |
| Q8_0 | 8-bit | ~22GB | ほぼオリジナル | ★☆☆ |

### 2.2 推奨構成
- **Q4_K_M (4-bit量子化)** を推奨
  - メモリ使用量: 12GB（現在のGPUで十分動作可能）
  - 品質: 実用レベルを維持
  - 速度: 高速推論可能

## 3. 実装方法

### 3.1 方法1: BitsandBytesによる量子化（推奨）

```python
# 量子化設定
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4量子化
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 二重量子化
)

# モデルロード
model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/calm3-22b-chat",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 3.2 方法2: AWQ量子化（高精度）

```bash
# AWQインストール
pip install autoawq

# 量子化実行
python -m awq.entry --model_path cyberagent/calm3-22b-chat \
                    --w_bit 4 \
                    --q_group_size 128 \
                    --output_path ./outputs/calm3-22b-awq
```

### 3.3 方法3: GGUF形式への変換（Ollama標準）

```bash
# llama.cppを使用したGGUF変換
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 変換実行
python convert.py ../calm3-22b-chat \
    --outfile ../calm3-22b.gguf \
    --outtype q4_K_M
```

## 4. Ollama統合手順

### 4.1 Modelfile作成

```dockerfile
FROM ./calm3-22b-q4.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"

SYSTEM "あなたは道路設計の専門家です。技術基準に基づいて正確な回答を提供してください。"

TEMPLATE """{{ if .System }}System: {{ .System }}
{{ end }}User: {{ .Prompt }}
Assistant: """
```

### 4.2 Ollamaへの登録

```bash
# モデル作成
ollama create calm3-22b-q4 -f Modelfile

# テスト実行
ollama run calm3-22b-q4 "設計速度100km/hの最小曲線半径は？"
```

## 5. LoRAアダプタの統合

### 5.1 既存LoRAアダプタ
- `lora_20250830_223432`
- `lora_20250831_122140`

### 5.2 マージ手順

```python
from peft import PeftModel

# ベースモデルロード
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/calm3-22b-chat",
    torch_dtype=torch.bfloat16
)

# LoRAアダプタをマージ
model = PeftModel.from_pretrained(
    base_model,
    "./outputs/lora_20250831_122140"
)

# マージして保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./outputs/calm3-22b-merged")
```

## 6. RAGシステムへの統合

### 6.1 設定ファイル更新

```yaml
# src/rag/config/rag_config.yaml
llm:
  use_ollama_fallback: true
  ollama_model: calm3-22b-q4  # 量子化モデル
  ollama_host: http://localhost:11434
  temperature: 0.1
  max_new_tokens: 512
```

### 6.2 コード修正

```python
# src/rag/core/query_engine.py
class LLMGenerator:
    def _enable_ollama_fallback(self):
        # カスタムモデルを使用
        self.ollama_model = self.config.llm.get(
            'ollama_model', 
            'calm3-22b-q4'  # 量子化モデル
        )
```

## 7. 実装スクリプト

`scripts/quantize_model_for_ollama.py`を作成済み：

```bash
# 実行例
python scripts/quantize_model_for_ollama.py \
    --model-path cyberagent/calm3-22b-chat \
    --quantization Q4_K_M \
    --lora-path ./outputs/lora_20250831_122140
```

## 8. 期待される効果

### 8.1 メリット
- **メモリ効率**: 44GB → 12GB（73%削減）
- **モデル品質**: CALM3-22Bの知識を活用
- **専門性**: LoRAによる道路設計特化
- **速度**: 量子化による推論高速化

### 8.2 デメリット
- **精度低下**: 約2-5%の性能低下
- **初期設定**: 変換作業が必要
- **互換性**: GGUF形式への変換が必要な場合あり

## 9. 推奨実装順序

1. **Phase 1**: BitsandBytesでの4-bit量子化テスト
2. **Phase 2**: Ollama統合（GGUF変換）
3. **Phase 3**: LoRAアダプタのマージ
4. **Phase 4**: RAGシステムへの統合
5. **Phase 5**: 性能評価と調整

## 10. 代替案

### 10.1 vLLM使用（PagedAttention）
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="cyberagent/calm3-22b-chat",
    quantization="awq",  # AWQ量子化
    tensor_parallel_size=2,  # GPU並列化
)
```

### 10.2 Text Generation Inference (TGI)
```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id cyberagent/calm3-22b-chat \
    --quantize bitsandbytes-nf4
```

## まとめ

CALM3-22Bモデルの量子化によるOllama統合は技術的に実現可能です。特に4-bit量子化（Q4_K_M）を使用することで、現在のGPUメモリ（36GB）でも十分動作可能となり、llama3.2:3bよりも高品質な回答生成が期待できます。