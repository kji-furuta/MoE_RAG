# GPT-NeoX-20B 動的LoRA適用 完全ガイド

## 概要
GPT-NeoX-20Bモデルに対して、LoRAアダプターを動的に適用してRAGシステムで使用する方法を説明します。

## 1. LoRAアダプターの作成

### 1.1 データセット準備
```bash
# 道路設計データセットを準備
cat > /workspace/data/road_design_dataset.json << 'EOF'
[
  {
    "text": "質問: 設計速度80km/hの道路の最小曲線半径は？\n回答: 設計速度80km/hの道路の最小曲線半径は280mです。"
  },
  {
    "text": "質問: 縦断勾配の最大値について教えてください。\n回答: 設計速度80km/hの場合は5%、60km/hの場合は7%が標準値です。"
  },
  {
    "text": "質問: 横断勾配の標準値は？\n回答: 横断勾配の標準値は1.5%から2.0%です。"
  }
]
EOF
```

### 1.2 LoRAファインチューニング実行

#### 方法1: Webインターフェース経由
```
1. http://localhost:8050/finetune にアクセス
2. 以下を設定:
   - Model: EleutherAI/gpt-neox-20b
   - Training Method: LoRA
   - Dataset: road_design_dataset.json
   - LoRA Rank: 16
   - Learning Rate: 3e-4
   - Epochs: 3
3. "Start Training"をクリック
```

#### 方法2: CLIスクリプト
```bash
# GPT-NeoX専用LoRAトレーニングスクリプトを作成
cat > /workspace/scripts/train_gpt_neox_lora.py << 'EOF'
#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json
from datetime import datetime

# モデルとトークナイザーをロード
model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 8bit量子化でモデルをロード（メモリ節約）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# k-bit training用に準備
model = prepare_model_for_kbit_training(model)

# GPT-NeoX用のLoRA設定
lora_config = LoraConfig(
    r=16,  # LoRAのランク
    lora_alpha=32,
    target_modules=[
        "attention.query_key_value",  # GPT-NeoX特有のQKV統合層
        "attention.dense",
        "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRAを適用
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# データセットをロード
with open("/workspace/data/road_design_dataset.json", 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# トレーニング設定
output_dir = f"/workspace/outputs/gpt_neox_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    learning_rate=3e-4,
    fp16=True
)

# トレーナー
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# トレーニング実行
trainer.train()

# モデルを保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ LoRA adapter saved to: {output_dir}")
EOF

# 実行
docker exec ai-ft-container python /workspace/scripts/train_gpt_neox_lora.py
```

## 2. LoRAアダプターのGGUF変換と動的適用

### 2.1 自動変換・適用スクリプト
```bash
# 動的適用スクリプトを実行
docker exec ai-ft-container python /workspace/scripts/apply_lora_gpt_neox_dynamic.py \
    --base-gguf /workspace/models/gpt-neox-20b.Q4_K_M.gguf \
    --lora-adapter /workspace/outputs/gpt_neox_lora_latest \
    --output-dir /workspace/outputs/gpt_neox_dynamic \
    --ollama-create gpt-neox-20b-road-design
```

### 2.2 生成される構成
```
/workspace/outputs/gpt_neox_dynamic/
├── lora_adapter.gguf        # GGUF形式のLoRAアダプター
├── run_with_lora.sh         # CLIモード実行スクリプト
├── start_server.sh          # サーバーモード起動スクリプト
├── gpt_neox_lora_api.py     # Python APIラッパー
└── Modelfile                # Ollama用設定ファイル
```

## 3. RAGシステムの設定

### 3.1 設定ファイルの更新
```yaml
# /workspace/src/rag/config/rag_config.yaml を編集

llm:
  # 動的LoRA適用を有効化
  use_dynamic_lora: true
  provider: dynamic_lora
  base_model: gpt-neox-20b-dynamic-lora
  
  # ベースモデルとLoRAの設定
  dynamic_lora:
    base_gguf_path: /workspace/models/gpt-neox-20b.Q4_K_M.gguf
    lora_adapter_path: /workspace/outputs/gpt_neox_lora_latest
    server_port: 8081  # llama.cppサーバーポート
    
  # パラメータ設定
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 1024
  repetition_penalty: 1.1
```

### 3.2 RAGシステムの再起動
```bash
# サーバー再起動
docker exec ai-ft-container bash -c "
  pkill -f uvicorn
  sleep 2
  python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
"
```

## 4. 使用方法

### 4.1 WebUI経由
```
1. http://localhost:8050/rag にアクセス
2. 質問を入力（例: "設計速度80km/hの道路の最小曲線半径は？"）
3. "Search"をクリック
4. GPT-NeoX-20B + LoRAによる回答を確認
```

### 4.2 API経由
```bash
# ハイブリッド検索 + 動的LoRA回答生成
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "道路の縦断勾配の最大値について",
       "top_k": 5,
       "use_dynamic_lora": true
     }'
```

### 4.3 直接実行（llama.cpp経由）
```bash
# CLIモード
/workspace/outputs/gpt_neox_dynamic/run_with_lora.sh \
  "設計速度100km/hの道路の最小曲線半径を教えてください"

# サーバーモード
/workspace/outputs/gpt_neox_dynamic/start_server.sh
# 別ターミナルで
curl -X POST http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "道路設計について", "n_predict": 200}'
```

### 4.4 Ollama経由
```bash
# モデル確認
ollama list | grep gpt-neox

# 推論実行
ollama run gpt-neox-20b-road-design "横断勾配の標準値は？"
```

## 5. ハイブリッド検索の詳細設定

### 5.1 検索重みの調整
```yaml
# rag_config.yaml
retrieval:
  hybrid_search:
    enabled: true
    vector_weight: 0.7    # ベクトル検索の重み
    keyword_weight: 0.3   # キーワード検索の重み
  reranking:
    enabled: true
    model: gpt-neox-20b-dynamic-lora
  top_k: 10
  rerank_top_k: 5
```

### 5.2 専門用語辞書の追加
```python
# /workspace/src/rag/retrieval/hybrid_search.py に追加
ROAD_DESIGN_TERMS = [
    "最小曲線半径", "縦断勾配", "横断勾配", "視距",
    "設計速度", "道路構造令", "路肩", "車道幅員"
]

# キーワード検索時にブースト
def boost_technical_terms(query: str) -> str:
    for term in ROAD_DESIGN_TERMS:
        if term in query:
            query = f"{query} {term}^2"  # ブースト係数2
    return query
```

## 6. トラブルシューティング

### 問題: LoRAアダプターが認識されない
```bash
# adapter_config.jsonの確認
cat /workspace/outputs/gpt_neox_lora_latest/adapter_config.json | grep base_model

# 期待される出力
"base_model_name_or_path": "EleutherAI/gpt-neox-20b"
```

### 問題: メモリ不足エラー
```bash
# 設定を調整
export CUDA_VISIBLE_DEVICES=0,1  # 使用するGPUを指定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 量子化レベルを上げる
--quantization Q4_0  # より高圧縮
```

### 問題: llama.cppサーバーが起動しない
```bash
# ポート確認
lsof -i :8081

# 手動でサーバー起動
/workspace/llama.cpp/server \
  -m /workspace/models/gpt-neox-20b.Q4_K_M.gguf \
  --lora /workspace/outputs/gpt_neox_dynamic/lora_adapter.gguf \
  --port 8081 \
  --host 0.0.0.0
```

## 7. パフォーマンス最適化

### 7.1 推論速度向上
```yaml
# GPU層数を増やす
dynamic_lora:
  n_gpu_layers: 40  # GPUメモリに応じて調整
  n_batch: 512      # バッチサイズ
  n_threads: 16     # CPUスレッド数
```

### 7.2 メモリ使用量削減
```yaml
# コンテキストサイズを制限
dynamic_lora:
  ctx_size: 2048    # デフォルト4096から削減
  use_mmap: true    # メモリマップファイル使用
  use_mlock: false  # メモリロック無効化
```

## 8. 検証とテスト

### 8.1 LoRA適用効果の検証
```python
# テストスクリプト
import requests

# ベースモデルのみ
base_response = requests.post(
    "http://localhost:8050/rag/query",
    json={"query": "最小曲線半径は？", "use_dynamic_lora": false}
).json()

# LoRA適用
lora_response = requests.post(
    "http://localhost:8050/rag/query",
    json={"query": "最小曲線半径は？", "use_dynamic_lora": true}
).json()

# 比較
print("Base:", base_response["answer"][:100])
print("LoRA:", lora_response["answer"][:100])
```

### 8.2 ベンチマーク
```bash
# 応答時間測定
time curl -X POST http://localhost:8050/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "use_dynamic_lora": true}'

# スループット測定
ab -n 100 -c 10 -p query.json -T application/json \
  http://localhost:8050/rag/query
```

## まとめ

GPT-NeoX-20Bの動的LoRA適用により：
1. **メモリ効率**: ベースモデルとLoRAを別管理（ベース13GB + LoRA数MB）
2. **柔軟性**: 複数のLoRAアダプターを切り替え可能
3. **精度向上**: 道路設計専門知識を注入
4. **互換性**: 既存のRAGシステムとシームレスに統合

推奨構成：
- GPU: 24GB以上のVRAM
- 量子化: Q4_K_M（品質と速度のバランス）
- LoRAランク: 16（精度とサイズのバランス）