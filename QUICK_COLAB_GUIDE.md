# Google Colabでの変換手順（メモリ最適化版）

## ⚠️ メモリ要件と制限

### GPU メモリ要件
- **DeepSeek-R1-Distill-Qwen-32B**: 
  - FP16: 約64GB必要
  - INT8: 約32GB必要
  - INT4: 約16-20GB必要
- **Google Colab A100 (40GB)**: INT4量子化で動作可能

### 推奨環境
- Google Colab Pro+ (A100 40GB)
- 実行時間: 約1-2時間
- Google Drive空き容量: 50GB以上

## 📋 事前準備（ローカルで実行）

### 1. LoRAアダプタの圧縮
```bash
cd /home/kjifu/MoE_RAG/outputs
tar -czf lora_20250830_223432.tar.gz lora_20250830_223432/
# ファイルサイズ: 約833MB
```

### 2. Google Driveへアップロード
1. [Google Drive](https://drive.google.com) を開く
2. 「マイドライブ」のルートに `lora_20250830_223432.tar.gz` をアップロード
3. アップロード完了を確認（約10分）

## 🚀 Google Colabでの実行

### Step 1: 新しいノートブック作成
1. [Google Colab](https://colab.research.google.com/) にアクセス
2. 「新しいノートブック」をクリック
3. **重要**: ランタイム → ランタイムのタイプを変更 → **A100 GPU** を選択

### Step 2: 変換スクリプトの実行

以下のセルを順番に実行してください：

#### セル1: GPU確認とセットアップ
```python
# GPU確認
!nvidia-smi

# 必要なパッケージをインストール（4bit量子化対応版）
!pip install -q transformers==4.44.0 peft accelerate bitsandbytes
!pip install -q sentencepiece protobuf
!pip install -q auto-gptq optimum  # 4bit量子化用

# llama.cppをインストール
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make clean && make LLAMA_CUDA=1 -j8
```

#### セル2: Google Driveマウントとファイル準備
```python
from google.colab import drive
import os
import torch
import gc

# Google Driveをマウント
drive.mount('/content/drive')

# LoRAアダプタを解凍
!tar -xzf /content/drive/MyDrive/lora_20250830_223432.tar.gz -C /content/
print("LoRA adapter extracted to /content/lora_20250830_223432")
```

#### セル3: モデルのマージ（4bit量子化版）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import gc

# メモリクリア
torch.cuda.empty_cache()
gc.collect()

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model with 4-bit quantization...")
print("Expected memory usage: ~16-20GB")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "39GB"}  # A100の最大メモリを指定
    )
    print("✅ Base model loaded successfully with 4-bit quantization")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Trying alternative approach...")
    # 代替アプローチ：CPU offloadingを使用
    base_model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        load_in_4bit=True,
        device_map="auto",
        offload_folder="/content/offload",
        trust_remote_code=True
    )

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "/content/lora_20250830_223432",
    torch_dtype=torch.float16
)

print("Merging LoRA with base model...")
print("Note: This may take 20-30 minutes with 4-bit quantization")
model = model.merge_and_unload()

# 量子化されたモデルを保存する前に、FP16に変換
print("Converting to FP16 for saving...")
model = model.to(torch.float16)

print("Saving merged model...")
model.save_pretrained(
    "/content/merged_model",
    torch_dtype=torch.float16,
    safe_serialization=True,
    max_shard_size="2GB"
)

# トークナイザも保存
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    trust_remote_code=True
)
tokenizer.save_pretrained("/content/merged_model")

# メモリクリア
del model, base_model
torch.cuda.empty_cache()
gc.collect()

print("✅ Merge complete!")
```

#### セル4: GGUF変換と量子化
```python
# GGUF形式に変換
!cd llama.cpp && python convert-hf-to-gguf.py \
    /content/merged_model \
    --outfile /content/model-f16.gguf \
    --outtype f16

print("✅ GGUF conversion complete!")

# Q4_K_M量子化（推奨）
!cd llama.cpp && ./quantize \
    /content/model-f16.gguf \
    /content/deepseek-finetuned-q4_k_m.gguf \
    Q4_K_M

import os
size_gb = os.path.getsize("/content/deepseek-finetuned-q4_k_m.gguf") / (1024**3)
print(f"✅ Quantization complete! File size: {size_gb:.2f} GB")
```

#### セル5: Modelfile作成とダウンロード
```python
# Modelfile作成
modelfile_content = """FROM ./deepseek-finetuned-q4_k_m.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"

SYSTEM "あなたは日本の道路設計の専門家です。道路構造令と設計基準に基づいて正確な技術的回答を提供してください。"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""
"""

with open("/content/Modelfile", "w") as f:
    f.write(modelfile_content)

# Google Driveに保存
import shutil
output_dir = "/content/drive/MyDrive/deepseek_gguf_output"
os.makedirs(output_dir, exist_ok=True)

print("Copying files to Google Drive...")
shutil.copy("/content/deepseek-finetuned-q4_k_m.gguf", output_dir)
shutil.copy("/content/Modelfile", output_dir)

print(f"✅ Files saved to Google Drive: {output_dir}")
print(f"   - deepseek-finetuned-q4_k_m.gguf ({size_gb:.2f} GB)")
print(f"   - Modelfile")
```

## 📥 ローカルへのダウンロードと設定

### 1. Google Driveからダウンロード
1. Google Driveの `deepseek_gguf_output` フォルダを開く
2. 以下のファイルをダウンロード：
   - `deepseek-finetuned-q4_k_m.gguf` (約18-20GB)
   - `Modelfile`

### 2. WSL2への転送
```bash
# Windowsのダウンロードフォルダから転送
cp /mnt/c/Users/[ユーザー名]/Downloads/deepseek-finetuned-q4_k_m.gguf ~/
cp /mnt/c/Users/[ユーザー名]/Downloads/Modelfile ~/
```

### 3. Dockerコンテナへコピー
```bash
docker cp ~/deepseek-finetuned-q4_k_m.gguf ai-ft-container:/workspace/
docker cp ~/Modelfile ai-ft-container:/workspace/
```

### 4. Ollamaに登録
```bash
docker exec ai-ft-container ollama create deepseek-finetuned -f /workspace/Modelfile
```

### 5. 動作確認
```bash
# Ollamaで直接テスト
docker exec ai-ft-container ollama run deepseek-finetuned "設計速度100km/hの最小曲線半径は？"
# 期待される回答: 460m

# RAGシステムでテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度100km/hの最小曲線半径は？"}'
```

### 6. RAG設定更新
```yaml
# src/rag/config/rag_config.yaml
llm:
  use_ollama_fallback: true
  ollama_model: deepseek-finetuned  # 変更
  ollama_host: http://localhost:11434
```

## ⏱️ 処理時間の目安
- GPU確認とセットアップ: 5分
- モデルのロード（4bit量子化）: 15-20分
- LoRAマージ: 20-30分
- GGUF変換: 10-15分
- 量子化: 10-15分
- **合計: 約60-85分**

## 💰 コスト
- Google Colab Pro+: $49.99/月（A100使用可能）
- 1回の変換で約1-2時間のGPU使用

## ⚠️ トラブルシューティング

### メモリ不足エラーが出た場合

#### オプション1: より積極的な量子化
```python
# 4bit量子化でもメモリ不足の場合
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # ダブル量子化を有効化
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # bfloat16の代わりにfloat16を使用
)
```

#### オプション2: バッチ処理でマージ
```python
# レイヤーごとにマージ（メモリ効率的だが時間がかかる）
import gc
import torch
from tqdm import tqdm

# LoRAアダプタをレイヤーごとにマージ
for name, module in tqdm(model.named_modules()):
    if hasattr(module, 'merge_and_unload'):
        module.merge_and_unload()
        torch.cuda.empty_cache()
        gc.collect()
```

#### オプション3: CPUオフロード使用
```python
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    load_in_4bit=True,
    device_map="auto",
    offload_folder="/content/offload",
    offload_state_dict=True,
    trust_remote_code=True
)
```

### セッションがクラッシュした場合
```python
# セッションをリスタート後、以下を実行
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import gc
import torch
torch.cuda.empty_cache()
gc.collect()
```

### llama.cpp のビルドエラー
```bash
# CUDAなしでビルド
!cd llama.cpp && make clean && make
```

### 変換が途中で止まった場合
- ランタイム → セッションの管理 → 終了
- 新しいセッションで最初からやり直し

## 📝 重要な注意事項
1. **必ずA100 GPUを選択**してください（無料版では不可）
2. **Google Driveの容量**を事前に確認（40GB以上必要）
3. **セッションタイムアウト**に注意（90分制限がある場合あり）
4. **ダウンロードは分割**して行うことも可能