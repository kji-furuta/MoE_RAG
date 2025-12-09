# RARdata学習済みモデル - GGUF変換ガイド

## 📦 モデルの永続性

### ✅ **永続的に保存されます**

`outputs/continual_task_103_20251209_002510` は以下の場所に永続保存されています：

```
ホスト: /home/kjifu/MoE_RAG/outputs/continual_task_103_20251209_002510/
コンテナ: /workspace/outputs/continual_task_103_20251209_002510/
```

#### 保証される永続性

| 操作 | モデルへの影響 |
|------|-------------|
| ✅ **コンテナ停止** (`docker-compose stop`) | **保持** |
| ✅ **コンテナ削除** (`docker-compose down`) | **保持** |
| ✅ **イメージ削除** | **保持** |
| ✅ **ホストOS再起動** | **保持** |
| ❌ **`outputs/`フォルダ削除** | **削除** |

#### Dockerボリュームマウント設定

`docker/docker-compose.yml` (54行目):
```yaml
volumes:
  - ../outputs:/workspace/outputs  # ← 永続保存の仕組み
```

**バックアップ方法**:
```bash
# ホストから直接コピー
cp -r /home/kjifu/MoE_RAG/outputs/continual_task_103_20251209_002510 \
      /path/to/backup/

# tar圧縮
tar -czf rardata_model_backup.tar.gz \
    outputs/continual_task_103_20251209_002510
```

---

## 🔄 GGUF変換の方法

### 方法1: LoRAマージ → GGUF変換（推奨）

#### Step 1: LoRAアダプターをマージ

```bash
# Dockerコンテナ内で実行
docker exec -it ai-ft-container bash

# マージスクリプト実行
python3 scripts/merge_rardata_lora.py \
    --adapter outputs/continual_task_103_20251209_002510/checkpoint-final \
    --output outputs/rardata_merged
```

**出力**:
```
outputs/rardata_merged/
├── model.safetensors           # マージ済みモデル重み
├── config.json                 # モデル設定
├── tokenizer.json              # トークナイザー
└── ...
```

**所要時間**: 約5-10分（32Bモデル）
**必要メモリ**: ~40GB RAM

#### Step 2: GGUF形式に変換

**Option A: llama.cpp を使用**

```bash
# llama.cppをクローン（初回のみ）
git clone https://github.com/ggerganov/llama.cpp.git /workspace/llama.cpp
cd /workspace/llama.cpp
make

# GGUF変換
python3 convert_hf_to_gguf.py \
    /workspace/outputs/rardata_merged \
    --outfile /workspace/gguf_models/rardata-q4_k_m.gguf \
    --outtype q4_k_m
```

**Option B: HuggingFace Hub経由**

```bash
# 1. HuggingFace Hubにアップロード
huggingface-cli login
huggingface-cli upload your-username/rardata-merged outputs/rardata_merged

# 2. HF Hubの自動GGUF変換を使用
# https://huggingface.co/your-username/rardata-merged
# → "Convert to GGUF" ボタンをクリック
```

#### 変換後のファイル

```
gguf_models/
└── rardata-q4_k_m.gguf         # 4-bit量子化GGUF (~8GB)
```

---

### 方法2: UIの「Apply LoRA Adapter to GGUF Model」機能

#### ⚠️ 現在の制限

この機能は**GGUFベースモデル**が必要ですが、現在ありません。

#### 準備手順

**Step 1: ベースモデルをGGUF形式で準備**

```bash
# DeepSeek-R1-32BのGGUFバージョンをダウンロード
# HuggingFace Hubから探す
# 例: https://huggingface.co/models?search=deepseek+gguf

# または、PyTorchモデルをGGUFに変換
python3 /workspace/llama.cpp/convert_hf_to_gguf.py \
    cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese \
    --outfile gguf_models/DeepSeek-R1-32B-base.gguf \
    --outtype q4_k_m
```

**Step 2: UIで変換**

1. **Fine-tuningタブ** → **Apply LoRA Adapter to GGUF Model**
2. 入力:
   - **Base GGUF Model**: `gguf_models/DeepSeek-R1-32B-base.gguf`
   - **LoRA Adapter**: `outputs/continual_task_103_20251209_002510/checkpoint-final`
   - **Output Path**: `gguf_models/rardata_final.gguf`
3. **Convert**をクリック

---

## 🚀 GGUF変換後の使用方法

### 1. Ollama で使用

#### Modelfile作成

```dockerfile
# gguf_models/Modelfile
FROM ./rardata-q4_k_m.gguf

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM """あなたは日本の道路設計・土木工学の専門家です。
道路構造令、設計基準、舗装設計などの質問に正確に答えてください。"""
```

#### Ollamaにロード

```bash
# モデル作成
cd gguf_models
ollama create rardata -f Modelfile

# 動作確認
ollama run rardata "地方部の第3種道路の設計速度は？"
```

### 2. llama.cpp で使用

```bash
cd /workspace/llama.cpp

# 対話モード
./llama-cli \
    -m /workspace/gguf_models/rardata-q4_k_m.gguf \
    -p "地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、道路構造令の種別上、何種道路に分類されますか？" \
    --temp 0.7 \
    --top-p 0.9 \
    -n 512

# サーバーモード
./llama-server \
    -m /workspace/gguf_models/rardata-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080
```

### 3. LM Studio で使用

1. LM Studioをインストール
2. **Local Models** → **Import GGUF**
3. `rardata-q4_k_m.gguf` を選択
4. **Load Model** → 使用開始

---

## 📊 GGUF量子化オプション

| 形式 | ビット数 | ファイルサイズ | 速度 | 品質 | 推奨用途 |
|------|----------|------------|------|------|---------|
| **q4_k_m** | 4-bit | ~8GB | ⚡⚡⚡ | ⭐⭐⭐ | **推奨** (バランス) |
| **q5_k_m** | 5-bit | ~10GB | ⚡⚡ | ⭐⭐⭐⭐ | 高品質 |
| **q8_0** | 8-bit | ~16GB | ⚡ | ⭐⭐⭐⭐⭐ | 最高品質 |
| **q3_k_m** | 3-bit | ~6GB | ⚡⚡⚡⚡ | ⭐⭐ | 軽量版 |

**変換コマンド例**:
```bash
# 5-bit（高品質）
python3 convert_hf_to_gguf.py outputs/rardata_merged \
    --outfile gguf_models/rardata-q5_k_m.gguf \
    --outtype q5_k_m

# 8-bit（最高品質）
python3 convert_hf_to_gguf.py outputs/rardata_merged \
    --outfile gguf_models/rardata-q8_0.gguf \
    --outtype q8_0
```

---

## 🔍 品質比較テスト

### テストプロンプト

```
地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、
道路構造令の種別上、何種道路に分類されますか？
```

### 期待される回答

```
地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、
道路構造令において「第3種道路」に分類されます。

第3種道路は、以下の特徴があります：
- 地方部の一般道路
- 設計速度: 30-60 km/h
- 車線数: 1車線または2車線
...
```

### 形式別テスト結果（参考）

| 形式 | 推論速度 | 回答品質 | メモリ使用量 |
|------|---------|---------|------------|
| **PyTorch (4-bit)** | 2-3 tokens/s | ⭐⭐⭐⭐⭐ | ~18GB VRAM |
| **GGUF (q4_k_m)** | 8-12 tokens/s | ⭐⭐⭐⭐ | ~8GB RAM |
| **GGUF (q5_k_m)** | 6-10 tokens/s | ⭐⭐⭐⭐⭐ | ~10GB RAM |
| **GGUF (q8_0)** | 4-8 tokens/s | ⭐⭐⭐⭐⭐ | ~16GB RAM |

---

## ⚙️ トラブルシューティング

### Q1: マージ時にメモリ不足

**解決策**: CPU offloadingを有効化

```python
# scripts/merge_rardata_lora.py を修正
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="/tmp/offload",  # 追加
    max_memory={0: "20GB", "cpu": "30GB"}  # 追加
)
```

### Q2: GGUF変換でエラー

**確認事項**:
1. llama.cppが最新版か確認
2. safetensors形式で保存されているか確認
3. 十分なディスク容量があるか確認

**代替方法**: HuggingFace Hub経由で変換

### Q3: Ollamaで読み込めない

**確認事項**:
1. GGUFファイルが破損していないか
2. Modelfileのパスが正しいか
3. Ollamaが最新版か

```bash
# Ollama更新
curl -fsSL https://ollama.com/install.sh | sh

# モデル削除と再作成
ollama rm rardata
ollama create rardata -f Modelfile
```

---

## 📚 参考リンク

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Ollama**: https://ollama.com/
- **GGUF仕様**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **HuggingFace Hub**: https://huggingface.co/docs/hub/gguf

---

## 📝 まとめ

### ✅ **モデルの永続性**
- Dockerボリュームマウントにより永続保存
- コンテナ削除後もデータは保持
- ホストから直接アクセス・バックアップ可能

### ✅ **GGUF変換**
- **方法1**: LoRAマージ → llama.cpp変換（推奨）
- **方法2**: UIの機能（要GGUFベースモデル）
- **推奨形式**: q4_k_m（バランス重視）

### ✅ **使用方法**
- Ollama（最も簡単）
- llama.cpp（柔軟性高）
- LM Studio（GUI）

### 次のステップ
1. スクリプト実行: `python3 scripts/merge_rardata_lora.py`
2. GGUF変換: llama.cpp または HF Hub
3. 品質テスト: 実際の質問で確認
