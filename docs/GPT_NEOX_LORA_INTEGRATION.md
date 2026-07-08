# GPT-NeoX-20B LoRA統合ガイド

## 概要
GPT-NeoX-20BモデルのLoRAファインチューニングを量子化モデルで使用するための2つのワークフローを提供します。

## 重要な技術的制約

### なぜ直接マージが不可能なのか
量子化されたGGUFモデル（Q4_K_M等）にLoRAを直接マージすることは**技術的に不可能**です：
- 量子化により重みが低精度（4-bit等）に圧縮されている
- LoRAの差分適用には高精度の演算が必要
- 情報の損失により正確な重み更新ができない

### 解決策: 2つのワークフロー
1. **ワークフローA**: 事前量子化マージ（推奨）
   - 高精度モデルでLoRAをマージしてから量子化
   - 最高の忠実度を維持
   
2. **ワークフローB**: 動的ランタイム適用
   - GGUFモデルとLoRAを別々に保持
   - 推論時に動的に適用

## 実装内容

### 1. 自動判定スクリプト (`scripts/apply_lora_to_gguf_auto.py`)
- LoRAアダプタのconfig.jsonからモデルタイプを自動検出
- GPT-NeoX、DeepSeek、Qwen、Llamaなどを識別
- モデルタイプに応じて適切な変換スクリプトを選択

### 2. GPT-NeoX専用変換 (`scripts/apply_lora_gpt_neox.py`)
- レイヤー名マッピング:
  - `attention.query_key_value` → `blk.{layer}.attn_q/k/v.weight` (3分割)
  - `attention.dense` → `blk.{layer}.attn_output.weight`
  - `mlp.dense_h_to_4h` → `blk.{layer}.ffn_up.weight`
  - `mlp.dense_4h_to_h` → `blk.{layer}.ffn_down.weight`

### 3. UI統合 (`app/main_unified.py`)
- 自動判定スクリプトを優先的に使用
- 既存のUIワークフローとシームレスに統合

## 使用方法

### ワークフローB: 動的ランタイム適用（推奨）

#### 1. LoRAアダプタをGGUF形式に変換して動的適用

```bash
# 動的適用スクリプトを実行
python scripts/apply_lora_gpt_neox_dynamic.py \
    --base-gguf models/gpt-neox-20b.Q4_K_M.gguf \
    --lora-adapter outputs/lora_20250906_104924 \
    --output-dir outputs/workflow_b

# Ollamaモデルとして登録（オプション）
python scripts/apply_lora_gpt_neox_dynamic.py \
    --base-gguf models/gpt-neox-20b.Q4_K_M.gguf \
    --lora-adapter outputs/lora_20250906_104924 \
    --output-dir outputs/workflow_b \
    --ollama-create gpt-neox-lora-dynamic
```

#### 2. 生成されたツールの使用

```bash
# CLIモード
./outputs/workflow_b/run_with_lora.sh "道路設計の最小曲線半径を説明"

# サーバーモード
./outputs/workflow_b/start_server.sh
# 別ターミナルで:
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "道路設計について", "max_tokens": 100}'

# Ollama経由
ollama run gpt-neox-lora-dynamic "質問内容"
```

### ワークフローA: 事前量子化マージ（高精度が必要な場合）

**注意**: このワークフローには元のHuggingFace形式の高精度モデル（20GB以上）が必要です。

```bash
# 高精度モデルでマージしてから量子化
python scripts/apply_lora_gpt_neox_full.py \
    --base-model EleutherAI/gpt-neox-20b \
    --lora-adapter outputs/lora_20250906_104924 \
    --output-dir outputs/workflow_a \
    --quantization Q4_K_M
```

## テスト

```bash
# テストスクリプトを実行
python scripts/test_gpt_neox_lora.py

# Ollamaでテスト
ollama run gpt-neox-20b-finetuned "道路設計における最小曲線半径について説明してください"
```

## サポートされるモデル

自動判定により以下のモデルタイプをサポート:

| モデルタイプ | 例 | 変換方式 |
|------------|---|---------|
| GPT-NeoX | EleutherAI/gpt-neox-20b | 専用マッピング（QKV分割） |
| DeepSeek | cyberagent/DeepSeek-R1-Distill-Qwen-32B | 標準処理 |
| Qwen | Qwen/Qwen-* | 標準処理 |
| Llama | meta-llama/Llama-* | 標準処理 |

## 技術的詳細

### なぜ2つのワークフローが必要なのか

#### 直接マージの問題点
```
量子化モデル (Q4_K_M): W_quantized (4-bit)
LoRA差分: ΔW = B × A (高精度)
マージ試行: W_new = W_quantized + ΔW ❌ (精度不一致)
```

#### ワークフローA: 事前マージ
```
1. 高精度モデル: W_original (FP16/32)
2. LoRA適用: W_new = W_original + B × A ✅
3. 量子化: W_final = quantize(W_new, Q4_K_M) ✅
```

#### ワークフローB: 動的適用
```
1. 実行時: W_dequant = dequantize(W_quantized)
2. LoRA適用: W_temp = W_dequant + B × A
3. 計算実行: output = compute(W_temp, input)
```

## トラブルシューティング

### 問題: llama.cppがLoRAを認識しない
- LoRAアダプタがGGUF形式に変換されているか確認
- `convert-lora-to-gguf.py`スクリプトの実行を確認

### 問題: メモリ不足エラー
- ワークフローBを使用（メモリ効率的）
- `--n-gpu-layers`パラメータを調整

### 問題: Ollama ADAPTERコマンドが動作しない
- Ollamaのバージョンを確認（v0.1.24以降が必要）
- ModelfileのFROMとADAPTERパスが正しいか確認

## 更新履歴

- 2025-09-08: ワークフローA/B実装 - 技術的制約を正しく理解した実装に更新
- 2025-09-07: GPT-NeoX-20B専用変換機能を実装
- 2025-09-07: 自動モデルタイプ判定機能を追加
- 2025-09-07: UI統合完了