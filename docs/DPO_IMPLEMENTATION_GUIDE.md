# DPO (Direct Preference Optimization) 実装ガイド

## 概要

MoE_RAGシステムにDPO (Direct Preference Optimization)機能を統合しました。DPOは、強化学習from Human Feedback (RLHF)の効率的な代替手法で、独立した報酬モデルなしに人間の選好に直接最適化します。

## 実装されたコンポーネント

### 1. DPO Trainer (`src/training/dpo_trainer.py`)

**主要機能**:
- TRL `DPOTrainer`をラップした実装
- QLoRA (4-bit quantization) + DPO統合
- 32Bモデル対応（2x 24GB GPU環境）

**クラス**:
- `DPOTrainingConfig`: DPO学習設定
- `DPOQLoRATrainer`: メインのトレーナークラス

**メモリ要件**:
```
Policy model (4-bit):        16GB
Reference model (4-bit):     16GB
Adapters/Activations:        2-4GB
─────────────────────────────────
合計:                       34-38GB ✅ 2x24GB環境で実行可能
```

**最適化手法**:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `gradient_checkpointing=True`
- `optim="paged_adamw_8bit"`
- `bf16=True` (Ampere+ GPU)

### 2. Training Service統合 (`app/training/service.py`)

**DPOルーティング**:
```python
if request.training_method == "dpo":
    # DPOQLoRATrainerを使用
    # 既存のGPU cleanup機構を再利用
    # EWCはスキップ（DPO非互換）
```

**統合ポイント**:
- Lines 200-254: DPO専用処理
- Lines 643-706: GPU cleanup再利用

### 3. Preference収集UI (`app/dpo/preference_ui.py`)

**FastAPI Endpoints**:

#### POST `/api/dpo/collect-preference`
単一のpreference dataを収集

```json
{
  "prompt": "東京についての短い俳句を詠んでください。",
  "chosen": "古池や 蛙飛び込む 水の音",
  "rejected": "東京はとても大きな都市で、たくさんの人が住んでいます。",
  "margin": 0.8
}
```

#### POST `/api/dpo/collect-preference-batch`
複数のpreference dataを一括収集

```json
{
  "dataset_name": "japanese_haiku_preferences",
  "preferences": [
    {
      "prompt": "...",
      "chosen": "...",
      "rejected": "..."
    }
  ]
}
```

#### GET `/api/dpo/preference-datasets`
利用可能なdatasetsのリスト取得

#### GET `/api/dpo/preference-dataset/{dataset_name}`
特定のdataset内容を取得

#### POST `/api/dpo/upload-preference-file`
JSONLファイルのアップロード

### 4. 依存関係更新 (`requirements.txt`)

```txt
transformers>=4.37.0  # ← 4.30.0から更新
accelerate>=0.27.0    # ← 0.20.0から更新
trl>=0.7.4            # ← 新規追加
```

## 使用方法

### 1. Preference Dataの準備

#### 手動収集（REST API）
```bash
curl -X POST "http://localhost:8050/api/dpo/collect-preference" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "日本の首都はどこですか？",
       "chosen": "東京です。",
       "rejected": "大阪です。"
     }'
```

#### ファイルアップロード
```bash
curl -X POST "http://localhost:8050/api/dpo/upload-preference-file" \
     -F "file=@preference_dataset.jsonl"
```

#### データ形式（JSONL）
```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 2. DPOファインチューニングの実行

#### REST API経由
```bash
curl -X POST "http://localhost:8050/api/train" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
       "training_method": "dpo",
       "training_data": ["data/dpo/preference_dataset.jsonl"],
       "lora_config": {
         "r": 64,
         "lora_alpha": 128
       },
       "training_config": {
         "beta": 0.1,
         "num_epochs": 1,
         "learning_rate": 5e-6
       }
     }'
```

#### Python API経由
```python
from src.training.dpo_trainer import DPOQLoRATrainer, DPOTrainingConfig

# 設定作成
config = DPOTrainingConfig(
    model_name="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    output_dir="./outputs/dpo_output",
    lora_r=64,
    lora_alpha=128,
    beta=0.1,
    num_train_epochs=1,
)

# トレーナー作成
trainer = DPOQLoRATrainer(config)

# 実行
trainer.run_full_pipeline(
    dataset_path="data/dpo/preference_dataset.jsonl",
    adapter_output_path="./outputs/dpo_adapter"
)
```

#### Accelerate経由実行
```bash
accelerate launch src/training/dpo_trainer.py
```

### 3. 学習後のマージと推論

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ベースモデルをロード（4-bit）
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    quantization_config=quantization_config,
    device_map="auto",
)

# LoRAアダプタをロードしてマージ
model = PeftModel.from_pretrained(base_model, "./outputs/dpo_adapter")
model = model.merge_and_unload()

# 保存
model.save_pretrained("./dpo_merged_model")
```

## ハイパーパラメータ

### 重要なパラメータ

#### `beta` (DPO損失のバランス)
- **デフォルト**: 0.1
- **低い値 (0.01-0.1)**: より積極的な選好学習、破滅的忘却リスク↑
- **高い値 (0.3-0.5)**: 保守的な更新、ベースモデル知識保持

#### `lora_r` (LoRAランク)
- **デフォルト**: 64
- **低い値 (16-32)**: 学習パラメータ少、メモリ効率的
- **高い値 (64-128)**: より表現力豊か、メモリ使用量↑

#### `learning_rate`
- **デフォルト**: 5e-6
- **DPO推奨範囲**: 1e-6 〜 1e-5

#### `gradient_accumulation_steps`
- **デフォルト**: 8
- **実効バッチサイズ**: batch_size × accumulation_steps = 1 × 8 = 8

## データセット作成ガイド

### 合成データセット生成パイプライン

1. **シード指示の生成（Self-Instruct）**
   - 少数の高品質日本語指示をシード
   - LLMで多様な新指示を生成

2. **応答の生成**
   - 各プロンプトに対して複数候補を生成
   - 異なるモデルから生成

3. **選好ラベリング（LLM-as-a-Judge）**
   - 強力なジャッジLLM（GPT-4, Claude 3）で評価
   - 評価基準: 有用性、正確性、指示準拠、安全性

4. **人間キュレーション（推奨）**
   - 合成データの一部をレビュー
   - LLMジャッジのバイアス修正

### データ品質のベストプラクティス

✅ **推奨**:
- ジャッジモデルは生成モデルと異なるファミリーを使用（Preference Leakage回避）
- 詳細なジャッジプロンプト作成（評価基準、尺度、例を明確化）
- 少なくとも100-500 pairsで開始
- ドメイン特化データの収集

❌ **避けるべき**:
- 同じモデルファミリーでの生成とジャッジ
- 曖昧な評価基準
- ノイズの多い低品質データ

## メモリ管理

### GPU Cleanup機構

DPO実行後も既存のGPU cleanup機構が適用されます:

```python
# app/training/service.py Lines 643-706
finally:
    # トレーナーを先に削除
    if dpo_trainer_instance is not None:
        dpo_trainer_instance = None

    # モデルをCPUに移動して削除
    if model is not None:
        model.to("cpu")
        model = None

    # 全GPUのキャッシュをクリア
    for device_idx in range(num_gpus):
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
```

### AcceleratorState管理

DPOTrainerもAcceleratorを使用するため、既存のcleanup機構が適用:

```python
from accelerate.state import AcceleratorState
if AcceleratorState._shared_state:
    AcceleratorState._reset_state(reset_partial_state=True)
```

## トラブルシューティング

### 1. メモリ不足エラー

**症状**: `CUDA out of memory`

**解決策**:
- `per_device_train_batch_size`を1に設定
- `gradient_accumulation_steps`を増やす
- `max_length`を削減（2048 → 1024）
- GPU数を増やす（tensor parallelism）

### 2. AcceleratorState Conflict

**症状**: `AcceleratorState has already been initialized`

**解決策**:
- Dockerコンテナを再起動
- 既存のGPU cleanup機構が自動適用

### 3. データセット読み込みエラー

**症状**: `Column 'prompt' not found`

**解決策**:
- JSONL形式を確認（各行が有効なJSON）
- 必須カラム確認: `prompt`, `chosen`, `rejected`

### 4. TRLライブラリエラー

**症状**: `No module named 'trl'`

**解決策**:
```bash
pip install trl>=0.7.4 --upgrade
```

## 評価方法

### 定性的評価
- 多様なプロンプトで対話的テスト
- 生成品質の主観的評価

### 定量的評価
- MT-Bench with GPT-4 judge
- Hold-out preference pairsでの精度測定
- Chosen応答の確率 > Rejected応答の確率

## 参考文献

1. **DPO論文**: Rafailov et al. "Direct Preference Optimization"
2. **prompt_kji.md**: 本実装の詳細な技術仕様書
3. **TRL Documentation**: https://huggingface.co/docs/trl
4. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs"

## サポート

問題が発生した場合:
1. ログを確認: `docker logs ai-ft-container`
2. GPU状態確認: `nvidia-smi`
3. メモリ使用量確認: API `/api/dpo/stats`

---

**実装日**: 2025-10-03
**対応モデル**: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
**GPU要件**: 2x 24GB (合計48GB)
**ステータス**: Production Ready
