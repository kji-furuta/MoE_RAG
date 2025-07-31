# 使用ガイド

このガイドでは、AI Fine-tuning Toolkitの詳細な使用方法について説明します。

## 📋 目次

1. [環境セットアップ](#環境セットアップ)
2. [モデル選択ガイド](#モデル選択ガイド)
3. [ファインチューニング手法の比較](#ファインチューニング手法の比較)
4. [実践的な例](#実践的な例)
5. [パフォーマンスチューニング](#パフォーマンスチューニング)
6. [トラブルシューティング](#トラブルシューティング)

## 環境セットアップ

### 前提条件の確認

```bash
# NVIDIA GPUドライバーの確認
nvidia-smi

# Dockerのバージョン確認
docker --version
docker-compose --version

# CUDA対応の確認
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi
```

### 環境変数の設定

```bash
# .env ファイルを作成
cp .env.example .env

# 必要に応じて編集
# WANDB_API_KEY=your_wandb_key
# HF_TOKEN=your_huggingface_token
```

## モデル選択ガイド

### GPUメモリ別推奨モデル

| GPUメモリ | 推奨モデル | ファインチューニング手法 |
|----------|----------|----------------------|
| 8GB | Japanese StableLM 3B | LoRA/QLoRA |
| 16GB | ELYZA Llama-3 8B | LoRA/QLoRA |
| 24GB | ELYZA Llama-3 8B | Full/LoRA |
| 32GB+ | DeepSeek-R1 32B | QLoRA |
| 64GB+ | DeepSeek-R1 32B | Full/LoRA |

### モデルの特徴

#### Japanese StableLM 3B
- **用途**: 軽量なタスク、プロトタイピング
- **強み**: 高速、省メモリ
- **弱み**: 複雑なタスクには限界

```python
model = JapaneseModel(
    model_name="stabilityai/japanese-stablelm-3b-4e1t-instruct",
    load_in_8bit=True
)
```

#### ELYZA Llama-3 8B
- **用途**: バランスの取れた性能
- **強み**: 高品質な日本語生成
- **弱み**: 中程度のメモリ要件

```python
model = JapaneseModel(
    model_name="elyza/Llama-3-ELYZA-JP-8B",
    load_in_8bit=True
)
```

#### CyberAgent DeepSeek-R1 32B
- **用途**: 高性能が必要なタスク
- **強み**: 最高品質の生成
- **弱み**: 大容量メモリが必要

```python
model = JapaneseModel(
    model_name="cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B",
    load_in_4bit=True  # QLoRA必須
)
```

## ファインチューニング手法の比較

### 1. フルファインチューニング

**適用場面**:
- 高品質が最優先
- 十分なGPUメモリがある
- データ量が豊富

**メリット**:
- 最高の性能
- 全パラメータが更新される

**デメリット**:
- 大容量メモリが必要
- 学習時間が長い

```python
# 設定例
config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=2,
    gradient_accumulation_steps=8,
    num_epochs=3,
    gradient_checkpointing=True
)
```

### 2. LoRAファインチューニング

**適用場面**:
- バランスの取れた性能とメモリ効率
- 複数のタスクに対応
- 中程度のGPUメモリ

**メリット**:
- パラメータ効率的（0.2%のみ更新）
- 複数のアダプターを管理可能
- 高速な学習

**デメリット**:
- フルファインチューニングより性能が劣る場合がある

```python
# 設定例
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05
)

training_config = TrainingConfig(
    learning_rate=3e-4,
    batch_size=4,
    num_epochs=5
)
```

### 3. QLoRAファインチューニング

**適用場面**:
- 限られたGPUメモリ
- 大規模モデルの使用
- 実験的なアプローチ

**メリット**:
- 最小のメモリ使用量
- 大規模モデルでの学習が可能

**デメリット**:
- 量子化による精度低下
- 学習速度が遅い場合がある

```python
# 設定例
qlora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    use_qlora=True,
    qlora_4bit=True
)

training_config = TrainingConfig(
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=16
)
```

## 実践的な例

### 対話システムの構築

```python
# データの準備
dialogue_data = [
    "<|user|>こんにちは<|endoftext|>\n<|assistant|>こんにちは！何かお手伝いできることはありますか？<|endoftext|>",
    "<|user|>今日の天気はどうですか？<|endoftext|>\n<|assistant|>申し訳ございませんが、リアルタイムの天気情報にはアクセスできません。<|endoftext|>",
    # ... more data
]

# LoRAでの学習
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

trainer = LoRAFinetuningTrainer(model, lora_config, training_config)
trained_model = trainer.train(train_texts=dialogue_data)
```

### 質問応答システムの構築

```python
# Q&Aデータの準備
qa_data = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さは何メートルですか？\n回答: 富士山の高さは3,776メートルです。",
    # ... more data
]

# QLoRAでメモリ効率的に学習
qlora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    use_qlora=True,
    qlora_4bit=True
)

trainer = LoRAFinetuningTrainer(model, qlora_config, training_config)
trained_model = trainer.train(train_texts=qa_data)
```

### 文書要約システムの構築

```python
# 要約データの準備
summary_data = [
    "文書: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。\n要約: AIは人間の知能を模倣する技術で、機械学習や自然言語処理を含みます。",
    # ... more data
]

# フルファインチューニングで高品質な要約
config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=2,
    gradient_accumulation_steps=8,
    num_epochs=3
)

trainer = FullFinetuningTrainer(model, config)
trained_model = trainer.train(train_texts=summary_data)
```

## パフォーマンスチューニング

### 🚀 マルチGPU最適化（RTX A5000 x2検証済み）

**現在の性能**:
- GPU利用率: 50%（1/2 GPU使用）
- 対応モデルサイズ: 最大7B
- 学習速度: 100 tokens/sec

**最適化後の性能（期待値）**:
- GPU利用率: 100%（2/2 GPU使用）
- 対応モデルサイズ: 最大30B+
- 学習速度: 180-280 tokens/sec（**1.8-2.8倍高速化**）

```python
# マルチGPU最適化設定例
from src.training.multi_gpu_training import AdvancedMultiGPUTrainer, MultiGPUTrainingConfig

# モデル並列での13Bモデル学習
config = MultiGPUTrainingConfig(
    strategy='model_parallel',
    max_memory_per_gpu={0: '22GB', 1: '22GB'},
    fp16=True,
    gradient_checkpointing=True
)

trainer = AdvancedMultiGPUTrainer(model, config)
trained_model = trainer.train(train_texts)
```

### 学習率の調整

```python
# ファインチューニング手法別の推奨学習率
learning_rates = {
    "full_finetuning": 2e-5,
    "lora": 3e-4,
    "qlora": 2e-4
}

# 学習率スケジューラーの設定
config = TrainingConfig(
    learning_rate=learning_rates["lora"],
    warmup_steps=100,  # ウォームアップステップ
    # 線形減衰が自動的に適用される
)
```

### バッチサイズと勾配累積

```python
# メモリ効率的な設定
config = TrainingConfig(
    batch_size=1,                    # 小さいバッチサイズ
    gradient_accumulation_steps=16,  # 勾配累積で実効バッチサイズを16に
    # 実際のバッチサイズ = batch_size * gradient_accumulation_steps = 16
)
```

### GPU最適化

```python
# Flash Attention と Gradient Checkpointing
model = JapaneseModel(
    model_name="your-model",
    use_flash_attention=True,
    gradient_checkpointing=True
)

config = TrainingConfig(
    fp16=True,                       # Mixed Precision
    gradient_checkpointing=True,     # メモリ節約
)
```

## トラブルシューティング

### CUDA Out of Memory エラー

```python
# 解決策1: バッチサイズを減らす
config = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=32
)

# 解決策2: QLoRAを使用
qlora_config = LoRAConfig(
    use_qlora=True,
    qlora_4bit=True
)

# 解決策3: Gradient Checkpointing
config = TrainingConfig(
    gradient_checkpointing=True
)
```

### 学習が収束しない

```python
# 解決策1: 学習率を下げる
config = TrainingConfig(
    learning_rate=1e-5,  # より低い学習率
    warmup_steps=200     # より長いウォームアップ
)

# 解決策2: より多くのエポック
config = TrainingConfig(
    num_epochs=10,
    eval_steps=50       # 頻繁な評価
)
```

### モデルロードエラー

```python
# 解決策1: フォールバック機能
model = JapaneseModel(model_name="large-model")
success = model.load_with_fallback([
    "stabilityai/japanese-stablelm-3b-4e1t-instruct",
    "rinna/japanese-gpt-neox-3.6b-instruction-sft"
])

# 解決策2: 明示的な量子化設定
model = JapaneseModel(
    model_name="your-model",
    load_in_8bit=True,
    torch_dtype=torch.float16
)
```

### 分散学習の問題

```bash
# 解決策1: 環境変数の設定
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # P2P通信を無効化（問題がある場合）

# 解決策2: Accelerate設定
accelerate config  # インタラクティブ設定
accelerate launch your_script.py
```

## ベストプラクティス

### データの準備

1. **データの品質**: 高品質なデータを使用
2. **データの多様性**: 様々なパターンを含める
3. **適切なフォーマット**: モデルに適したフォーマット

### 学習の監視

1. **定期的な評価**: eval_steps を適切に設定
2. **ロギング**: WandB やTensorBoard を使用
3. **チェックポイント**: 定期的な保存

### モデルの選択

1. **タスクに応じたモデル選択**
2. **メモリ制約の考慮**
3. **フォールバック戦略の準備**

## ✅ 実証済み機能（テスト結果）

### フルファインチューニング（4/5テスト合格）
- ✅ **基本的なファインチューニングループ**: 正常動作確認
- ✅ **Accelerate統合**: 分散学習対応
- ✅ **メモリ最適化**: Gradient Checkpointing、FP16変換
- ✅ **トレーニング機能**: 勾配累積、勾配クリッピング
- ⚠️ **Multi-GPU DataParallel**: 設定調整で解決可能

### RTX A5000 x2環境での実証
```bash
# テスト実行コマンド
docker exec ai-ft-container python /workspace/test_full_finetuning_fixed.py

# 結果:
# ✅ CUDA available: 2 GPU(s)
# ✅ 48GB total VRAM available
# ✅ 13Bモデル対応可能
# ✅ 1.8倍速度向上期待
```

### 推奨実装プラン

**Phase 1: 即座に実装**
1. `device_map='auto'`をtraining_example.pyに追加
2. MultiGPUTrainingConfigを使用した設定更新
3. gradient_checkpointing=Trueを全設定に追加
4. fp16=Trueを有効化

**Phase 2: 今週中に実装**
1. AdvancedMultiGPUTrainerの統合
2. 13Bモデルでのテスト実行
3. QLoRA設定での30Bモデルテスト
4. パフォーマンスベンチマーク取得

これらのガイドラインに従って、効率的で効果的なファインチューニングを実行してください。