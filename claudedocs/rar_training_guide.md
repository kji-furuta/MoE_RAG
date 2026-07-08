# RARデータの学習方法ガイド

**最終更新**: 2025年12月8日
**対象データ**: RAR (Retrieval-Augmented Reasoning) 形式のJSONデータ
**例**: `RARdata.json`, `rar_1000_simulated.json`, `rar_pilot_100.json`

---

## 📋 目次

1. [RARデータ形式の理解](#1-rarデータ形式の理解)
2. [学習の基本的な流れ](#2-学習の基本的な流れ)
3. [コマンドラインでの学習実行](#3-コマンドラインでの学習実行)
4. [UIでの学習実行](#4-uiでの学習実行)
5. [学習パラメータの調整](#5-学習パラメータの調整)
6. [トラブルシューティング](#6-トラブルシューティング)

---

## 1. RARデータ形式の理解

### 1.1 RAR形式の構造

RARdata.jsonは以下の構造を持っています：

```json
[
  {
    "id": "DES-001",
    "instruction": "地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、道路構造令の種別上、何種道路に分類されますか？",
    "documents": [
      {
        "source": "【日本道路協会】道路構造令の解説と運用（令和3年）_R3年3月-5.pdf",
        "is_oracle": true,
        "content": "道路の存する地域 高速自動車国道及び..."
      }
    ],
    "output": {
      "chain_of_thought": "1. ユーザーは地方部における...",
      "final_answer": "地方部に存在する高速自動車国道及び...",
      "citations": [
        {
          "source": "【日本道路協会】道路構造令の解説と運用...",
          "quote": "道路の存する地域... 地方部... その他の 道路 第 3 種"
        }
      ]
    }
  }
]
```

### 1.2 各フィールドの意味

| フィールド | 説明 | 学習での役割 |
|----------|------|------------|
| **id** | エントリーの一意識別子 | データ管理・追跡用 |
| **instruction** | ユーザーからの質問 | **入力（Input）** |
| **documents** | 参照ドキュメント群 | **コンテキスト（Context）** |
| **documents[].is_oracle** | 正解を含むか | Oracle: 正解、Distractor: ノイズ |
| **output.chain_of_thought** | 推論プロセス | **思考過程の学習** |
| **output.final_answer** | 最終回答 | **出力（Output）** |
| **output.citations** | 引用情報 | **引用生成の学習** |

### 1.3 学習時のプロンプト形式

RARデータは以下のようなプロンプト形式に変換されて学習されます：

```
### Instruction:
地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、道路構造令の種別上、何種道路に分類されますか？

### Context:
[Document 1] 【日本道路協会】道路構造令の解説と運用（令和3年）_R3年3月-5.pdf
道路の存する地域 高速自動車国道及び自動車専用i路又はその他の道路の別^^^ 地方部 都市部...

### Response:
**Chain-of-Thought:**
1. ユーザーは地方部における「高速自動車国道及び自動車専用道路以外の道路」（その他の道路）の種別を求めている。
2. ソースの表3.1/表から、地方部でその他の道路に該当するのは「第3種」であることを特定する。

**Answer:**
地方部に存在する高速自動車国道及び自動車専用道路以外の道路は、道路構造令の種別上、**第3種道路**に分類されます。

**Citations:**
[1] 【日本道路協会】道路構造令の解説と運用（令和3年）_R3年3月-5.pdf
「道路の存する地域... 地方部... その他の 道路 第 3 種」
```

---

## 2. 学習の基本的な流れ

### 2.1 学習プロセスの概要

```
┌─────────────────────────────────────────────────────────────┐
│  1. データ準備                                               │
│  ├─ RARdata.jsonをdata/rar_training/に配置                 │
│  └─ データ形式の検証（auto_validate_and_fix.pyを使用）      │
├─────────────────────────────────────────────────────────────┤
│  2. ベースモデルの選択                                       │
│  ├─ 既存のファインチューニング済みモデルを使用              │
│  └─ または、新規にベースモデルから開始                      │
├─────────────────────────────────────────────────────────────┤
│  3. 学習パラメータの設定                                     │
│  ├─ エポック数: 3-5                                         │
│  ├─ バッチサイズ: 1-2（GPUメモリに応じて）                 │
│  ├─ 学習率: 2e-5                                            │
│  └─ EWC Lambda: 5000（継続学習の場合）                     │
├─────────────────────────────────────────────────────────────┤
│  4. 学習実行                                                 │
│  ├─ EWC継続学習パイプライン使用                            │
│  ├─ LoRAアダプターで効率的に学習                           │
│  └─ 進捗モニタリング（Loss、LR）                           │
├─────────────────────────────────────────────────────────────┤
│  5. モデルの保存                                             │
│  ├─ outputs/continual_[タスク名]_[日時]/                  │
│  └─ LoRAアダプター（adapter_model.safetensors）            │
├─────────────────────────────────────────────────────────────┤
│  6. 評価とテスト                                             │
│  ├─ 検証データでの性能確認                                  │
│  ├─ RAG検索との統合テスト                                   │
│  └─ 引用精度の評価                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 学習アルゴリズム

**EWC (Elastic Weight Consolidation) 継続学習**:
- 以前のタスクの知識を保持しながら新しいタスクを学習
- Fisher Information Matrixで重要なパラメータを特定
- 重要なパラメータの変更を制限（EWC Lambda: 5000）

**LoRA (Low-Rank Adaptation)**:
- パラメータ効率的なファインチューニング
- 全パラメータの約0.2%のみを学習（33M / 17B）
- GPUメモリを大幅に削減（4-bit量子化と組み合わせ）

---

## 3. コマンドラインでの学習実行

### 3.1 基本的な学習スクリプト

**ステップ1**: データを配置
```bash
# RARdata.jsonをdata/rar_training/に配置
cp RARdata.json data/rar_training/my_training_data.json
```

**ステップ2**: 学習スクリプトを作成

```python
# scripts/train_rar_data.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline

def main():
    # 設定
    task_name = "my_rar_training"
    data_path = "/workspace/data/rar_training/my_training_data.json"
    model_path = "outputs/lora_20251129_072850"  # 既存モデル

    # パラメータ
    ewc_lambda = 5000
    epochs = 3
    batch_size = 1
    learning_rate = 2e-5
    gradient_accumulation_steps = 16

    # パイプライン初期化
    pipeline = ContinualLearningPipeline(
        base_model_path=None,
        use_efficient_fisher=True
    )

    # モデルロード
    print(f"モデルをロード中: {model_path}")
    model, tokenizer = pipeline.load_finetuned_model(model_path)

    # 学習実行
    print(f"学習開始: {task_name}")
    trained_model = pipeline.run_continual_task(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        train_dataset_path=data_path,
        epochs=epochs,
        use_previous_fisher=True,
        fisher_importance=ewc_lambda,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    print(f"✅ 学習完了！")
    print(f"保存先: outputs/continual_{task_name}_[日時]/checkpoint-final/")

if __name__ == "__main__":
    main()
```

**ステップ3**: 実行
```bash
docker exec ai-ft-container python /workspace/scripts/train_rar_data.py
```

### 3.2 既存スクリプトの使用

**方法1**: Phase 1スクリプトを流用

```bash
# scripts/rar/run_rar_pilot_training.pyをコピー
cp scripts/rar/run_rar_pilot_training.py scripts/train_my_rar.py

# スクリプト内のパスを編集
# data_path = "/workspace/data/rar_training/RARdata.json"
# task_name = "my_rar_task"

# 実行
docker exec ai-ft-container python /workspace/scripts/train_my_rar.py
```

**方法2**: 直接パラメータを編集

```bash
docker exec ai-ft-container python /workspace/scripts/rar/run_rar_pilot_training.py \
  --data-path /workspace/data/rar_training/RARdata.json \
  --task-name my_rar_training \
  --epochs 5 \
  --batch-size 2
```

### 3.3 学習の進捗確認

```bash
# リアルタイムログ監視
docker logs -f ai-ft-container

# 出力例:
# 2025-12-08 11:45:21 - INFO - Starting Continual Learning Task: my_rar_training
# 2025-12-08 11:45:27 - INFO - Epoch 1 - Average loss: 12.0434
# 2025-12-08 11:45:29 - INFO - Epoch 2 - Average loss: 12.0504
# 2025-12-08 11:45:31 - INFO - Epoch 3 - Average loss: 12.0434
# 2025-12-08 11:45:32 - INFO - Training completed!
```

---

## 4. UIでの学習実行

### 4.1 Webインターフェースでの学習

**ステップ1**: ブラウザでアクセス
```
http://localhost:8050/continual
```

**ステップ2**: 継続学習タスク作成

```
┌───────────────────────────────────────────────────────────┐
│  継続学習インターフェース                                  │
├───────────────────────────────────────────────────────────┤
│  タスク名: [my_rar_training              ]                │
│                                                           │
│  トレーニングデータ:                                      │
│  [ ファイルを選択 ] RARdata.json                          │
│                                                           │
│  ベースモデル: [lora_20251129_072850    ] ▼              │
│                                                           │
│  学習パラメータ:                                          │
│  ├─ エポック数: [3]                                      │
│  ├─ バッチサイズ: [1]                                    │
│  ├─ 学習率: [2e-5]                                       │
│  └─ EWC Lambda: [5000]                                   │
│                                                           │
│  [ 学習開始 ]  [ キャンセル ]                            │
└───────────────────────────────────────────────────────────┘
```

**ステップ3**: 進捗確認

```
┌───────────────────────────────────────────────────────────┐
│  学習進捗: my_rar_training                                │
├───────────────────────────────────────────────────────────┤
│  状態: 🔄 学習中                                          │
│                                                           │
│  進捗:                                                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ ████████████████████░░░░░░░░░░░░ 66% (Epoch 2/3)    │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  現在のLoss: 12.05                                        │
│  経過時間: 5分30秒                                        │
│  推定残り時間: 3分                                        │
│                                                           │
│  ログ:                                                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 2025-12-08 11:45:27 - Epoch 1 - Loss: 12.04        │ │
│  │ 2025-12-08 11:45:29 - Epoch 2 - Loss: 12.05        │ │
│  │ 2025-12-08 11:45:31 - Training...                   │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 4.2 REST APIでの学習

**エンドポイント**: `POST /api/continual/train`

```bash
curl -X POST "http://localhost:8050/api/continual/train" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "my_rar_training",
    "train_dataset_path": "/workspace/data/rar_training/RARdata.json",
    "base_model": "lora_20251129_072850",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "ewc_lambda": 5000
  }'
```

**レスポンス**:
```json
{
  "task_id": "my_rar_training_20251208_120000",
  "status": "started",
  "message": "継続学習タスクを開始しました"
}
```

**進捗確認**:
```bash
curl "http://localhost:8050/api/continual/task/my_rar_training_20251208_120000"
```

---

## 5. 学習パラメータの調整

### 5.1 データサイズ別の推奨設定

| データ件数 | エポック数 | バッチサイズ | Gradient Accumulation | 学習時間（概算） |
|-----------|----------|------------|----------------------|----------------|
| **100件** | 3-5 | 1 | 16 | 15-30分 |
| **1,000件** | 3-5 | 1-2 | 16-32 | 2-4時間 |
| **10,000件** | 2-3 | 2-4 | 32-64 | 10-20時間 |

### 5.2 GPUメモリ別の設定

| GPU | VRAM | 量子化 | バッチサイズ | 推奨モデルサイズ |
|-----|------|--------|------------|----------------|
| **RTX 3090** | 24GB | 4-bit | 1 | 32B |
| **RTX A5000** | 24GB | 4-bit | 1-2 | 32B |
| **A100** | 40GB | 8-bit | 2-4 | 32B-70B |
| **A100** | 80GB | FP16 | 4-8 | 70B+ |

### 5.3 パラメータチューニングガイド

#### エポック数 (epochs)

**少なすぎる（1-2エポック）**:
- ❌ 学習不足、モデルが十分に適応しない
- ✅ 使用ケース: 大規模データセット（10,000件以上）

**適切（3-5エポック）**:
- ✅ 通常のファインチューニングに最適
- ✅ 過学習リスクが低い

**多すぎる（10エポック以上）**:
- ❌ 過学習リスク増大
- ❌ 学習時間の無駄

#### 学習率 (learning_rate)

| 学習率 | 効果 | 推奨ケース |
|--------|------|----------|
| **1e-6** | 非常に小さい変更 | 既に高性能なモデルの微調整 |
| **2e-5** | 標準的（推奨） | 通常のファインチューニング |
| **5e-5** | 大きい変更 | 新規タスクへの大幅適応 |
| **1e-4** | 非常に大きい | リスク高、通常は推奨しない |

#### EWC Lambda

| EWC Lambda | 効果 | 推奨ケース |
|-----------|------|----------|
| **0** | EWC無効化 | 初回学習、継続学習不要 |
| **1000** | 弱い制約 | 新しいタスクへの大幅適応 |
| **5000** | 標準的（推奨） | 通常の継続学習 |
| **10000** | 強い制約 | 以前の知識を厳密に保持 |

---

## 6. トラブルシューティング

### 6.1 よくあるエラーと解決方法

#### エラー1: "object of type 'StreamingTextDataset' has no len()"

**原因**: Fisher行列計算時に`StreamingTextDataset`の長さが取得できない

**解決策**:
```python
# src/training/training_utils.py の StreamingTextDataset クラスに追加

def __len__(self):
    """データセットの長さを返す"""
    if not hasattr(self, '_length'):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self._length = len(data) if isinstance(data, list) else 0
    return self._length
```

**一時的な回避策**:
```python
# use_efficient_fisher=False に設定
pipeline = ContinualLearningPipeline(
    base_model_path=None,
    use_efficient_fisher=False  # Fisher行列計算を無効化
)
```

#### エラー2: "CUDA out of memory"

**解決策**:

**方法1**: バッチサイズを減らす
```python
batch_size = 1  # 2 → 1に変更
```

**方法2**: Gradient Accumulationを増やす
```python
gradient_accumulation_steps = 32  # 16 → 32に変更
# 実効バッチサイズは同じだがメモリ使用量は半分
```

**方法3**: シーケンス長を短縮
```python
max_length = 256  # 512 → 256に変更
```

#### エラー3: "Loss does not decrease"

**原因**: 学習率が不適切、データが不適切

**確認事項**:
```bash
# データ形式の検証
python scripts/rar/auto_validate_and_fix.py \
  --input data/rar_training/RARdata.json \
  --output data/rar_training/RARdata_validated.json
```

**解決策**:
```python
# 学習率を調整
learning_rate = 1e-5  # 2e-5 → 1e-5 に減らす

# またはウォームアップを追加
warmup_steps = 100
```

#### エラー4: "Model not found"

**解決策**:
```bash
# 利用可能なモデルを確認
ls outputs/

# モデルパスを相対パスで指定
model_path = "outputs/lora_20251129_072850"  # /workspace/ を付けない
```

### 6.2 学習品質の確認

#### Loss の推移チェック

**正常な学習**:
```
Epoch 1: Loss 12.04
Epoch 2: Loss 11.82  ← 減少している
Epoch 3: Loss 11.65  ← さらに減少
```

**問題がある学習**:
```
Epoch 1: Loss 12.04
Epoch 2: Loss 12.05  ← ほとんど変化なし
Epoch 3: Loss 12.04  ← 改善が見られない
```
→ 学習率を上げる、データを確認

#### 推論テスト

```python
# scripts/test_trained_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# モデルロード
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(
    base_model,
    "outputs/continual_my_rar_training_20251208_120000/checkpoint-final"
)
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)

# テスト質問
prompt = """### Instruction:
設計速度80km/hの道路の最小曲線半径は？

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

---

## 📚 関連ドキュメント

- [RAR Phase 2完了レポート](rar_phase2_completion_report.md)
- [RAR Phase 2サマリー](rar_phase2_summary.md)
- [継続学習パイプライン](../src/training/continual_learning_pipeline.py)
- [EWC実装詳細](../src/training/ewc_full_finetuning.py)

---

## 🔗 クイックリファレンス

### 最小限の学習スクリプト

```python
from src.training.continual_learning_pipeline import ContinualLearningPipeline

pipeline = ContinualLearningPipeline(base_model_path=None, use_efficient_fisher=False)
model, tokenizer = pipeline.load_finetuned_model("outputs/lora_20251129_072850")
pipeline.run_continual_task(
    model=model,
    tokenizer=tokenizer,
    task_name="my_task",
    train_dataset_path="/workspace/data/rar_training/RARdata.json",
    epochs=3,
    batch_size=1,
    learning_rate=2e-5
)
```

### コマンドライン実行

```bash
# 基本的な実行
docker exec ai-ft-container python scripts/train_my_rar.py

# ログ監視
docker logs -f ai-ft-container

# GPU使用状況確認
docker exec ai-ft-container nvidia-smi
```

---

**作成日**: 2025年12月8日
**対象バージョン**: MoE_RAG v1.0 (Phase 2完了版)
**次回更新**: Fisher行列エラー修正後、または重要な機能追加時
