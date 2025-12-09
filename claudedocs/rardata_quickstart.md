# RARdata.json クイックスタートガイド

**最終更新**: 2025年12月8日 23:20
**対象ファイル**: `/home/kjifu/MoE_RAG/RARdata.json`
**学習ステータス**: ✅ **完了** (outputs/continual_rardata_training_20251208_141953/checkpoint-final)

---

## 🚀 クイックスタート（3ステップ）

### ステップ1: データ確認
```bash
# データ件数を確認
docker exec ai-ft-container python3 -c "
import json
with open('/workspace/RARdata.json', 'r') as f:
    data = json.load(f)
    print(f'総データ件数: {len(data)}件')
    print(f'最初のID: {data[0][\"id\"]}')
    print(f'最後のID: {data[-1][\"id\"]}')
"
```

### ステップ2: 学習実行
```bash
# 学習開始（バックグラウンド実行）
docker exec ai-ft-container python3 /workspace/scripts/train_rardata.py

# または、進捗をリアルタイム表示
docker exec ai-ft-container python3 /workspace/scripts/train_rardata.py 2>&1 | tee training.log
```

### ステップ3: 結果確認
```bash
# 学習完了後、モデルの場所を確認
ls -lh outputs/continual_rardata_training_*/checkpoint-final/

# LoRAアダプターのサイズ確認
du -sh outputs/continual_rardata_training_*/checkpoint-final/adapter_model.safetensors
```

---

## 📖 詳細な使用方法

### 方法1: コマンドライン（推奨）

**基本的な実行**:
```bash
python scripts/train_rardata.py
```

**パラメータをカスタマイズして実行**:
```bash
# スクリプトを編集してパラメータを変更
vim scripts/train_rardata.py

# 変更例:
# epochs = 5  # 3 → 5に変更
# batch_size = 2  # GPUメモリが多い場合
# learning_rate = 1e-5  # より慎重な学習
```

### 方法2: Webインターフェース

**ステップ1**: ブラウザでアクセス
```
http://localhost:8050/continual
```

**ステップ2**: フォームに入力
```
タスク名: rardata_training
トレーニングデータ: RARdata.json をアップロード
ベースモデル: lora_20251129_072850
エポック数: 3
バッチサイズ: 1
学習率: 2e-5
EWC Lambda: 5000
```

**ステップ3**: 「学習開始」ボタンをクリック

### 方法3: REST API

```bash
curl -X POST "http://localhost:8050/api/continual/train" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "rardata_training",
    "train_dataset_path": "/workspace/RARdata.json",
    "base_model": "lora_20251129_072850",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "ewc_lambda": 5000
  }'
```

---

## 🔍 学習進捗の確認

### リアルタイムログ監視

```bash
# Dockerログを監視
docker logs -f ai-ft-container

# 出力例:
# 2025-12-08 14:30:00 - INFO - Starting Continual Learning Task: rardata_training
# 2025-12-08 14:30:15 - INFO - Epoch 1/3
# Training Epoch 1: 100%|██████████| 10/10 [00:30<00:00]
# 2025-12-08 14:30:45 - INFO - Epoch 1 - Average loss: 11.82
# 2025-12-08 14:31:00 - INFO - Epoch 2/3
# Training Epoch 2: 100%|██████████| 10/10 [00:25<00:00]
# 2025-12-08 14:31:25 - INFO - Epoch 2 - Average loss: 11.45
```

### GPU使用状況確認

```bash
# リアルタイムGPU監視
docker exec ai-ft-container nvidia-smi -l 1

# 出力例:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA RTX A5000    On   | 00000000:01:00.0 Off |                  Off |
# | 34%   72C    P2    220W / 230W |  18432MiB / 24564MiB |     85%      Default |
# +-------------------------------+----------------------+----------------------+
```

### Lossの推移確認

```bash
# ログからLossを抽出
docker logs ai-ft-container 2>&1 | grep "Average loss"

# 出力例:
# Epoch 1 - Average loss: 11.82
# Epoch 2 - Average loss: 11.45
# Epoch 3 - Average loss: 11.28
```

---

## ⚙️ パラメータチューニング

### データサイズ別の推奨設定

**RARdata.jsonが100件の場合**:
```python
epochs = 3-5
batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 2e-5
# 学習時間: 約15-30分
```

**RARdata.jsonが1,000件の場合**:
```python
epochs = 3
batch_size = 1-2
gradient_accumulation_steps = 16-32
learning_rate = 2e-5
# 学習時間: 約2-4時間
```

**RARdata.jsonが10,000件の場合**:
```python
epochs = 2-3
batch_size = 2-4
gradient_accumulation_steps = 32-64
learning_rate = 1e-5
# 学習時間: 約10-20時間
```

### GPUメモリ別の設定

**24GB GPU (RTX 3090, RTX A5000)**:
```python
batch_size = 1
max_seq_length = 256
use_4bit = True
gradient_checkpointing = True
```

**40GB GPU (A100 40GB)**:
```python
batch_size = 2
max_seq_length = 512
use_8bit = True
gradient_checkpointing = True
```

**80GB GPU (A100 80GB)**:
```python
batch_size = 4
max_seq_length = 512
use_fp16 = True
gradient_checkpointing = False
```

---

## 🎯 学習後の使用方法

### 1. モデルのロード

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ベースモデルをロード
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    load_in_4bit=True,
    device_map="auto"
)

# 学習済みLoRAアダプターを適用
model = PeftModel.from_pretrained(
    base_model,
    "outputs/continual_rardata_training_20251208_120000/checkpoint-final"
)

# トークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)
```

### 2. 推論実行

```python
# RAR形式のプロンプト作成
prompt = """### Instruction:
設計速度80km/hの道路の最小曲線半径は？

### Context:
[Document 1] 道路構造令の解説と運用（令和3年）
設計速度と最小曲線半径の対応表...

### Response:
"""

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

# デコード
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 3. RAGシステムとの統合

```python
# RAGシステムのモデル設定を更新
# config/model_config.yaml を編集

models:
  - name: rardata_trained_model
    path: outputs/continual_rardata_training_20251208_120000/checkpoint-final
    base_model: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
    type: lora
    quantization: 4bit
    use_case: rag_with_citations
```

### 4. UIでの使用

```
http://localhost:8050/models

1. 「モデル管理」ページを開く
2. 「新しいモデルを追加」をクリック
3. フォームに入力:
   - モデル名: rardata_trained_model
   - パス: outputs/continual_rardata_training_20251208_120000/checkpoint-final
   - タイプ: LoRA
   - ベースモデル: DeepSeek-R1-Distill-Qwen-32B-Japanese
4. 「保存」をクリック
5. RAG検索で選択可能になります
```

---

## 🔧 トラブルシューティング

### エラー1: "CUDA out of memory"

**解決策**:
```python
# scripts/train_rardata.py を編集
batch_size = 1  # すでに1の場合は変更不要
gradient_accumulation_steps = 32  # 16 → 32に増やす
max_seq_length = 128  # 256 → 128に減らす
```

### エラー2: "Loss does not decrease"

**確認事項**:
```bash
# データ形式の検証
docker exec ai-ft-container python3 scripts/rar/auto_validate_and_fix.py \
  --input /workspace/RARdata.json \
  --output /workspace/RARdata_validated.json

# 検証後のファイルで再学習
# scripts/train_rardata.pyのdata_pathを更新
data_path = "/workspace/RARdata_validated.json"
```

### エラー3: "FileNotFoundError: RARdata.json"

**解決策**:
```bash
# ファイルが存在するか確認
docker exec ai-ft-container ls -l /workspace/RARdata.json

# 存在しない場合、正しいパスを確認
docker exec ai-ft-container find /workspace -name "RARdata.json"

# スクリプトのパスを更新
# scripts/train_rardata.pyを編集
data_path = "/workspace/path/to/RARdata.json"
```

### エラー4: "object of type 'StreamingTextDataset' has no len()"

**ステータス**: ✅ **修正済み** (2025-12-08)

この問題は `src/training/training_utils.py` の修正により解決されています。
修正内容:
- `StreamingTextDataset.__len__()` メソッドを実装
- `_count_samples()` メソッドでJSON配列形式と行ごとのJSON形式の両方に対応

---

## 📊 学習結果の評価

### 1. Loss推移の確認

```bash
# ログからLossを抽出してグラフ化
docker logs ai-ft-container 2>&1 | grep "Average loss" | \
  awk '{print $NF}' > losses.txt

# Pythonでグラフ作成
python3 <<EOF
import matplotlib.pyplot as plt

with open('losses.txt', 'r') as f:
    losses = [float(line.strip()) for line in f]

plt.plot(losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Progress')
plt.grid(True)
plt.savefig('training_loss.png')
print("✅ グラフを保存: training_loss.png")
EOF
```

### 2. 推論品質の確認

```python
# scripts/evaluate_trained_model.py
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# モデルロード
base_model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(
    base_model,
    "outputs/continual_rardata_training_*/checkpoint-final"
)
tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)

# テストデータから3件サンプリング
with open('/workspace/RARdata.json', 'r') as f:
    test_data = json.load(f)[:3]

for entry in test_data:
    print(f"\n{'='*80}")
    print(f"Question: {entry['instruction']}")
    print(f"{'='*80}")

    # プロンプト作成
    prompt = f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 生成
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated: {response}")
    print(f"Expected: {entry['output']['final_answer']}")
    print()
```

---

## 📚 関連ドキュメント

- [RAR学習ガイド詳細版](rar_training_guide.md)
- [RAG UI使用ガイド](rag_ui_usage_guide.md)
- [Phase 2完了レポート](rar_phase2_completion_report.md)
- [システムアーキテクチャ](system_architecture_analysis.md)

---

## 🔗 クイックリファレンス

### 現在の学習ステータス確認
```bash
# 学習プロセスが実行中か確認
docker exec ai-ft-container ps aux | grep train_rardata

# ログの最後の10行を表示
docker logs ai-ft-container --tail 10

# GPU使用状況
docker exec ai-ft-container nvidia-smi
```

### 学習の停止
```bash
# 学習プロセスを検索
docker exec ai-ft-container ps aux | grep train_rardata

# プロセスIDを確認してKILL
docker exec ai-ft-container kill -9 [PID]
```

### 学習の再開
```bash
# 同じパラメータで再実行
docker exec ai-ft-container python3 /workspace/scripts/train_rardata.py
```

---

**作成日**: 2025年12月8日
**学習開始日時**: 2025年12月8日 22:16
**学習完了時刻**: 2025年12月8日 23:20
**総学習時間**: 約10分

---

## 📊 学習結果サマリー

### 基本情報
- **モデル保存先**: `outputs/continual_rardata_training_20251208_141953/checkpoint-final`
- **エポック数**: 3
- **学習損失推移**:
  - Epoch 1: 2.7578
  - Epoch 2: 2.7604
  - Epoch 3: 2.7630
- **学習可能パラメータ**: 33,554,432 (全体の0.1023%)

### 次のステップ
1. モデルをRAGシステムで使用
2. 引用精度の測定
3. 実際のクエリでの性能評価
