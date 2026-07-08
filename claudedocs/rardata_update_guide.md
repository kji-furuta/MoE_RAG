# RARdata.json 更新・追加学習ガイド

**最終更新**: 2025年12月8日 23:45
**対象**: RARdata.jsonにデータを追加・変更した場合の対応方法

---

## 📋 目次

1. [データ更新の種類と対応方法](#データ更新の種類と対応方法)
2. [手順A: 新しいデータを追加（推奨）](#手順a-新しいデータを追加推奨)
3. [手順B: 既存データを修正](#手順b-既存データを修正)
4. [手順C: データを完全に入れ替え](#手順c-データを完全に入れ替え)
5. [データ品質チェック](#データ品質チェック)
6. [トラブルシューティング](#トラブルシューティング)

---

## 📊 データ更新の種類と対応方法

### ケース1: 新しいエントリを追加 ✅ **推奨**

**例**: DES-001～DES-100 → DES-001～DES-150 (50件追加)

```json
[
  {
    "id": "DES-001",
    "instruction": "...",
    ...
  },
  ...
  {
    "id": "DES-150",  // 新規追加
    "instruction": "新しい質問",
    "documents": [...],
    "output": {...}
  }
]
```

**対応方法**: **継続学習（Continual Learning）** を使用
- ✅ 以前の知識を保持
- ✅ 新しい知識を追加
- ✅ Catastrophic Forgetting（破滅的忘却）を防止

---

### ケース2: 既存エントリを修正

**例**: DES-005の回答を改善、引用を追加

```json
{
  "id": "DES-005",
  "instruction": "設計速度80km/hの道路の最小曲線半径は？",
  "output": {
    "chain_of_thought": "改善された推論プロセス",  // 変更
    "final_answer": "より詳細な回答",  // 変更
    "citations": [...]  // 追加
  }
}
```

**対応方法**: **全体を再学習** または **差分のみ継続学習**

---

### ケース3: データ構造を変更

**例**: 新しいフィールド追加、形式変更

```json
{
  "id": "DES-001",
  "instruction": "...",
  "documents": [...],
  "output": {...},
  "metadata": {  // 新規フィールド追加
    "difficulty": "medium",
    "category": "道路構造"
  }
}
```

**対応方法**: **データ検証 → 再学習**

---

## 🚀 手順A: 新しいデータを追加（推奨）

**最も一般的なケース**: 既存の100件に50件追加して150件にする

### ステップ1: データファイルを準備

#### オプション1: 既存ファイルに追加
```bash
# 既存のRARdata.jsonにエントリを追加
# JSONエディタで DES-101 ～ DES-150 を追加
```

#### オプション2: 新しいファイルとして保存
```bash
# 追加分のみの新規ファイルを作成
# RARdata_additional.json (DES-101 ～ DES-150)
```

---

### ステップ2: データ品質チェック

```python
import json

def validate_rar_data(file_path: str):
    """RARデータの品質チェック"""

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"総エントリ数: {len(data)}")

    # ID重複チェック
    ids = [entry['id'] for entry in data]
    duplicates = [id for id in ids if ids.count(id) > 1]

    if duplicates:
        print(f"❌ ID重複: {set(duplicates)}")
        return False
    else:
        print(f"✅ ID重複なし")

    # 必須フィールドチェック
    required_fields = ['id', 'instruction', 'documents', 'output']

    for i, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                print(f"❌ エントリ {i}: {field} が欠落")
                return False

    print(f"✅ 全エントリが有効")

    # Oracle文書の比率
    oracle_count = sum(
        1 for entry in data
        for doc in entry.get('documents', [])
        if doc.get('is_oracle', False)
    )
    total_docs = sum(len(entry.get('documents', [])) for entry in data)
    oracle_ratio = oracle_count / total_docs if total_docs > 0 else 0

    print(f"Oracle比率: {oracle_ratio:.1%} (推奨: 60-70%)")

    return True

# 実行
validate_rar_data('/home/kjifu/MoE_RAG/RARdata.json')
```

---

### ステップ3: Dockerコンテナにコピー

```bash
# ホストからDockerコンテナにファイルをコピー
docker cp /home/kjifu/MoE_RAG/RARdata.json ai-ft-container:/workspace/RARdata_updated.json

# 確認
docker exec ai-ft-container ls -lh /workspace/RARdata_updated.json
```

---

### ステップ4: 継続学習を実行

#### 方法1: CLIスクリプト（推奨）

```bash
# 専用スクリプトを作成
cat > /home/kjifu/MoE_RAG/scripts/update_rardata_model.py << 'EOF'
#!/usr/bin/env python3
"""
RARdata.json更新後の継続学習スクリプト

使用方法:
    python scripts/update_rardata_model.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("RARdata.json 更新後の継続学習")
    logger.info("=" * 80)

    # 設定
    task_name = f"rardata_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_path = "/workspace/RARdata_updated.json"

    # 前回の学習済みモデルをベースに使用
    base_model_path = "outputs/continual_rardata_training_20251208_141953/checkpoint-final"

    # 学習パラメータ
    ewc_lambda = 5000
    epochs = 3
    batch_size = 1
    learning_rate = 2e-5

    logger.info(f"タスク名: {task_name}")
    logger.info(f"データパス: {data_path}")
    logger.info(f"ベースモデル: {base_model_path}")
    logger.info(f"EWC Lambda: {ewc_lambda}")

    # 継続学習パイプライン初期化
    pipeline = ContinualLearningPipeline(
        base_model_path=None,
        use_efficient_fisher=True
    )

    # 前回の学習済みモデルをロード
    model, tokenizer = pipeline.load_finetuned_model(base_model_path)

    logger.info("前回の学習済みモデルをロードしました")

    # 継続学習タスクを開始（前回のFisher行列を使用）
    trained_model = pipeline.run_continual_task(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        train_dataset_path=data_path,
        epochs=epochs,
        use_previous_fisher=True,  # 重要: 前回の知識を保持
        fisher_importance=ewc_lambda,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    logger.info("=" * 80)
    logger.info("継続学習が完了しました！")
    logger.info(f"モデル保存先: outputs/continual_{task_name}/checkpoint-final")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
EOF

# 実行
docker exec ai-ft-container python3 /workspace/scripts/update_rardata_model.py
```

---

#### 方法2: Web UI（簡単）

1. **Continual Learning UIにアクセス**
   ```
   http://localhost:8050/continual
   ```

2. **新しいタスクを作成**
   - **Task Name**: `rardata_update_20251208`
   - **Base Model**: `outputs/continual_rardata_training_20251208_141953/checkpoint-final`
   - **Training Data**: `RARdata_updated.json` をアップロード
   - **Use Previous Fisher Matrix**: ✅ ON（重要！）
   - **EWC Lambda**: `5000`
   - **Epochs**: `3`
   - **Batch Size**: `1`
   - **Learning Rate**: `2e-5`

3. **Start Training** をクリック

4. **進捗モニタリング**
   - ログで学習状況を確認
   - 完了まで待機（データ量により時間は変動）

---

#### 方法3: REST API

```bash
# タスク作成
curl -X POST "http://localhost:8050/api/continual/train" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "rardata_update_20251208",
    "base_model_path": "outputs/continual_rardata_training_20251208_141953/checkpoint-final",
    "train_dataset_path": "/workspace/RARdata_updated.json",
    "epochs": 3,
    "ewc_lambda": 5000,
    "use_previous_fisher": true,
    "batch_size": 1,
    "learning_rate": 2e-5
  }'

# ステータス確認
curl "http://localhost:8050/api/continual/task/rardata_update_20251208"
```

---

### ステップ5: 学習完了後の確認

```bash
# モデルファイルの確認
docker exec ai-ft-container ls -lh /workspace/outputs/continual_rardata_update_*/checkpoint-final/

# テストクエリ実行
curl -X POST "http://localhost:8050/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "新しく追加したデータに関する質問",
    "use_reasoning_model": true
  }'
```

---

## 🔄 手順B: 既存データを修正

**ケース**: DES-001～DES-100の一部（例: 10件）を改善

### 修正パターンによる対応

#### パターン1: 軽微な修正（引用追加、表現改善）

**対応**: 継続学習（手順Aと同じ）

```python
# 修正後のファイルで継続学習
# Base Model: 前回の学習済みモデル
# Use Previous Fisher: True
```

---

#### パターン2: 大幅な修正（50%以上のデータ変更）

**対応**: 全体の再学習を検討

```python
#!/usr/bin/env python3
"""全体再学習スクリプト"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline

def main():
    task_name = "rardata_full_retrain"
    data_path = "/workspace/RARdata_updated.json"

    # 元のベースモデルから再開
    base_model = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"

    pipeline = ContinualLearningPipeline(
        base_model_path=base_model,
        use_efficient_fisher=True
    )

    # Fisher行列をリセットして再学習
    trained_model = pipeline.run_continual_task(
        model=None,  # 新規ロード
        tokenizer=None,
        task_name=task_name,
        train_dataset_path=data_path,
        epochs=3,
        use_previous_fisher=False,  # Fisher行列をリセット
        fisher_importance=5000,
        batch_size=1,
        learning_rate=2e-5
    )

if __name__ == "__main__":
    main()
```

---

## 🔁 手順C: データを完全に入れ替え

**ケース**: 旧データを破棄して新しいデータセットに変更

### ステップ1: 新しいデータファイルを作成

```json
// RARdata_v2.json
[
  {
    "id": "NEW-001",
    "instruction": "完全に新しい質問",
    "documents": [...],
    "output": {...}
  },
  ...
]
```

---

### ステップ2: ゼロから学習

```bash
# 新しいベースモデルから学習開始
docker exec ai-ft-container python3 /workspace/scripts/train_rardata.py \
  --data-path /workspace/RARdata_v2.json \
  --task-name rardata_v2 \
  --base-model cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
```

---

## ✅ データ品質チェック

学習前に必ず実行すべきチェック項目:

### 自動チェックスクリプト

```python
#!/usr/bin/env python3
"""
RARデータ品質チェックスクリプト

使用方法:
    python scripts/check_rar_data.py RARdata_updated.json
"""

import json
import sys
from typing import Dict, List, Tuple

def check_rar_data_quality(file_path: str) -> Tuple[bool, List[str]]:
    """
    RARデータの品質をチェック

    Returns:
        (is_valid, error_messages)
    """

    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"ファイル読み込みエラー: {e}"]

    # 1. ID重複チェック
    ids = [entry.get('id') for entry in data]
    id_counts = {}
    for id in ids:
        id_counts[id] = id_counts.get(id, 0) + 1

    duplicates = [id for id, count in id_counts.items() if count > 1]
    if duplicates:
        errors.append(f"ID重複検出: {duplicates}")

    # 2. 必須フィールドチェック
    required_fields = ['id', 'instruction', 'documents', 'output']

    for i, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                errors.append(f"エントリ {i} (ID: {entry.get('id', 'N/A')}): {field} が欠落")

        # output内の必須フィールド
        if 'output' in entry:
            output = entry['output']
            if 'chain_of_thought' not in output:
                errors.append(f"エントリ {i}: chain_of_thought が欠落")
            if 'final_answer' not in output:
                errors.append(f"エントリ {i}: final_answer が欠落")
            if 'citations' not in output:
                errors.append(f"エントリ {i}: citations が欠落")

    # 3. Oracle文書の比率チェック
    oracle_count = 0
    total_docs = 0

    for entry in data:
        docs = entry.get('documents', [])
        total_docs += len(docs)
        oracle_count += sum(1 for doc in docs if doc.get('is_oracle', False))

    if total_docs > 0:
        oracle_ratio = oracle_count / total_docs
        if oracle_ratio < 0.6 or oracle_ratio > 0.7:
            errors.append(f"Oracle比率が推奨範囲外: {oracle_ratio:.1%} (推奨: 60-70%)")

    # 4. Citation整合性チェック
    for i, entry in enumerate(data):
        citations = entry.get('output', {}).get('citations', [])
        docs = entry.get('documents', [])
        doc_sources = {doc.get('source') for doc in docs}

        for citation in citations:
            if citation.get('source') not in doc_sources:
                errors.append(f"エントリ {i}: Citation元が documents に存在しない")

    # 5. データサイズチェック
    if len(data) == 0:
        errors.append("データが空です")

    print("=" * 60)
    print(f"データ品質チェック結果: {file_path}")
    print("=" * 60)
    print(f"総エントリ数: {len(data)}")
    print(f"ID範囲: {ids[0]} ～ {ids[-1]}" if ids else "N/A")
    print(f"Oracle比率: {oracle_ratio:.1%}" if total_docs > 0 else "N/A")

    if errors:
        print("\n❌ エラー検出:")
        for error in errors:
            print(f"  - {error}")
        return False, errors
    else:
        print("\n✅ 全チェック合格")
        return True, []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python check_rar_data.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    is_valid, errors = check_rar_data_quality(file_path)

    sys.exit(0 if is_valid else 1)
```

**実行例**:
```bash
python scripts/check_rar_data.py /home/kjifu/MoE_RAG/RARdata_updated.json
```

---

## 🛠️ トラブルシューティング

### Q1: 継続学習で前の知識が失われる

**症状**: 新しいデータは学習できたが、以前のデータに関する回答精度が低下

**原因**: EWC Lambda値が低すぎる、または `use_previous_fisher=False` になっている

**解決策**:
```python
# EWC Lambdaを増やす（デフォルト: 5000）
fisher_importance=10000

# Fisher行列を必ず使用
use_previous_fisher=True
```

---

### Q2: 新しいデータが学習されない

**症状**: 学習は完了するが、新データに関する回答が改善されない

**原因**: Learning Rateが低すぎる、またはEpoch数不足

**解決策**:
```python
# Learning Rateを調整
learning_rate=5e-5  # デフォルトの2.5倍

# Epoch数を増やす
epochs=5
```

---

### Q3: メモリ不足エラー

**症状**: `CUDA out of memory` エラー

**解決策**:
```python
# Batch sizeは1を維持
batch_size=1

# Gradient accumulation stepsを調整（内部で自動設定）
# 必要に応じてmax_lengthを短縮
max_length=512  # デフォルトから変更
```

---

### Q4: データ形式エラー

**症状**: `JSONDecodeError` または `KeyError`

**解決策**:
```bash
# データ検証スクリプトを実行
python scripts/check_rar_data.py RARdata_updated.json

# JSON構文チェック
cat RARdata_updated.json | python -m json.tool > /dev/null && echo "OK" || echo "NG"
```

---

## 📊 更新履歴の管理

学習履歴を記録するための推奨フォーマット:

```json
// data/continual_learning/update_history.json
{
  "updates": [
    {
      "date": "2025-12-08",
      "version": "v1.0",
      "data_file": "RARdata.json",
      "num_entries": 100,
      "model_path": "outputs/continual_rardata_training_20251208_141953/checkpoint-final",
      "notes": "初回学習"
    },
    {
      "date": "2025-12-15",
      "version": "v1.1",
      "data_file": "RARdata_updated.json",
      "num_entries": 150,
      "added_entries": 50,
      "modified_entries": 5,
      "model_path": "outputs/continual_rardata_update_20251215/checkpoint-final",
      "base_model": "outputs/continual_rardata_training_20251208_141953/checkpoint-final",
      "notes": "50件追加、5件修正"
    }
  ]
}
```

---

## 🎯 ベストプラクティス

### 1. **段階的な更新**
- 一度に大量のデータを追加するより、小分けにして段階的に更新
- 各更新後に性能評価を実施

### 2. **バージョン管理**
- データファイルにバージョン番号を付与: `RARdata_v1.0.json`, `RARdata_v1.1.json`
- モデルの世代管理

### 3. **品質チェックの自動化**
- CI/CDパイプラインに品質チェックを組み込む
- 学習前に必ず検証スクリプトを実行

### 4. **A/Bテスト**
- 新モデルと旧モデルで性能比較
- 本番適用前に十分なテスト

### 5. **ロールバック戦略**
- 以前のモデルを保持
- 問題発生時に即座に戻せる体制

---

## 📚 関連ドキュメント

- [学習済みモデル使用ガイド](./rardata_current_status.md) - モデルの使用方法
- [RAR学習ガイド](./rar_training_guide.md) - 学習の詳細説明
- [クイックスタート](./rardata_quickstart.md) - 初回学習手順

---

**作成日**: 2025年12月8日 23:45
**対象**: RARdata.json更新時の運用フロー
**次回更新**: 実際の更新操作後、または新しいベストプラクティス発見時
