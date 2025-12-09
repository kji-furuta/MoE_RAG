# RARdata.json データ更新・学習ガイド（UI統合版）

**最終更新**: 2025年12月9日 00:30
**対象**: 継続学習UIを使用したRARdata.jsonの更新・追加学習

---

## 📋 概要

本システムには**継続学習UI**が統合されており、RARdata.jsonを増加・変更した際の学習操作を
Web UIから簡単に実行できます。

### アクセス方法
```
http://localhost:8050/
→ 「Continual Learning」タブをクリック
```

---

## 🚀 RARdata.json更新時の完全フロー（UI版）

### ステップ0: 事前準備（推奨）

#### データ品質チェック

学習前に品質チェックを実施することを強く推奨します。

```bash
# ホストマシンで実行
python3 scripts/check_rar_data.py /home/kjifu/MoE_RAG/RARdata.json
```

**チェック項目**:
- ✅ JSON構文の妥当性
- ✅ ID重複の確認
- ✅ 必須フィールド（id, instruction, documents, output）
- ✅ Oracle文書比率（推奨: 60-70%）
- ✅ Citation整合性

**出力例**:
```
================================================================================
RARデータ品質チェック結果: RARdata.json
================================================================================

📊 データ統計:
  total_entries: 150
  id_range: DES-001 ～ DES-150
  oracle_ratio: 68.2%
  avg_cot_length: 125文字
  avg_answer_length: 73文字

✅ 全チェック合格 - 学習に使用できます
================================================================================
```

---

### ステップ1: 継続学習UIにアクセス

1. **ブラウザでアクセス**
   ```
   http://localhost:8050/
   ```

2. **「Continual Learning」タブをクリック**

   メイン画面上部のタブから選択します。

---

### ステップ2: 学習設定

継続学習UIには3つのサブタブがあります：

#### **2-1. Training タブ（学習実行）**

以下の設定を入力します：

| 項目 | 設定値 | 説明 |
|-----|-------|------|
| **Task Name** | `rardata_update_20251209` | タスク識別名（任意） |
| **Base Model** | `outputs/continual_rardata_training_20251208_141953/checkpoint-final` | 前回の学習済みモデルを選択 |
| **Dataset File** | RARdata.json をアップロード | 更新後のデータファイル（JSON/JSONL対応） |
| **Use Previous Fisher Matrix** | ✅ **ON** | **重要**: 既存知識を保持 |
| **EWC Lambda** | `5000` | 既存知識の保持強度 |
| **Epochs** | `3` | 学習エポック数 |
| **Learning Rate** | `0.00002` (2e-5) | 学習率 |

**注意**: ファイル形式はJSON配列形式（RARdata.json）またはJSONL形式の両方に対応しています。

---

#### **重要設定の詳細**

##### ✅ **Use Previous Fisher Matrix（必須）**

**必ず ON にしてください！**

- ✅ **ON**: 前回の学習内容を保持しながら新しい知識を追加（継続学習）
- ❌ **OFF**: 前回の知識が失われる（破滅的忘却）

**例**:
- 既存100件 + 新規50件 = 150件の知識を持つモデル（ON）
- 新規50件のみの知識を持つモデル（OFF）← **避けるべき**

---

##### **Base Model の選択**

**ドロップダウンから前回の学習済みモデルを選択**

利用可能なモデル例:
```
outputs/continual_rardata_training_20251208_141953/checkpoint-final  ← 前回学習済み
outputs/continual_rardata_update_20251208/checkpoint-final
cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese  ← 元のベースモデル
```

**選ぶべきモデル**:
- ✅ `outputs/continual_rardata_*` で始まる**最新の学習済みモデル**
- ❌ 元のベースモデル（継続学習にならない）

---

##### **EWC Lambda の調整**

EWC Lambda値は、既存知識と新知識のバランスを調整します。

| 値 | 効果 | 使用ケース |
|---|------|----------|
| `3000` | 新知識優先 | 大幅なデータ変更 |
| `5000` | **バランス（推奨）** | 通常の追加・修正 |
| `7000-10000` | 既存知識保持優先 | 軽微な修正のみ |

**推奨**: 初回は `5000` を使用し、結果に応じて調整

---

### ステップ3: 学習開始

1. **「Start Training」ボタンをクリック**

2. **進捗モニタリング**
   - 学習が開始されるとステータスが表示されます
   - ログでリアルタイム進捗を確認できます

3. **完了確認**
   - 学習完了後、成功メッセージが表示されます
   - 新しいモデルが自動的に保存されます

---

### ステップ4: 学習結果の確認

#### **4-1. Tasks タブで確認**

1. **「Tasks」タブをクリック**

2. **タスク一覧が表示**
   ```
   Task Name: rardata_update_20251209
   Status: Completed
   Model Path: outputs/continual_rardata_update_20251209/checkpoint-final
   Created: 2025-12-09 00:30
   ```

3. **詳細情報**
   - Training Loss
   - Epoch数
   - 学習時間

---

#### **4-2. History タブで履歴確認**

1. **「History」タブをクリック**

2. **過去の学習履歴が表示**
   ```
   2025-12-09 00:30 - rardata_update_20251209 - Completed
   2025-12-08 22:16 - rardata_training - Completed
   ```

---

### ステップ5: 学習済みモデルの使用

#### **方法A: RAGシステムで自動使用**

RAGシステムは自動的に最新の学習済みモデルを使用します。

1. **RAGタブにアクセス**
   ```
   http://localhost:8050/rag
   ```

2. **クエリを入力**
   ```
   設計速度80km/hの道路の最小曲線半径は？
   ```

3. **Advanced Options を展開**
   - **Use Reasoning Model**: ✅ ON

4. **Search をクリック**

5. **結果確認**
   - 新しく学習したデータに基づく回答が表示されます
   - Chain-of-Thought推論プロセスが表示されます
   - 引用元が明示されます

---

#### **方法B: REST API経由**

```bash
curl -X POST "http://localhost:8050/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "設計速度80km/hの道路の最小曲線半径は？",
    "use_reasoning_model": true
  }'
```

---

#### **方法C: モデルを明示的に指定**

特定の世代のモデルを使用したい場合:

```bash
curl -X POST "http://localhost:8050/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "テストクエリ",
    "model_path": "outputs/continual_rardata_update_20251209/checkpoint-final"
  }'
```

---

## 📊 実際の操作例

### 例1: 50件のデータを追加

**シナリオ**: 既存100件に50件追加して150件にする

#### UIでの操作:

1. **事前チェック**
   ```bash
   python3 scripts/check_rar_data.py RARdata_updated.json
   ```

2. **継続学習UI設定**
   - Task Name: `rardata_add_50`
   - Base Model: `outputs/continual_rardata_training_20251208_141953/checkpoint-final`
   - Dataset: RARdata_updated.json（150件）をアップロード
   - Use Previous Fisher: ✅ ON
   - EWC Lambda: `5000`
   - Epochs: `3`

3. **学習開始**
   - 「Start Training」をクリック
   - 推定時間: 10-15分

4. **結果確認**
   - Tasks タブで完了確認
   - RAGで新データに関するクエリをテスト

---

### 例2: 既存データの改善（10件修正）

**シナリオ**: DES-001～DES-010の回答品質を改善

#### UIでの操作:

1. **継続学習UI設定**
   - Task Name: `rardata_improve_10`
   - Base Model: 最新の学習済みモデル
   - Dataset: RARdata_improved.json（修正版）をアップロード
   - Use Previous Fisher: ✅ ON
   - EWC Lambda: `7000` ← 既存知識をより強く保持
   - Epochs: `3`

2. **学習開始**

3. **A/Bテスト**
   - 旧モデルと新モデルで同じクエリを実行
   - 改善を確認

---

### 例3: 完全な再学習

**シナリオ**: データセットを完全に入れ替え

#### UIでの操作:

1. **継続学習UI設定**
   - Task Name: `rardata_v2`
   - Base Model: `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese` ← 元のベースモデル
   - Dataset: RARdata_v2.json をアップロード
   - Use Previous Fisher: ❌ **OFF** ← Fisher行列をリセット
   - EWC Lambda: `5000`（使用されない）
   - Epochs: `3`

2. **学習開始**
   - 元のベースモデルから再学習されます

---

## ⚙️ 詳細設定ガイド

### EWC Lambda のチューニング

| 症状 | 原因 | 対処法 |
|-----|------|-------|
| 新データが学習されない | Lambda値が高すぎる | `3000-4000`に下げる |
| 旧データの精度が低下 | Lambda値が低すぎる | `7000-10000`に上げる |
| バランスが取れている | 適切な設定 | `5000`を維持 |

---

### Learning Rate のチューニング

| 値 | 効果 | 使用ケース |
|---|------|----------|
| `1e-5` | 保守的な学習 | 既存知識を重視 |
| `2e-5` | **標準（推奨）** | 通常の継続学習 |
| `5e-5` | 積極的な学習 | 新知識を優先 |

---

### Epochs 数の調整

| エポック数 | 学習時間（目安） | 使用ケース |
|----------|--------------|----------|
| `2` | 短い | 軽微な更新 |
| `3` | **標準（推奨）** | 通常の更新 |
| `5` | 長い | 大規模な更新 |

---

## 🔍 トラブルシューティング

### Q1: 「Model not found」エラー

**原因**: Base Modelのパスが正しくない

**解決策**:
```bash
# 利用可能なモデルを確認
docker exec ai-ft-container ls -lh /workspace/outputs/continual_*/checkpoint-final/
```

---

### Q2: アップロードエラー（ファイルが選択できない）

**原因1**: ファイル形式が対応していない（修正済み）

**解決策**:
- UIは`.json`と`.jsonl`の両方に対応しています
- RARdata.jsonは選択可能です

**原因2**: ファイルサイズが大きすぎる、またはJSON形式が不正

**解決策**:
```bash
# JSON構文チェック
python3 -m json.tool RARdata.json > /dev/null && echo "OK" || echo "NG"

# 品質チェック実行
python3 scripts/check_rar_data.py RARdata.json
```

**原因3**: ブラウザのキャッシュ問題

**解決策**:
```
1. ブラウザをリロード（Ctrl+F5 または Cmd+Shift+R）
2. キャッシュをクリア
3. ブラウザを再起動
```

---

### Q3: 学習が進まない

**症状**: Loss値が変化しない

**解決策**:
1. **Learning Rateを上げる**: `2e-5` → `5e-5`
2. **Epoch数を増やす**: `3` → `5`
3. **ログを確認**: Tasks タブで詳細ログを確認

---

### Q4: メモリ不足エラー

**症状**: `CUDA out of memory`

**解決策**:
```bash
# GPUメモリをクリア
docker exec ai-ft-container python3 -c "import torch; torch.cuda.empty_cache()"

# コンテナ再起動
docker restart ai-ft-container
```

---

## 📋 チェックリスト

### 学習前チェックリスト

- [ ] データ品質チェック実行済み（`check_rar_data.py`）
- [ ] JSONファイルのバックアップ取得済み
- [ ] Base Modelに前回の学習済みモデルを指定
- [ ] **Use Previous Fisher Matrix が ON**
- [ ] EWC Lambdaが適切（通常は `5000`）

---

### 学習後チェックリスト

- [ ] Tasks タブで完了確認
- [ ] モデルファイルが生成されている（outputs/continual_*/checkpoint-final/）
- [ ] RAGシステムでテストクエリ実行
- [ ] 新データに関する回答精度が向上
- [ ] 既存データに関する回答精度が維持
- [ ] 性能評価結果を記録

---

## 🎯 ベストプラクティス

### 1. 段階的な更新

**推奨アプローチ**:
```
v1.0: 100件（初回学習）
  ↓ 20件追加
v1.1: 120件（継続学習）
  ↓ 30件追加
v1.2: 150件（継続学習）
```

**非推奨**:
```
v1.0: 100件
  ↓ 一度に100件追加
v2.0: 200件  ← リスクが高い
```

---

### 2. バージョン管理

**ファイル命名規則**:
```
RARdata_v1.0.json  (100件 - 初回)
RARdata_v1.1.json  (120件 - 20件追加)
RARdata_v1.2.json  (150件 - 30件追加)
```

**履歴ファイル**:
```json
// data/update_history.json
{
  "updates": [
    {
      "version": "v1.0",
      "date": "2025-12-08",
      "entries": 100,
      "model": "outputs/continual_rardata_training_20251208_141953/checkpoint-final"
    },
    {
      "version": "v1.1",
      "date": "2025-12-09",
      "entries": 150,
      "added": 50,
      "model": "outputs/continual_rardata_update_20251209/checkpoint-final"
    }
  ]
}
```

---

### 3. 性能評価

学習後は必ず性能評価を実施:

```python
# テストクエリを用意
test_queries = [
    "設計速度80km/hの道路の最小曲線半径は？",  # 旧データ
    "新しく追加したデータに関する質問",  # 新データ
]

# 各クエリで回答精度を確認
for query in test_queries:
    # RAGで実行し、回答を評価
```

---

### 4. ロールバック戦略

問題発生時に備えて:

1. **前回のモデルを保持**
   ```bash
   # モデルをバックアップ
   docker exec ai-ft-container cp -r \
     outputs/continual_rardata_training_20251208_141953 \
     outputs/backup/
   ```

2. **Base Modelを前世代に変更**
   - UI で Base Model を1つ前のバージョンに戻す

---

## 📚 関連ドキュメント

| ドキュメント | 用途 | 対象者 |
|------------|------|-------|
| [rardata_ui_integrated_guide.md](./rardata_ui_integrated_guide.md) | **UI統合版（本ガイド）** | 全員 |
| [rardata_operations_summary.md](./rardata_operations_summary.md) | 全体フロー | 開発者 |
| [rardata_update_guide.md](./rardata_update_guide.md) | CLI詳細手順 | 開発者 |
| [rardata_current_status.md](./rardata_current_status.md) | モデル使用方法 | ユーザー |
| [rag_ui_usage_guide.md](./rag_ui_usage_guide.md) | RAG UI使用方法 | エンドユーザー |

---

## 🔧 関連ツール

| ツール | 用途 |
|-------|------|
| `scripts/check_rar_data.py` | データ品質チェック |
| `scripts/update_rardata_model.py` | CLI版継続学習 |
| 継続学習UI（http://localhost:8050/） | Web UI版継続学習 |

---

## 🎓 クイックリファレンス

### UI アクセス
```
http://localhost:8050/ → Continual Learning タブ
```

### 必須設定
```
✅ Use Previous Fisher Matrix: ON
✅ Base Model: 前回の学習済みモデル
✅ EWC Lambda: 5000（標準）
```

### データチェック
```bash
python3 scripts/check_rar_data.py RARdata.json
```

### 学習開始
```
1. Dataset をアップロード
2. 設定を入力
3. Start Training をクリック
```

### 結果確認
```
Tasks タブ → 完了ステータス確認
RAG タブ → テストクエリ実行
```

---

**作成日**: 2025年12月9日 00:30
**対象**: UI統合版のRARdata.json更新・学習操作
**次回更新**: UI機能追加時、または新しい運用パターン確立時
