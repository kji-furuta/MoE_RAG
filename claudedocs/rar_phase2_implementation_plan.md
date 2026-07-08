# RAR Phase 2 実装計画 - 1,000件データ生成

**作成日**: 2025年12月8日
**Phase 1完了日**: 2025年12月8日
**目標**: 100件 → 1,000件へのスケールアップ

---

## 📊 Phase 1からの学習

### ✅ 成功した要素
1. **NotebookLM生成品質**: CoTスコア0.82、98.7%成功率
2. **システム統合**: citation_engine 100%互換
3. **学習安定性**: Loss 12.04で収束、±0.6%変動

### ⚠️ 改善が必要な要素
1. **JSON形式**: 複数行フォーマット → 正しい配列形式への自動変換が必要
2. **LaTeX記号**: エスケープ漏れ → 自動修正機能を実装
3. **引用検証**: 架空ファイル名の検出 → ホワイトリスト検証を自動化
4. **Fisher行列**: `StreamingTextDataset.__len__`未実装 → 修正が必要

---

## 🎯 Phase 2の目標

### 定量目標
| 指標 | Phase 1 | Phase 2目標 |
|------|---------|-------------|
| データ件数 | 100件 | **1,000件** |
| データ品質（成功率） | 98.7% | ≥95% |
| CoT品質スコア | 0.82 | ≥0.80 |
| Oracle文書比率 | 68.9% | 60-70% |
| 学習Loss安定性 | ±0.6% | ±5%以内 |

### 定性目標
1. **完全自動化**: データ生成 → 検証 → 修正 → 統合の全プロセス
2. **品質保証**: 自動検証で不合格データを排除
3. **効率化**: バッチ処理で段階的生成（20バッチ × 50件）
4. **再現性**: すべての処理がスクリプト化され、再実行可能

---

## 🔧 実装済みツール

### 1. NotebookLMプロンプトテンプレート v2.0
**ファイル**: [`data/rar_training/notebooklm_prompt_template_v2.md`](../data/rar_training/notebooklm_prompt_template_v2.md)

**Phase 1からの改善点**:
- ✅ 正しいJSON配列形式の明示
- ✅ LaTeX記号エスケープの指示
- ✅ 利用可能ファイル名のホワイトリスト提供
- ✅ Chain-of-Thought品質基準の明確化（番号付け、推論語）
- ✅ Oracle vs Distractor比率の指定（60-70% vs 30-40%）

**使用方法**:
```bash
# NotebookLMに以下を入力:
1. 対象PDF文書をアップロード
2. プロンプトテンプレートをコピー＆ペースト
3. 生成数とID範囲を調整
4. 生成実行
```

### 2. JSON自動検証・修正スクリプト
**ファイル**: [`scripts/rar/auto_validate_and_fix.py`](../scripts/rar/auto_validate_and_fix.py)

**機能**:
- ✅ JSON形式の自動修正（複数行 → 配列）
- ✅ LaTeX記号の自動エスケープ
- ✅ データ構造の検証
- ✅ Chain-of-Thought品質の評価
- ✅ 引用の長さチェック
- ✅ ファイル名のホワイトリスト検証

**使用方法**:
```bash
# 検証と自動修正
python scripts/rar/auto_validate_and_fix.py input.jsonl output_fixed.json

# 検証のみ（修正なし）
python scripts/rar/auto_validate_and_fix.py input.jsonl

# 出力例:
# ✅ 正しいJSON配列形式です
# ✓ LaTeX記号をエスケープしました
# ✓ 手動パーサーで149件のオブジェクトを抽出しました
# 📊 検証結果サマリー
# 総エントリー数: 149
# エラー数: 2
# 警告数: 15
```

### 3. バッチ処理システム
**ファイル**: [`scripts/rar/batch_processing_system.py`](../scripts/rar/batch_processing_system.py)

**機能**:
- ✅ バッチごとの生成指示書作成（ID範囲、重複防止）
- ✅ バッチファイルの自動検証
- ✅ 複数バッチの統合（重複ID検出）
- ✅ 進捗レポート生成

**使用方法**:
```bash
# 1. バッチ1の指示書生成
python scripts/rar/batch_processing_system.py generate 1

# 2. NotebookLMで生成 → batch_01_raw.json として保存

# 3. 検証
python scripts/rar/batch_processing_system.py validate batch_01_raw.json

# 4. 進捗レポート
python scripts/rar/batch_processing_system.py report 1

# 5. 全バッチ統合（最終ステップ）
python scripts/rar/batch_processing_system.py merge \
  data/rar_training/rar_1000.json \
  batch_01_validated.json batch_02_validated.json ... batch_20_validated.json
```

---

## 📋 Phase 2 実行ワークフロー

### 全体フロー（20バッチ × 50件）

```
┌─────────────────────────────────────────────────────────┐
│ Phase 2: 1,000件データ生成ワークフロー                   │
└─────────────────────────────────────────────────────────┘

[Batch 1-20の繰り返し]
  │
  ├─ 1. 指示書生成
  │   └─ python batch_processing_system.py generate <N>
  │       → batch_<N>_instructions.md
  │
  ├─ 2. NotebookLM生成
  │   ├─ PDF文書アップロード
  │   ├─ プロンプト実行（指示書参照）
  │   └─ 結果を batch_<N>_raw.json に保存
  │
  ├─ 3. 自動検証・修正
  │   └─ python auto_validate_and_fix.py \
  │       batch_<N>_raw.json batch_<N>_validated.json
  │       → エラー/警告レポート
  │
  ├─ 4. 手動確認（必要時）
  │   ├─ エラーがある場合: NotebookLMで再生成
  │   └─ 警告のみ: 内容確認後、次へ進行
  │
  └─ 5. 進捗レポート
      └─ python batch_processing_system.py report <N>
          → 達成率、残りバッチ数

[全バッチ完了後]
  │
  ├─ 6. 全バッチ統合
  │   └─ python batch_processing_system.py merge \
  │       rar_1000.json batch_*_validated.json
  │       → 1,000件統合ファイル
  │
  ├─ 7. 最終品質検証
  │   └─ python scripts/rar/validate_rar_json.py rar_1000.json
  │       → 全体統計、CoT品質スコア
  │
  └─ 8. 学習実験
      └─ python scripts/rar/run_rar_pilot_training.py
          （データパス: rar_1000.json）
```

### 詳細ステップ

#### Step 1: 環境準備
```bash
# 1. 作業ディレクトリ作成
mkdir -p data/rar_training/batches

# 2. スクリプトに実行権限付与
chmod +x scripts/rar/*.py

# 3. NotebookLM準備
# - 対象PDF文書をアップロード
# - notebooklm_prompt_template_v2.md を確認
```

#### Step 2-5: バッチ1生成（繰り返しパターン）
```bash
# Step 2: Batch 1指示書生成
python scripts/rar/batch_processing_system.py generate 1

# NotebookLMで実行:
# - 指示書の内容に従ってプロンプト実行
# - DES-001 ~ DES-050 を生成
# - 結果を data/rar_training/batches/batch_01_raw.json に保存

# Step 3: 自動検証・修正
python scripts/rar/auto_validate_and_fix.py \
  data/rar_training/batches/batch_01_raw.json \
  data/rar_training/batches/batch_01_validated.json

# Step 4: 検証結果確認
# エラー数が0であれば次へ
# エラーがある場合: 手動修正またはNotebookLMで再生成

# Step 5: 進捗レポート
python scripts/rar/batch_processing_system.py report 1
```

#### Step 6-8: 統合と最終検証
```bash
# Step 6: 全バッチ統合
python scripts/rar/batch_processing_system.py merge \
  data/rar_training/rar_1000.json \
  data/rar_training/batches/batch_*_validated.json

# Step 7: 最終品質検証
python scripts/rar/validate_rar_json.py data/rar_training/rar_1000.json

# Step 8: 学習実験（1,000件データ）
# （後述の学習パラメータ調整セクションを参照）
```

---

## 🎯 バッチスケジュール

### 推奨スケジュール（20バッチ）

| バッチ | ID範囲 | 生成数 | 累計 | 達成率 |
|--------|--------|--------|------|--------|
| Batch 01 | DES-001 ~ DES-050 | 50件 | 50 | 5% |
| Batch 02 | DES-051 ~ DES-100 | 50件 | 100 | 10% |
| Batch 03 | DES-101 ~ DES-150 | 50件 | 150 | 15% |
| Batch 04 | DES-151 ~ DES-200 | 50件 | 200 | 20% |
| Batch 05 | DES-201 ~ DES-250 | 50件 | 250 | 25% |
| ... | ... | ... | ... | ... |
| Batch 10 | DES-451 ~ DES-500 | 50件 | 500 | **50%** |
| ... | ... | ... | ... | ... |
| Batch 20 | DES-951 ~ DES-1000 | 50件 | **1000** | **100%** |

### 質問の多様性確保（バッチごとの焦点）

| バッチ範囲 | 質問焦点 | 例 |
|-----------|----------|-----|
| Batch 1-4 | 道路設計基準 | 設計速度、曲線半径、視距 |
| Batch 5-8 | 舗装設計 | 舗装厚さ、材料規格、交通量 |
| Batch 9-12 | 構造物設計 | 橋梁、トンネル、擁壁 |
| Batch 13-16 | 交通安全 | 防護柵、標識、照明 |
| Batch 17-20 | 施工・維持管理 | 施工方法、品質管理、点検 |

---

## 🔧 技術的課題の解決

### 課題1: Fisher行列計算エラー

**現状**: `StreamingTextDataset`に`__len__`メソッドがない

**修正方法**:
```python
# src/training/training_utils.py の StreamingTextDataset クラスに追加

class StreamingTextDataset:
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length

        # データ件数を事前にカウント
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self._length = len(self.data) if isinstance(self.data, list) else 0

    def __len__(self):
        """データセットの長さを返す"""
        return self._length
```

**検証方法**:
```bash
# 修正後、Phase 1のパイロットデータで再テスト
docker exec ai-ft-container python3 /workspace/scripts/rar/run_rar_pilot_training.py

# Fisher行列計算が成功することを確認
# → outputs/ewc_data/fisher_rar_pilot_phase1.pt が生成されるはず
```

### 課題2: 大規模データでのメモリ最適化

**1,000件学習時のメモリ管理**:
```yaml
# 学習パラメータ調整
batch_size: 1  # Phase 1と同じ
gradient_accumulation_steps: 32  # Phase 1の16から増加
max_seq_length: 256  # 変更なし
epochs: 5  # Phase 1の3から増加（より多くのデータで学習）
```

**期待効果**:
- 実効バッチサイズ: 1 × 32 = 32
- エポック数増加でデータ活用効率向上
- メモリ使用量は変わらず

---

## 📊 品質管理基準

### データ品質ゲート

| ゲート | 基準 | 判定 | アクション |
|--------|------|------|-----------|
| バッチ検証 | エラー数 = 0 | Pass/Fail | Fail時は再生成 |
| CoT品質 | 平均スコア ≥ 0.75 | Pass/Warning | Warning時は手動確認 |
| 引用検証 | 無効ファイル = 0 | Pass/Fail | Fail時は修正 |
| 最終統合 | 総件数 = 1,000 | Pass/Fail | 不足分を追加生成 |
| 重複チェック | 重複ID = 0 | Pass/Fail | 重複排除 |

### 学習品質ゲート

| 指標 | Phase 1実績 | Phase 2目標 | 判定基準 |
|------|-------------|-------------|----------|
| 学習Loss | 12.04 (安定) | 10.0-15.0 | ±30%以内 |
| Loss変動 | ±0.6% | ±5%以内 | 収束確認 |
| 学習時間 | 16分 (100件) | <180分 (1,000件) | 3時間以内 |
| モデルサイズ | 128MB | <200MB | LoRA効率維持 |

---

## 🚀 Phase 2実行開始

### クイックスタートガイド

```bash
# === 第1バッチ生成 ===

# 1. 指示書生成
cd /home/kjifu/MoE_RAG
python scripts/rar/batch_processing_system.py generate 1

# 2. NotebookLMで生成
# → data/rar_training/batches/batch_01_instructions.md を参照
# → 生成結果を batch_01_raw.json として保存

# 3. 検証
python scripts/rar/auto_validate_and_fix.py \
  data/rar_training/batches/batch_01_raw.json \
  data/rar_training/batches/batch_01_validated.json

# 4. 進捗確認
python scripts/rar/batch_processing_system.py report 2

# === バッチ2-20を繰り返し ===

# 最後に全バッチ統合
python scripts/rar/batch_processing_system.py merge \
  data/rar_training/rar_1000.json \
  data/rar_training/batches/batch_*_validated.json
```

---

## 📝 チェックリスト

### Phase 2開始前
- [x] NotebookLMプロンプトテンプレート v2.0 作成完了
- [x] JSON自動検証・修正スクリプト作成完了
- [x] バッチ処理システム作成完了
- [ ] `StreamingTextDataset.__len__` 修正
- [ ] 作業ディレクトリ `data/rar_training/batches/` 作成
- [ ] NotebookLMに対象PDF文書アップロード

### 各バッチ完了時
- [ ] バッチN指示書生成完了
- [ ] NotebookLMでDES-XXX ~ DES-YYY 生成完了
- [ ] 自動検証でエラー0件確認
- [ ] 警告内容を手動確認
- [ ] batch_N_validated.json 保存完了
- [ ] 進捗レポート生成完了

### Phase 2完了時
- [ ] 全20バッチ完了
- [ ] rar_1000.json 統合完了
- [ ] 最終品質検証（成功率≥95%、CoT≥0.80）
- [ ] 重複ID = 0件確認
- [ ] 学習実験実行（1,000件データ）
- [ ] Phase 2完了レポート作成

---

## 🎯 成功基準

### Phase 2完了判定

**必須条件（すべて満たす）**:
- ✅ 総データ件数: 1,000件
- ✅ データ品質: 成功率 ≥ 95%
- ✅ CoT品質: 平均スコア ≥ 0.80
- ✅ 重複ID: 0件
- ✅ 無効ファイル名: 0件

**推奨条件（80%以上満たす）**:
- ✅ Oracle文書比率: 60-70%
- ✅ 質問の多様性: 5カテゴリーすべてカバー
- ✅ 学習Loss: Phase 1の±30%以内
- ✅ 学習時間: 3時間以内

---

**作成者**: Claude (Sonnet 4.5)
**Phase 1完了**: 2025年12月8日
**Phase 2開始**: 2025年12月8日
