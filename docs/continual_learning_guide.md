# 継続学習システム 操作マニュアル

## 🎯 継続学習システムとは

MoE_RAGの継続学習システムは、**EWC（Elastic Weight Consolidation）** を使用して、AIモデルが新しい知識を学習しながら既存の知識を保持できる革新的な機能です。複数のタスクを順番に学習させても、以前に学習した内容を忘れることなく、新しい能力を追加できます。

## 📢 最新の更新 (2025年9月25日)

### ✅ システム完全稼働確認
- **継続学習システム**: EWCベースの継続学習が完全動作確認済み
- **Fisher情報行列**: 正常に計算・保存され、破滅的忘却を防止
- **タスク管理**: 複数タスクの状態管理とモデル保存が正常動作
- **Web UI統合**: `/continual` インターフェースで全機能利用可能
- **統合テスト**: 全ての継続学習関連テストが正常通過

## 🚀 クイックスタート

### 1. システムへアクセス
```
http://localhost:8050/continual
```

### 2. 画面構成

#### 📊 メインダッシュボード
- **タスク一覧**: 実行中・完了済みタスクの状態表示
- **モデル選択**: ベースモデルまたは学習済みモデルの選択
- **新規タスク作成**: 継続学習タスクの設定と開始

## 📖 使い方ガイド

### ステップ1: ベースモデルの選択

1. **モデル選択ドロップダウン**から使用するモデルを選択
   - 推奨: `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese`
   - 軽量版: `deepseek-ai/deepseek-llm-7b-base`

2. **モデル更新ボタン**
   - 新しく学習したモデルがリストに表示されない場合にクリック
   - `outputs/`ディレクトリをスキャンして最新のモデルリストを取得

### ステップ2: データセットの準備

#### JSONL形式のデータセット作成
```jsonl
{"text": "道路設計速度80km/hの最小曲線半径は280mです。"}
{"text": "縦断勾配の最大値は設計速度により決定されます。"}
{"text": "歩道の有効幅員は原則として2.0m以上とします。"}
```

#### アップロード方法
1. **「ファイルを選択」** ボタンをクリック
2. 準備したJSONLファイルを選択
3. ファイル名が表示されることを確認

### ステップ3: 学習パラメータの設定

#### 基本設定

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| **タスク名** | 学習タスクの識別名 | 例: `road_design_v2` |
| **エポック数** | 学習の反復回数 | 小規模: 3, 大規模: 1 |
| **バッチサイズ** | 一度に処理するデータ数 | 小規模: 4, 大規模: 1 |
| **学習率** | パラメータ更新の大きさ | 2e-5 |

#### EWC設定（継続学習）

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| **EWCラムダ** | 既存知識の保護強度 | 5000 |
| **前タスク使用** | 以前のタスクの知識を保持 | ✓ チェック |

### ステップ4: 学習の開始と監視

1. **「学習開始」ボタン**をクリック
2. **進捗状況**がリアルタイムで表示されます：
   - 🟡 実行中: 学習進行中
   - 🟢 完了: 正常終了
   - 🔴 エラー: 問題発生（ログを確認）

3. **進捗バー**で完了率を確認
   - 現在のステップ数/総ステップ数が表示
   - 推定残り時間の目安

### ステップ5: 学習結果の確認

#### モデルの保存場所
```
outputs/
├── continual_task_[タスク名]/    # 学習済みモデル
│   ├── pytorch_model.bin         # モデルの重み
│   ├── config.json               # 設定ファイル
│   └── tokenizer_config.json    # トークナイザー設定
└── ewc_data/
    ├── fisher_task_[タスク名].pt # Fisher情報行列
    └── task_history.json         # タスク履歴
```

## 🛠️ 詳細設定ガイド

### モデルサイズ別推奨設定

#### 🚀 小規模モデル（7B以下）
```json
{
  "batch_size": 4,
  "learning_rate": 2e-5,
  "epochs": 3,
  "use_memory_efficient": false,
  "gradient_accumulation_steps": 1
}
```

#### 💪 中規模モデル（14B-22B）
```json
{
  "batch_size": 2,
  "learning_rate": 1.5e-5,
  "epochs": 2,
  "use_memory_efficient": true,
  "gradient_accumulation_steps": 2,
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32
}
```

#### 🔥 大規模モデル（32B以上）
```json
{
  "batch_size": 1,
  "learning_rate": 1e-5,
  "epochs": 1,
  "use_memory_efficient": true,
  "gradient_accumulation_steps": 4,
  "use_lora": true,
  "lora_r": 8,
  "lora_alpha": 16,
  "quantization": "4bit"
}
```

### EWCラムダ値の調整ガイド

| ラムダ値 | 効果 | 使用場面 |
|---------|------|----------|
| **100-1000** | 低い保護 | 新タスクへの適応を優先 |
| **1000-5000** | 標準 | バランスの取れた学習 |
| **5000-10000** | 強い保護 | 既存知識を強く保持 |
| **10000以上** | 最大保護 | 既存知識の変更を最小限に |

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー
```
CUDA out of memory
```
**解決方法:**
- バッチサイズを1に減らす
- LoRA/QLoRAを有効にする
- gradient_checkpointingを有効化
- 量子化（4bit/8bit）を使用

#### 2. モデルロードエラー
```
Model not found
```
**解決方法:**
- Hugging Faceトークンを設定
- インターネット接続を確認
- モデル名の正確性を確認
- キャッシュをクリア: `rm -rf ~/.cache/huggingface/`

#### 3. 学習が進まない
**解決方法:**
- 学習率を調整（大きくする）
- データセットの品質を確認
- エポック数を増やす
- GPUが正しく認識されているか確認: `nvidia-smi`

#### 4. タスクが失敗する
**確認手順:**
```bash
# ログ確認
docker logs ai-ft-container --tail 100

# タスク詳細確認
curl http://localhost:8050/api/continual/task/{task_id}

# システム状態確認
curl http://localhost:8050/api/continual/tasks
```

## 📊 タスク履歴の管理

### タスク情報の確認
```bash
# 全タスク一覧
cat data/continual_learning/tasks_state.json | jq .

# 特定タスクの詳細
cat data/continual_learning/tasks_state.json | jq '.tasks["タスク名"]'
```

### Fisher行列の確認
```bash
# Fisher行列ファイル一覧
ls -la outputs/ewc_data/fisher_*.pt

# タスク履歴
cat outputs/ewc_data/task_history.json | jq .
```

## 🎯 ベストプラクティス

### 1. データセット準備
- **品質重視**: 少量でも高品質なデータを使用
- **バランス**: 各カテゴリのデータ量を均等に
- **形式統一**: JSONLフォーマットを厳守

### 2. 学習戦略
- **段階的学習**: 簡単なタスクから複雑なタスクへ
- **定期保存**: 重要なモデルは別途バックアップ
- **検証**: 各タスク後に性能評価を実施

### 3. リソース管理
- **GPU監視**: `nvidia-smi -l 1` でリアルタイム監視
- **ディスク容量**: モデル保存に十分な空き容量を確保
- **メモリ最適化**: 大規模モデルは量子化を活用

## 🔗 API利用ガイド（上級者向け）

### 継続学習の開始
```bash
curl -X POST http://localhost:8050/api/continual/train \
  -F 'dataset=@your_dataset.jsonl' \
  -F 'config={
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "my_task",
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "ewc_lambda": 5000
  }'
```

### タスク状態の確認
```bash
# 全タスク一覧
curl http://localhost:8050/api/continual/tasks

# 特定タスクの詳細
curl http://localhost:8050/api/continual/task/{task_id}
```

### モデルリストの更新
```bash
curl -X POST http://localhost:8050/api/continual/update-models
```

## 📝 関連ドキュメント

- [メインREADME](../README.md) - システム全体の概要
- [RAGシステムガイド](./rag_system_guide.md) - RAG機能の詳細
- [ファインチューニングガイド](./finetuning_guide.md) - 基本的なファインチューニング

## 🆘 サポート

問題が解決しない場合は、以下の情報を含めてissueを報告してください：
- エラーメッセージの全文
- 使用したモデル名とパラメータ
- データセットのサンプル（機密情報を除く）
- システム環境（GPU、メモリ、Docker版）

---

**最終更新**: 2025年9月25日
**バージョン**: 1.0.0
**ステータス**: ✅ 完全稼働中