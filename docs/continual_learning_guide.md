# 継続学習システムガイド

## 概要
MoE_RAGの継続学習システムは、EWC（Elastic Weight Consolidation）を使用して、破滅的忘却を防ぎながら複数のタスクを順次学習できる機能を提供します。

## 修正済みの問題（2025年9月19日）

### 1. APIエンドポイントの修正
- **問題**: `/api/continual-learning/tasks`が404エラー
- **原因**: ルーターのプレフィックス重複
- **解決**: `continual_learning_ui.py`でプレフィックスを削除

### 2. オフロードディレクトリエラーの修正
- **問題**: 大規模モデルで「We need an `offload_dir`」エラー
- **原因**: device_mapでCPUオフロードが必要なのにディレクトリ未指定
- **解決**:
  - 自動的にテンポラリディレクトリを作成
  - `offload_folder`パラメータを追加

### 3. メモリ管理の改善
- **32B/22Bモデル用の設定**:
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
  ```
- **メモリ配分**: GPU 24GB + CPU 40GB

### 4. 量子化モデルのトレーニング問題
- **問題**: 量子化モデルでファインチューニング不可
- **推奨**: LoRAまたはQLoRAを使用

### 5. ヘルパークラスによる改善（Codex MCP推奨）
- **ContinualLearningHelper導入**:
  - オフロードディレクトリの適切な管理とクリーンアップ
  - 環境変数の保持と復元
  - 包括的な量子化検出（8bit, 4bit, GPTQ, AWQ, EXL2, GGUF）
  - マルチGPUメモリマッピングサポート
  - 大規模モデル用のbfloat16最適化

## 使用方法

### Web UI経由
```
http://localhost:8050/continual
```

### API経由

#### 1. データセット準備（JSONL形式）
```jsonl
{"text": "道路設計速度80km/hの最小曲線半径は280mです。"}
{"text": "縦断勾配の最大値は設計速度により決定されます。"}
```

#### 2. 継続学習の開始
```bash
curl -X POST http://localhost:8050/api/continual-learning/start \
  -F 'dataset=@your_dataset.jsonl' \
  -F 'config={
    "base_model": "deepseek-ai/deepseek-llm-7b-base",
    "task_name": "road_design",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "use_memory_efficient": true,
    "use_previous_tasks": true,
    "ewc_lambda": 5000.0
  }'
```

#### 3. タスク状態の確認
```bash
# タスク一覧
curl http://localhost:8050/api/continual-learning/tasks

# 特定タスクの状態
curl http://localhost:8050/api/continual-learning/status/{task_id}
```

## 推奨設定

### 小規模モデル（7B）
```json
{
  "batch_size": 4,
  "learning_rate": 2e-5,
  "epochs": 3,
  "use_memory_efficient": false
}
```

### 大規模モデル（32B/22B）
```json
{
  "batch_size": 1,
  "learning_rate": 1e-5,
  "epochs": 1,
  "use_memory_efficient": true,
  "use_lora": true,  // 推奨
  "lora_r": 8,
  "lora_alpha": 16
}
```

## EWC（Elastic Weight Consolidation）の仕組み

EWCは、以前のタスクで重要だったパラメータの変更にペナルティを課すことで、破滅的忘却を防ぎます。

### Fisher Information Matrix
各タスク完了後に計算され、`outputs/ewc_data/`に保存されます：
- `fisher_task_{task_name}.pt`: Fisher行列
- `task_history.json`: タスク履歴

### EWCラムダ値の調整
- **高い値（5000-10000）**: 前のタスクをより強く保持
- **低い値（100-1000）**: 新しいタスクへの適応を優先

## トラブルシューティング

### メモリ不足エラー
```bash
# GPUメモリ確認
nvidia-smi

# 対策
1. batch_sizeを減らす
2. LoRA/QLoRAを使用
3. gradient_checkpointingを有効化
```

### モデルロードエラー
```bash
# Hugging Faceトークン設定
export HF_TOKEN=your_token_here

# キャッシュクリア
rm -rf ~/.cache/huggingface/
```

### タスク失敗時の対処
```bash
# ログ確認
docker logs ai-ft-container --tail 100 | grep ERROR

# タスク詳細確認
curl http://localhost:8050/api/continual-learning/status/{task_id}
```

## 9月18日機能の復元状況

### ✅ 復元完了
- 継続学習基本機能
- APIエンドポイント
- メモリ管理改善
- タスクスケジューラー

### ⚠️ 制限事項
- 32Bモデルは非常に大きなGPUメモリが必要（40GB以上）
- 量子化モデルでの直接トレーニングは不可（LoRA推奨）
- GGUF変換は別途llama-cpp-pythonのインストールが必要

## 関連ファイル

- UI実装: `/app/continual_learning/continual_learning_ui.py`
- パイプライン: `/src/training/continual_learning_pipeline.py`
- タスクスケジューラー: `/app/continual_learning/task_scheduler.py`
- EWCユーティリティ: `/src/training/ewc_utils.py`
- GGUF統合: `/src/training/gguf_integration.py`

## テストスクリプト

```bash
# 基本テスト
./scripts/test_continual_fix.sh

# 完全なワークフロー
python scripts/test_complete_workflow.py

# LoRAテスト
python scripts/test_continual_lora.py
```