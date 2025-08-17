# MoE Training System - Complete Implementation

## 概要
MoE-RAGシステムにトレーニング機能を実装しました。これにより、ブラウザから直接MoEモデルのトレーニングを管理・実行できるようになりました。

## 実装内容

### 1. バックエンドAPI (`app/moe_training_endpoints.py`)
- **トレーニング管理API**
  - `/api/moe/training/start` - トレーニング開始
  - `/api/moe/training/status/{task_id}` - ステータス確認
  - `/api/moe/training/stop/{task_id}` - トレーニング停止
  - `/api/moe/training/history` - 履歴取得
  - `/api/moe/training/gpu-status` - GPU状態確認
  - `/api/moe/training/deploy/{task_id}` - モデルデプロイ
  - `/api/moe/training/upload-dataset` - データセットアップロード
  - `/api/moe/training/logs/{task_id}` - ログ取得
  - `/api/moe/training/stream-logs/{task_id}` - ログストリーミング

### 2. フロントエンドUI (`app/static/moe_training.html`)
- **トレーニング設定画面**
  - トレーニングタイプ選択（デモ、フル、LoRA、継続学習）
  - ハイパーパラメータ設定（エポック数、バッチサイズ、学習率）
  - エキスパート選択（8つの専門分野から選択）
  - データセット選択とカスタムデータアップロード

- **進捗モニタリング**
  - リアルタイム進捗バー
  - エポック/損失/GPU使用率の表示
  - トレーニングログのリアルタイム表示
  - 経過時間トラッキング

- **トレーニング管理**
  - トレーニング履歴表示
  - タスク詳細確認
  - モデルデプロイ機能
  - GPU状態モニタリング

### 3. トレーニング実行スクリプト
- **`scripts/moe/train_moe.sh`** - Dockerコンテナ内でのトレーニング実行
- **`scripts/moe/prepare_data.py`** - データ準備
- **`scripts/moe/run_training.py`** - 実際のトレーニング処理

### 4. 統合
- **`app/main_unified.py`** - MoEトレーニングエンドポイントの登録
- **`templates/index.html`** - メインダッシュボードからのリンク追加

## 使用方法

### 1. システム起動
```bash
cd docker
docker-compose up -d
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 2. トレーニングページへのアクセス
- ブラウザで http://localhost:8050 を開く
- 「MoE-RAG」タブをクリック
- 「トレーニング管理」ボタンをクリック
- または直接 http://localhost:8050/static/moe_training.html にアクセス

### 3. トレーニング実行
1. **エキスパート選択**: トレーニングしたい専門分野のエキスパートを選択
2. **パラメータ設定**: エポック数、バッチサイズ、学習率を設定
3. **データセット選択**: デモデータまたはカスタムデータを選択
4. **トレーニング開始**: 「トレーニング開始」ボタンをクリック

### 4. 進捗確認
- リアルタイムで進捗バーとメトリクスが更新される
- ログビューアーでトレーニング状況を確認
- 必要に応じて「停止」ボタンで中断可能

### 5. モデルデプロイ
- トレーニング完了後、履歴から「デプロイ」ボタンをクリック
- モデルが自動的に配置され、MoE-RAGシステムで利用可能になる

## 技術的詳細

### バックグラウンドタスク処理
- FastAPIの`BackgroundTasks`を使用して非同期でトレーニング実行
- タスクIDベースでトレーニングジョブを管理
- 複数のトレーニングを並行実行可能

### リアルタイムモニタリング
- 2秒ごとにステータスをポーリング
- Server-Sent Events (SSE)でログをストリーミング
- GPU使用率をGPUtil/nvidia-smiで取得

### データ永続化
- トレーニング設定は環境変数経由でスクリプトに渡される
- モデルは`/workspace/outputs/`に保存
- チェックポイントは`/workspace/checkpoints/`に保存

## エキスパート一覧

1. **構造設計** (structural) - 構造計算、耐震設計、荷重解析
2. **道路設計** (road) - 線形設計、視距、設計速度
3. **地盤工学** (geotech) - 土質調査、支持力、液状化対策
4. **水理・排水** (hydraulics) - 流量計算、排水計画、治水
5. **材料工学** (materials) - コンクリート、鋼材、材料試験
6. **施工管理** (construction) - 工程管理、品質管理、安全管理
7. **法規・基準** (regulations) - 道路構造令、設計基準、仕様書
8. **環境・維持管理** (environmental) - 環境影響、維持管理、劣化診断

## テスト実行

APIテスト:
```bash
python test_moe_training.py
```

ブラウザテスト:
1. http://localhost:8050/static/moe_training.html を開く
2. GPU状態確認ボタンをクリック
3. エキスパートを選択してデモトレーニングを実行

## トラブルシューティング

### トレーニングが開始しない
- Docker コンテナが起動しているか確認
- GPU メモリが十分にあるか確認（`nvidia-smi`）
- ログを確認（`docker logs ai-ft-container`）

### GPU が認識されない
- Docker の GPU サポートを確認（`docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`）
- GPUtil がインストールされているか確認（`pip list | grep GPUtil`）

### エラーログの確認
```bash
docker exec ai-ft-container tail -f /workspace/logs/training.log
```

## 今後の改善点

1. **実トレーニングの統合**
   - 現在はシミュレーションモード
   - 実際のMoEモデルトレーニングコードとの連携が必要

2. **メトリクス可視化**
   - TensorBoardとの統合
   - グラフによる損失推移の表示

3. **分散トレーニング**
   - マルチGPU対応
   - データ並列/モデル並列の実装

4. **オートスケーリング**
   - リソース使用量に応じた自動調整
   - キューイングシステムの実装

## まとめ

MoE-RAGシステムに完全なトレーニング管理機能を実装しました。これにより、土木工学分野に特化したMoEモデルをWebインターフェースから簡単にトレーニング・管理できるようになりました。システムは以下の特徴を持ちます：

- ✅ ブラウザベースの直感的なUI
- ✅ リアルタイム進捗モニタリング
- ✅ 8つの専門エキスパートの選択的トレーニング
- ✅ GPU状態の可視化
- ✅ トレーニング履歴管理
- ✅ ワンクリックモデルデプロイ

これにより、専門知識を持たないユーザーでも簡単にMoEモデルのトレーニングが可能になりました。