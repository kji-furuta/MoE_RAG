# メモリ効率的な継続学習（Phase 2）実装ガイド

## 概要
Phase 2では、大規模モデルでの継続学習を可能にするメモリ最適化機能を実装しました。

## 実装された機能

### 1. 効率的なFisher行列管理 (`EfficientFisherManager`)
- **ブロック単位の計算**: パラメータを1Mずつのブロックに分割して計算
- **HDF5圧縮保存**: gzip圧縮によりディスク使用量を削減
- **レイヤーごとのグループ化**: 局所性を保持してメモリアクセスを最適化
- **低ランク近似**: オプションで圧縮率を指定可能

### 2. 動的バッチサイズ調整 (`DynamicBatchSizeManager`)
- **自動バッチサイズ調整**: GPU使用率に基づいて動的に調整
- **OOM処理**: Out of Memoryエラーを検出して自動的にバッチサイズを削減
- **最適バッチサイズ推定**: モデルとサンプルから最適なバッチサイズを計算

### 3. メモリプロファイリング (`MemoryProfiler`)
- **リアルタイム監視**: CPU/GPUメモリ使用量を記録
- **可視化**: メモリ使用状況のグラフを生成
- **イベント記録**: 重要なタイミングでのメモリ状態を記録
- **レポート生成**: ピーク使用量や統計情報をレポート

### 4. メモリ最適化ユーティリティ (`MemoryOptimizer`)
- **勾配チェックポイント**: メモリ使用量を削減
- **積極的なキャッシュクリア**: 未使用メモリを解放
- **バッチサイズ推定**: 利用可能メモリから最適なバッチサイズを計算

## 使用方法

### 基本的な使用例

```bash
# メモリ効率的な継続学習の実行
python scripts/continual_learning/run_pipeline.py \
    --config configs/continual_tasks.yaml \
    --use-memory-efficient \
    --profile-memory \
    --dynamic-batch-size
```

### オプション説明

- `--use-memory-efficient`: 効率的なFisher行列計算を使用（デフォルト: True）
- `--profile-memory`: メモリ使用状況をプロファイリング
- `--dynamic-batch-size`: 動的バッチサイズ調整を有効化（デフォルト: True）

### 設定ファイルでの制御

```yaml
# configs/continual_tasks.yaml
continual_learning:
  ewc_config:
    fisher_computation:
      use_memory_efficient: true
      block_size: 1000000  # 1Mパラメータごとにブロック分割
      use_compression: true
      compression_ratio: 0.1  # 低ランク近似の圧縮率
```

## メモリ使用量の目安

| モデルサイズ | 通常のEWC | 効率的EWC | 削減率 |
|------------|-----------|-----------|--------|
| 7B | 56GB | 14GB | 75% |
| 22B | 176GB | 44GB | 75% |
| 70B | 560GB | 140GB | 75% |

※ Fisher行列の保存に必要なメモリ量の比較

## トラブルシューティング

### OOMエラーが発生する場合

1. バッチサイズを小さくする
```yaml
tasks:
  - name: "task1"
    batch_size: 1  # 最小値から開始
```

2. ブロックサイズを小さくする
```yaml
fisher_computation:
  block_size: 500000  # 500Kに削減
```

3. 環境変数を設定
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
```

### Fisher行列の読み込みが遅い場合

1. SSDに保存先を変更
2. 圧縮レベルを調整（1-9、デフォルト: 9）

### メモリプロファイルの確認

生成されたプロファイルは以下の場所に保存されます：
- `logs/memory_profile/memory_profile_YYYYMMDD_HHMMSS.json`
- `logs/memory_profile/memory_profile_YYYYMMDD_HHMMSS.png`

## テストの実行

```bash
# メモリ最適化機能のテスト
python scripts/test/test_memory_optimization.py
```

## 次のステップ（Phase 3）

Phase 3では以下を実装予定：
- Web UIからの継続学習実行
- 進捗モニタリング
- 評価レポートの自動生成

---

最終更新: 2025年8月4日
