#!/bin/bash
# scripts/run_continual_learning.sh

# 継続学習パイプラインの自動実行スクリプト

set -e  # エラーが発生したら停止

# デフォルト値
MODEL_BASE="${1:-outputs/フルファインチューニング_20250803_213130}"
TASK_CONFIG="${2:-configs/continual_tasks.yaml}"

echo "=== 継続学習パイプライン開始 ==="
echo "ベースモデル: $MODEL_BASE"
echo "タスク設定: $TASK_CONFIG"
echo ""

# Python環境のアクティベート（必要に応じて）
if [ -f "ai_ft_env/bin/activate" ]; then
    echo "仮想環境をアクティベート中..."
    source ai_ft_env/bin/activate
fi

# ログディレクトリの作成
LOG_DIR="logs/continual_learning/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "ログディレクトリ: $LOG_DIR"

# 継続学習の実行
python -m scripts.continual_learning.run_pipeline \
    --base_model "$MODEL_BASE" \
    --task_config "$TASK_CONFIG" \
    --ewc_lambda 5000 \
    --save_all_checkpoints \
    --use_memory_efficient_fisher \
    --monitor_forgetting \
    --log_dir "$LOG_DIR" \
    --cleanup_memory

# 終了コードの確認
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== 継続学習完了 ==="
    echo "結果は以下に保存されました:"
    echo "- ログ: $LOG_DIR"
    echo "- モデル: outputs/continual_*"
    echo "- 評価レポート: outputs/ewc_data/evaluation/"
else
    echo ""
    echo "=== エラーが発生しました (終了コード: $EXIT_CODE) ==="
    echo "詳細はログを確認してください: $LOG_DIR"
fi

exit $EXIT_CODE
