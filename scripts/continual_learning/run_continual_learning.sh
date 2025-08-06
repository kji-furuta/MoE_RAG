#!/bin/bash
# run_continual_learning.sh
# 継続学習パイプラインの実行スクリプト

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# デフォルト値の設定
MODEL_BASE="${1:-outputs/フルファインチューニング_20250803_213130}"
TASK_CONFIG="${2:-configs/continual_tasks.yaml}"
EWC_LAMBDA="${3:-5000}"

echo "=== 継続学習パイプライン開始 ==="
echo "プロジェクトルート: $PROJECT_ROOT"
echo "ベースモデル: $MODEL_BASE"
echo "タスク設定: $TASK_CONFIG"
echo "EWCラムダ: $EWC_LAMBDA"
echo ""

# ログディレクトリの作成
mkdir -p logs/continual_learning
mkdir -p outputs/ewc_data/fisher_matrices
mkdir -p outputs/ewc_data/reference_models
mkdir -p outputs/ewc_data/evaluation
mkdir -p data/continual

# Python環境のアクティベート（仮想環境を使用している場合）
if [ -d "ai_ft_env" ]; then
    echo "Python仮想環境をアクティベート中..."
    source ai_ft_env/bin/activate
fi

# GPUメモリのクリア
if command -v nvidia-smi &> /dev/null; then
    echo "GPUメモリをクリア中..."
    nvidia-smi
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

# 環境変数の設定
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1"  # 使用するGPUを指定
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # メモリ断片化を防ぐ

# 継続学習の実行
echo ""
echo "継続学習パイプラインを実行中..."
python -u scripts/continual_learning/run_pipeline.py \
    --config "$TASK_CONFIG" \
    --base-model "$MODEL_BASE" \
    --ewc-lambda "$EWC_LAMBDA" \
    2>&1 | tee "logs/continual_learning/run_$(date +%Y%m%d_%H%M%S).log"

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 継続学習完了 ==="
    echo "結果は以下のディレクトリに保存されました:"
    echo "- Fisher行列: outputs/ewc_data/fisher_matrices/"
    echo "- モデル: outputs/continual_*/"
    echo "- ログ: logs/continual_learning/"
else
    echo ""
    echo "=== エラーが発生しました ==="
    echo "ログファイルを確認してください。"
fi
