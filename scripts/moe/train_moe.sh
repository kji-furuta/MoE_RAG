#!/bin/bash

# MoE Training Script for Civil Engineering Domain
# 土木・建設分野特化MoEモデルのトレーニング実行
# Docker環境対応版

set -e

echo "==========================================="
echo "MoE Training - AI_FT_7 Project"
echo "土木・建設分野特化モデルトレーニング"
echo "==========================================="

# 環境の判定（Dockerコンテナ内かどうか）
if [ -f /.dockerenv ]; then
    echo "Docker環境で実行中..."
    PROJECT_ROOT="/workspace"
else
    echo "ホスト環境で実行中..."
    PROJECT_ROOT="/home/kjifu/AI_FT_7"
fi

cd $PROJECT_ROOT

# デフォルト設定
MODE=${1:-"demo"}  # demo or full
EPOCHS=${2:-3}
BATCH_SIZE=${3:-2}

echo ""
echo "実行モード: $MODE"
echo "エポック数: $EPOCHS"
echo "バッチサイズ: $BATCH_SIZE"
echo ""

# データ準備の確認
if [ ! -d "data/civil_engineering/train" ]; then
    echo "データが見つかりません。データ準備を実行します..."
    python scripts/moe/prepare_data.py
fi

# トレーニング実行
if [ "$MODE" = "demo" ]; then
    echo "デモモードで実行します（小規模モデル）..."
    # ベースモデルの設定（環境変数から取得、デフォルトは小さいモデル）
    BASE_MODEL=${BASE_MODEL:-"cyberagent/open-calm-small"}
    echo "使用モデル: $BASE_MODEL"
    
    python scripts/moe/run_training.py \
        --demo_mode \
        --base_model "$BASE_MODEL" \
        --num_experts 8 \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --gradient_accumulation_steps 4 \
        --max_seq_length 128 \
        --output_dir ./outputs/moe_demo \
        --checkpoint_dir ./checkpoints/moe_demo
else
    echo "フルモードで実行します..."
    # ベースモデルの設定（環境変数から取得、デフォルトは小さいモデル）
    BASE_MODEL=${BASE_MODEL:-"cyberagent/open-calm-small"}
    echo "使用モデル: $BASE_MODEL"
    
    python scripts/moe/run_training.py \
        --base_model "$BASE_MODEL" \
        --num_experts 8 \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-5 \
        --use_mixed_precision \
        --gradient_checkpointing \
        --output_dir ./outputs/moe_civil \
        --checkpoint_dir ./checkpoints/moe_civil
fi

echo ""
echo "==========================================="
echo "トレーニング完了"
echo "==========================================="
