#!/bin/bash

echo "================================"
echo "量子化の再試行（クリーンアップ付き）"
echo "================================"

# 既存の失敗した出力をクリーンアップ
echo "既存の出力をクリーンアップ中..."
rm -rf /workspace/outputs/ollama_conversion/merged_model_fp16_*

echo ""
echo "量子化スクリプトを実行..."
cd /workspace

# 環境変数設定
export PYTHONPATH=/workspace:$PYTHONPATH

# 直接実行
python /workspace/scripts/qlora_to_ollama.py

echo ""
echo "================================"
echo "完了"
echo "================================"