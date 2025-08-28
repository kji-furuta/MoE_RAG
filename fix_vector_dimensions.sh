#!/bin/bash

# ベクトル次元統一の自動実行スクリプト

set -e

echo "========================================="
echo "🔧 RAGシステム ベクトル次元統一"  
echo "========================================="
echo ""
echo "対象設定:"
echo "  モデル: intfloat/multilingual-e5-large"
echo "  次元数: 1024"
echo "========================================="

# Pythonスクリプトを実行
cd /home/kjifu/MoE_RAG

echo ""
echo "📦 統一処理を開始..."
python3 scripts/rag_fixes/apply_dimension_fix.py

echo ""
echo "✅ 完了しました"
echo ""
echo "次のコマンドで検証を実行してください:"
echo "  python3 scripts/rag_fixes/verify_dimensions.py"
