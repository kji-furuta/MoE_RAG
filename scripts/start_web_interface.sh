#!/bin/bash

# AI_FT_3 統合Webインターフェース起動スクリプト
# 継続学習管理システムを含む全機能を起動

echo "🚀 AI_FT_3 統合Webインターフェースを起動中..."

# 作業ディレクトリを設定
cd /workspace

# 必要なディレクトリを作成
mkdir -p /workspace/outputs
mkdir -p /workspace/data/continual_learning
mkdir -p /workspace/logs

# 継続学習管理システムの初期化
echo "📋 継続学習管理システムを初期化中..."

# 継続学習用の設定ファイルを作成
cat > /workspace/config/continual_learning_config.yaml << EOF
# 継続学習管理システム設定
continual_learning:
  enabled: true
  base_models:
    - cyberagent/calm3-22b-chat
    - cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
    - Qwen/Qwen2.5-14B-Instruct
    - Qwen/Qwen2.5-32B-Instruct
  
  ewc_settings:
    default_lambda: 5000
    fisher_computation_batches: 100
    use_efficient_storage: true
  
  training_settings:
    default_epochs: 3
    default_learning_rate: 2e-5
    default_batch_size: 4
    max_sequence_length: 512

# モデル管理設定
model_management:
  auto_detect_finetuned: true
  include_base_models: true
  cache_enabled: true
EOF

echo "✅ 継続学習管理システムの設定を完了"

# Webサーバーを起動
echo "🌐 統合Webサーバーを起動中..."
echo "📊 利用可能な機能:"
echo "  - メインダッシュボード: http://localhost:8050/"
echo "  - ファインチューニング: http://localhost:8050/finetune"
echo "  - 継続学習管理: http://localhost:8050/continual"
echo "  - RAGシステム: http://localhost:8050/rag"
echo "  - モデル管理: http://localhost:8050/models"

# 統合Webサーバーを起動
exec python3 -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload