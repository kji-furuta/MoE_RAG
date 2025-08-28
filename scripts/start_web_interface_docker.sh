#!/bin/bash

# AI_FT_3 統合Webインターフェース起動スクリプト（Docker対応版）
# 継続学習管理システムを含む全機能を起動

echo "🚀 AI_FT_3 統合Webインターフェースを起動中..."
echo "🐳 Docker環境を検出..."

# Docker環境の検出
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "✅ Docker環境で実行中"
    export DOCKER_CONTAINER=true
else
    echo "ℹ️ 通常環境で実行中"
fi

# Ollamaサービスを起動（存在する場合）
if command -v ollama &> /dev/null; then
    echo "🤖 Ollamaサービスを起動中..."
    # バックグラウンドでOllamaを起動
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    
    # Ollamaが起動したか確認
    if ps -p $OLLAMA_PID > /dev/null; then
        echo "✅ Ollamaサービスを起動しました (PID: $OLLAMA_PID, port 11434)"
    else
        echo "⚠️ Ollamaサービスの起動に失敗しました"
    fi
    
    # Ollamaモデルの確認
    echo "📦 Ollamaモデルを確認中..."
    if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
        echo "✅ llama3.2:3bモデルが利用可能です"
    else
        echo "📥 llama3.2:3bモデルをダウンロード中..."
        ollama pull llama3.2:3b 2>/dev/null || echo "⚠️ モデルのダウンロードをスキップ"
    fi
else
    echo "ℹ️ Ollamaがインストールされていません"
fi

# 作業ディレクトリを設定
cd /workspace || cd /home/kjifu/MoE_RAG || exit 1

# 必要なディレクトリを作成
echo "📁 必要なディレクトリを作成中..."
mkdir -p outputs data/continual_learning logs temp_uploads
mkdir -p data/uploaded config

# パーミッションの修正（Docker環境では必要な場合のみ）
if [ -f scripts/setup_permissions.sh ]; then
    echo "🔐 パーミッションを設定中..."
    if [ "$EUID" -eq 0 ]; then
        # ルートユーザーの場合
        bash scripts/setup_permissions.sh 2>/dev/null || true
    else
        # 一般ユーザーの場合
        bash scripts/setup_permissions.sh --check 2>/dev/null || true
    fi
fi

# 継続学習用の設定ファイルを作成（存在しない場合）
if [ ! -f config/continual_learning_config.yaml ]; then
    echo "📋 継続学習管理システムの設定を作成中..."
    cat > config/continual_learning_config.yaml << 'EOF'
# 継続学習管理システム設定
continual_learning:
  enabled: true
  base_models:
    - cyberagent/calm3-22b-chat
    - Qwen/Qwen2.5-14B-Instruct
  
  ewc_settings:
    default_lambda: 5000
    fisher_computation_batches: 100
    use_efficient_storage: true
  
  training_settings:
    default_epochs: 3
    default_learning_rate: 2e-5
    default_batch_size: 4
    max_sequence_length: 512

model_management:
  auto_detect_finetuned: true
  include_base_models: true
  cache_enabled: true
EOF
    echo "✅ 継続学習管理システムの設定を作成しました"
fi

# Pythonパスを設定
export PYTHONPATH=/workspace:$PYTHONPATH

# メモリ最適化設定（Docker環境用）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo ""
echo "🌐 統合Webサーバーを起動中..."
echo "📊 利用可能な機能:"
echo "  - メインダッシュボード: http://localhost:8050/"
echo "  - API ドキュメント: http://localhost:8050/docs"
echo "  - ファインチューニング: http://localhost:8050/static/moe_training.html"
echo "  - 継続学習管理: http://localhost:8050/static/continual_learning/index.html"
echo "  - RAGシステム: http://localhost:8050/static/moe_rag_ui.html"
echo ""

# 既存のプロセスを確認
if lsof -i:8050 > /dev/null 2>&1; then
    echo "⚠️ ポート8050は既に使用されています。既存のプロセスを終了します..."
    kill $(lsof -t -i:8050) 2>/dev/null || true
    sleep 2
fi

# Docker環境用のmain_unifiedを使用
if [ -f app/main_unified_docker.py ]; then
    echo "🐳 Docker対応版のサーバーを起動します..."
    exec python3 -m uvicorn app.main_unified_docker:app \
        --host 0.0.0.0 \
        --port 8050 \
        --workers 1 \
        --loop asyncio \
        --log-level info
else
    echo "📦 通常版のサーバーを起動します..."
    exec python3 -m uvicorn app.main_unified:app \
        --host 0.0.0.0 \
        --port 8050 \
        --reload \
        --log-level info
fi
