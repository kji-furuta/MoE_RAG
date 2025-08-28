#!/bin/bash

# メモリ管理改善の変更をロールバック

echo "================================"
echo "メモリ管理改善の変更をロールバック"
echo "================================"

cd /home/kjifu/MoE_RAG || exit 1

# 1. 追加したファイルを削除
echo "1. 追加したファイルを削除中..."

# 新規作成したファイルを削除
rm -f src/core/memory_manager.py
rm -f src/core/quantization_manager.py
rm -f src/core/docker_memory_patch.py
rm -f app/memory_optimized_loader_v2.py
rm -f app/main_unified_docker.py
rm -f app/main_unified_lightweight.py
rm -f scripts/migrate_memory_management.py
rm -f scripts/test_memory_management.py
rm -f scripts/start_web_interface_docker.sh
rm -f scripts/diagnose_docker_resources.sh
rm -f scripts/check_docker_resources.sh
rm -f scripts/start_highspec.sh
rm -f docker/entrypoint.sh
rm -f docker/entrypoint_lightweight.sh
rm -f docker/docker-compose-highspec.yml
rm -f docker/wslconfig.example
rm -f src/rag/config/lightweight_config.py
rm -f MEMORY_MANAGEMENT_IMPROVEMENT.md

# core ディレクトリが空なら削除
if [ -d src/core ] && [ -z "$(ls -A src/core)" ]; then
    rmdir src/core
fi

echo "✓ 追加ファイルを削除しました"

# 2. 元のファイルに戻す（バックアップがある場合）
echo "2. 変更したファイルを元に戻す..."

# app/main_unified.py を元に戻す（環境変数設定を復元）
if [ -f app/backups/main_unified_*.py ]; then
    # 最新のバックアップを復元
    latest_backup=$(ls -t app/backups/main_unified_*.py | head -1)
    cp "$latest_backup" app/main_unified.py
    echo "✓ app/main_unified.py を復元しました"
fi

# docker-compose.yml を元に戻す
cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  ai-ft:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: ai-ft-rag:latest
    container_name: ai-ft-container
    hostname: ai-ft
    
    # GPU access - requires nvidia-docker2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # Environment variables
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - PYTHONPATH=/workspace
      - WANDB_API_KEY=${WANDB_API_KEY:-}
      - HF_TOKEN=${HF_TOKEN:-}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    
    # Port mappings
    ports:
      - "8888:8888"  # Jupyter Lab
      - "6006:6006"  # TensorBoard
      - "8050:8050"  # Web Interface (統合)
      - "8051:8051"  # RAG API (開発・テスト用)
      - "11434:11434"  # Ollama API
    
    # Volume mounts
    volumes:
      - ../src:/workspace/src
      - ../config:/workspace/config
      - ../scripts:/workspace/scripts
      - ../notebooks:/workspace/notebooks
      - ../data:/workspace/data
      - ../models:/workspace/models
      - ../tests:/workspace/tests
      - ../app:/workspace/app
      - ../templates:/workspace/templates
      - ../outputs:/workspace/outputs
      - ./logs:/workspace/logs
      - ~/.cache/huggingface:/home/ai-user/.cache/huggingface
      - ~/.wandb:/home/ai-user/.wandb
      # RAG専用ディレクトリ
      - ../temp_uploads:/workspace/temp_uploads
      - ../qdrant_data:/workspace/qdrant_data
      - ../docs:/workspace/docs
      - ../examples:/workspace/examples
      # RAGデータ永続化
      - ai_ft_rag_metadata:/workspace/metadata
      - ai_ft_rag_processed:/workspace/outputs/rag_index
      # Ollamaモデル永続化
      - ollama_models:/usr/share/ollama/.ollama/models
    
    # Working directory
    working_dir: /workspace
    
    # Keep container running
    tty: true
    stdin_open: true
    
    # Restart policy
    restart: unless-stopped
    
    # Shared memory size for DataLoader
    shm_size: '32gb'
    
    # Ulimits for better performance
    ulimits:
      memlock:
        soft: -1
        hard: -1
      stack:
        soft: 67108864
        hard: 67108864

  # TensorBoard service (optional separate container)
  tensorboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: ai-ft-tensorboard
    command: tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
    ports:
      - "6007:6006"  # Different port to avoid conflict
    volumes:
      - ./logs:/workspace/logs
    depends_on:
      - ai-ft
    profiles:
      - tensorboard

  # Jupyter Lab service (optional separate container)
  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: ai-ft-jupyter
    command: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    ports:
      - "8889:8888"  # Different port to avoid conflict
    volumes:
      - ../notebooks:/workspace/notebooks
      - ../data:/workspace/data
      - ../src:/workspace/src
    depends_on:
      - ai-ft
    profiles:
      - jupyter

  # Qdrant Vector Database Service
  qdrant:
    image: qdrant/qdrant:latest
    container_name: ai-ft-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__TELEMETRY_DISABLED=true
    restart: unless-stopped

# Networks
networks:
  default:
    name: ai-ft-network

# Volumes for persistent data
volumes:
  models_data:
    driver: local
  logs_data:
    driver: local
  qdrant_storage:
    driver: local
  ai_ft_rag_metadata:
    driver: local
  ai_ft_rag_processed:
    driver: local
  ollama_models:
    driver: local
EOF

echo "✓ docker-compose.yml を復元しました"

# 3. Dockerfileを元に戻す（エントリーポイントを削除）
# Dockerfileの最後の部分を元に戻す
sed -i '/# エントリーポイントスクリプトをコピー/,/CMD \[\]/d' docker/Dockerfile

# 元の設定を追加
cat >> docker/Dockerfile << 'EOF'

# 作業ユーザーを切り替え
USER ai-user

# ポートを公開
EXPOSE 8888 6006 8050 8051

# デフォルトコマンド（bashシェルを起動）
CMD ["/bin/bash"]
EOF

echo "✓ Dockerfile を復元しました"

# 4. 元の起動スクリプトを復元
cat > scripts/start_web_interface.sh << 'EOF'
#!/bin/bash

# AI_FT_3 統合Webインターフェース起動スクリプト
# 継続学習管理システムを含む全機能を起動

echo "🚀 AI_FT_3 統合Webインターフェースを起動中..."

# Ollamaサービスを起動（存在する場合）
if command -v ollama &> /dev/null; then
    echo "🤖 Ollamaサービスを起動中..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollamaサービスを起動しました (port 11434)"
    
    # Ollamaモデルの確認と自動ダウンロード
    echo "📦 Ollamaモデルを確認中..."
    if ! ollama list | grep -q "llama3.2:3b"; then
        echo "📥 llama3.2:3bモデルをダウンロード中..."
        ollama pull llama3.2:3b
        echo "✅ モデルのダウンロードが完了しました"
    else
        echo "✅ llama3.2:3bモデルが利用可能です"
    fi
fi

# 作業ディレクトリを設定
cd /workspace

# パーミッションをチェック・修正
echo "🔐 パーミッションをチェック中..."
if [ -f /workspace/scripts/setup_permissions.sh ]; then
    /workspace/scripts/setup_permissions.sh --check
    if [ $? -ne 0 ]; then
        echo "🔧 パーミッションを修正中..."
        if [ "$EUID" -eq 0 ]; then
            /workspace/scripts/setup_permissions.sh
        else
            sudo /workspace/scripts/setup_permissions.sh 2>/dev/null || /workspace/scripts/setup_permissions.sh
        fi
    fi
else
    echo "⚠️ パーミッション設定スクリプトが見つかりません"
    # フォールバック: 基本的なディレクトリ作成
    mkdir -p /workspace/outputs
    mkdir -p /workspace/data/continual_learning
    mkdir -p /workspace/logs
    mkdir -p /workspace/temp_uploads
fi

# 継続学習管理システムの初期化
echo "📋 継続学習管理システムを初期化中..."

# 継続学習用の設定ファイルを作成
cat > /workspace/config/continual_learning_config.yaml << 'EOFCONFIG'
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
EOFCONFIG

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
EOF

chmod +x scripts/start_web_interface.sh
echo "✓ scripts/start_web_interface.sh を復元しました"

# 5. app/memory_optimized_loader.py の元の環境変数設定を復元
if [ -f app/memory_optimized_loader.py ]; then
    # CUDA_LAUNCH_BLOCKINGを1に戻す（元の設定）
    sed -i 's/os.environ\["CUDA_LAUNCH_BLOCKING"\] = "0"/os.environ["CUDA_LAUNCH_BLOCKING"] = "1"/g' app/memory_optimized_loader.py
    echo "✓ app/memory_optimized_loader.py の環境変数を復元"
fi

echo ""
echo "================================"
echo "ロールバック完了"
echo "================================"
echo ""
echo "次のコマンドでDockerを再起動してください："
echo ""
echo "cd docker"
echo "docker-compose down"
echo "docker-compose build"
echo "docker-compose up -d"
echo ""
echo "その後、以下のコマンドで起動："
echo "docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh"
echo ""
