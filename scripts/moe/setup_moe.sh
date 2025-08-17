#!/bin/bash

# MoE (Mixture of Experts) Setup Script for Civil Engineering Domain
# 土木・建設分野特化MoEモデルのセットアップスクリプト
# Docker環境対応版

set -e

echo "==========================================="
echo "MoE Setup for AI_FT_7 Project"
echo "土木・建設分野特化モデルのセットアップ"
echo "==========================================="

# 環境の判定（Dockerコンテナ内かどうか）
if [ -f /.dockerenv ]; then
    echo "Docker環境を検出しました"
    PROJECT_ROOT="/workspace"
    IN_DOCKER=true
else
    echo "ホスト環境を検出しました"
    PROJECT_ROOT="/home/kjifu/AI_FT_7"
    IN_DOCKER=false
fi

cd $PROJECT_ROOT

# Docker環境の場合、コンテナ内で実行
if [ "$IN_DOCKER" = false ]; then
    echo "Dockerコンテナ内でセットアップを実行します..."
    
    # コンテナが起動しているか確認
    if ! docker ps | grep -q ai-ft-container; then
        echo "Dockerコンテナを起動中..."
        cd docker && docker-compose up -d && cd ..
        sleep 5
    fi
    
    # Dockerコンテナ内でこのスクリプトを実行
    docker exec ai-ft-container bash /workspace/scripts/moe/setup_moe.sh
    exit 0
fi

# ここからはDockerコンテナ内での処理
echo "コンテナ内でのセットアップを開始..."

# 必要なパッケージのインストール
echo "必要なパッケージをインストール中..."
pip install einops scipy tensorboardX --quiet

# ディレクトリ構造の作成
echo "ディレクトリ構造を作成中..."
mkdir -p src/moe
mkdir -p data/civil_engineering/{train,val,test}
mkdir -p outputs/moe_civil
mkdir -p checkpoints/moe_civil
mkdir -p logs/moe

# MoEモジュールファイルの確認
echo "MoEモジュールの状態を確認中..."
if [ -f "src/moe/__init__.py" ]; then
    echo "✓ MoEモジュールが検出されました"
else
    echo "✗ MoEモジュールが見つかりません"
    echo "  手動でファイルを配置してください："
    echo "  - src/moe/moe_architecture.py"
    echo "  - src/moe/moe_training.py"
    echo "  - src/moe/data_preparation.py"
fi

# GPUの確認
echo ""
echo "GPU情報："
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# メモリ情報
echo ""
echo "システムメモリ："
free -h | grep "Mem:"

# Python環境の確認
echo ""
echo "Python環境："
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# 設定ファイルの作成
echo ""
echo "MoE設定ファイルを作成中..."
mkdir -p configs
cat > configs/moe_config.yaml << EOF
# MoE Configuration for Civil Engineering Domain
model:
  base_model: "cyberagent/calm3-22b-chat"
  num_experts: 8
  num_experts_per_tok: 2
  hidden_size: 4096
  expert_capacity_factor: 1.25
  
training:
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-5
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 2048
  mixed_precision: "bf16"
  gradient_checkpointing: true
  
experts:
  - name: "structural_design"
    weight: 1.0
    keywords: ["構造", "梁", "柱", "基礎", "耐震"]
  - name: "road_design"
    weight: 0.95
    keywords: ["道路", "舗装", "線形", "勾配", "交差点"]
  - name: "geotechnical"
    weight: 0.9
    keywords: ["地盤", "土質", "支持力", "沈下", "液状化"]
  - name: "hydraulics"
    weight: 0.85
    keywords: ["排水", "流量", "管渠", "ポンプ", "洪水"]
  - name: "materials"
    weight: 0.9
    keywords: ["コンクリート", "鋼材", "アスファルト", "強度"]
  - name: "construction_management"
    weight: 0.8
    keywords: ["工程", "安全", "品質", "施工", "管理"]
  - name: "regulations"
    weight: 1.0
    keywords: ["基準", "法規", "JIS", "道路構造令", "建築基準法"]
  - name: "environmental"
    weight: 0.75
    keywords: ["環境", "騒音", "振動", "廃棄物", "維持"]

paths:
  data_dir: "./data/civil_engineering"
  output_dir: "./outputs/moe_civil"
  checkpoint_dir: "./checkpoints/moe_civil"
  log_dir: "./logs/moe"
EOF

echo "✓ MoE設定ファイルが作成されました: configs/moe_config.yaml"

# テストスクリプトの作成
echo ""
echo "テストスクリプトを作成中..."
cat > scripts/moe/test_moe_setup.py << 'EOF'
#!/usr/bin/env python3
"""
MoE Setup Test Script
セットアップの確認用スクリプト
"""

import sys
import os
sys.path.append('/home/kjifu/AI_FT_7')

def test_imports():
    """必要なモジュールのインポートテスト"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import einops
        print("✓ einops")
        
        import scipy
        print("✓ scipy")
        
        import tensorboardX
        print("✓ tensorboardX")
        
        # MoEモジュールのテスト（存在する場合）
        try:
            from src.moe import MoEConfig
            print("✓ MoEモジュール")
        except ImportError:
            print("! MoEモジュールは未配置です")
        
        return True
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False

def test_gpu():
    """GPU利用可能性のテスト"""
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU利用可能: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("✗ GPUが利用できません")
        return False

def test_memory():
    """メモリ容量のテスト"""
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"✓ GPU {i} メモリ: {mem_total:.1f}GB (使用中: {mem_allocated:.1f}GB)")
        return True
    return False

def main():
    print("=" * 50)
    print("MoEセットアップテスト")
    print("=" * 50)
    
    print("\n1. モジュールインポートテスト")
    test_imports()
    
    print("\n2. GPU利用可能性テスト")
    test_gpu()
    
    print("\n3. メモリ容量テスト")
    test_memory()
    
    print("\n" + "=" * 50)
    print("テスト完了")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/moe/test_moe_setup.py

# セットアップテストの実行
echo ""
echo "セットアップテストを実行中..."
python scripts/moe/test_moe_setup.py

echo ""
echo "==========================================="
echo "MoEセットアップが完了しました！"
echo "==========================================="
echo ""
echo "次のステップ："
echo "1. MoEモジュールファイルを配置："
echo "   - moe_architecture.py → src/moe/"
echo "   - moe_training.py → src/moe/"
echo "   - data_preparation.py → src/moe/"
echo ""
echo "2. データ準備スクリプトの実行："
echo "   python scripts/moe/prepare_data.py"
echo ""
echo "3. トレーニングの開始："
echo "   bash scripts/moe/train_moe.sh"
echo ""
echo "詳細は IMPLEMENTATION_GUIDE.md を参照してください。"
