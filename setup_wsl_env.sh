#!/bin/bash
# WSL環境設定スクリプト

echo "=== WSL環境設定 ==="
echo

# プロジェクトディレクトリを取得
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo "モデルディレクトリ: $MODELS_DIR"
echo

# modelsディレクトリを作成
if [ ! -d "$MODELS_DIR" ]; then
    mkdir -p "$MODELS_DIR"
    echo "✅ モデルディレクトリを作成しました: $MODELS_DIR"
else
    echo "✅ モデルディレクトリが既に存在します: $MODELS_DIR"
fi

# 環境変数を設定
export HF_HOME="$MODELS_DIR"
export TRANSFORMERS_CACHE="$MODELS_DIR"

echo "✅ 環境変数を設定しました:"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo

# .bashrcに追加するか確認
read -p ".bashrcに環境変数を追加しますか？ (y/n): " choice
if [[ $choice == "y" || $choice == "Y" ]]; then
    # 既存の設定を削除
    sed -i '/export HF_HOME=/d' ~/.bashrc
    sed -i '/export TRANSFORMERS_CACHE=/d' ~/.bashrc
    
    # 新しい設定を追加
    echo "export HF_HOME=\"$MODELS_DIR\"" >> ~/.bashrc
    echo "export TRANSFORMERS_CACHE=\"$MODELS_DIR\"" >> ~/.bashrc
    
    echo "✅ .bashrcに環境変数を追加しました"
    echo "変更を反映するには、新しいターミナルを開くか 'source ~/.bashrc' を実行してください"
else
    echo "環境変数は現在のセッションのみ有効です"
fi

echo
echo "=== 使用方法 ==="
echo "1. モデルをダウンロード:"
echo "   python3 download_models_wsl.py"
echo
echo "2. モデルの場所を確認:"
echo "   python3 check_model_locations.py"
echo
echo "3. 環境変数を確認:"
echo "   echo \$HF_HOME"
echo "   echo \$TRANSFORMERS_CACHE" 