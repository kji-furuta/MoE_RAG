#!/bin/bash

# AI_FT_7 開発環境セットアップスクリプト

echo "AI_FT_7 開発環境のセットアップを開始します..."

# 1. 現在のディレクトリを確認
echo "現在のディレクトリ: $(pwd)"

# 2. Python仮想環境の作成
echo "Python仮想環境を作成しています..."
python3 -m venv venv
source venv/bin/activate

# 3. pipのアップグレード
echo "pipをアップグレードしています..."
pip install --upgrade pip

# 4. 基本パッケージのインストール
echo "基本パッケージをインストールしています..."
pip install -r requirements.txt

# 5. RAGパッケージのインストール（オプション）
read -p "RAGシステムの依存関係もインストールしますか？ (y/n): " install_rag
if [ "$install_rag" = "y" ]; then
    echo "RAGパッケージをインストールしています..."
    pip install -r requirements_rag.txt
fi

# 6. 必要なディレクトリの作成（既に存在する場合はスキップ）
echo "必要なディレクトリを作成しています..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/rag_documents
mkdir -p outputs
mkdir -p models
mkdir -p logs
mkdir -p temp_uploads
mkdir -p qdrant_data

# 7. Git初期化
if [ ! -d ".git" ]; then
    echo "Gitリポジトリを初期化しています..."
    git init
    git add .
    git commit -m "Initial commit: AI_FT_7 setup"
else
    echo "Gitリポジトリは既に初期化されています。"
fi

# 8. 環境変数ファイルのテンプレート作成
if [ ! -f ".env" ]; then
    echo "環境変数ファイルのテンプレートを作成しています..."
    cat > .env << EOF
# AI_FT_7 環境変数設定

# Hugging Face Token
HF_TOKEN=your_huggingface_token_here

# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key_here

# OpenAI API Key (RAGで使用する場合)
OPENAI_API_KEY=your_openai_api_key_here

# Jupyter Lab Token
JUPYTER_TOKEN=your_secure_token_here
EOF
    echo ".envファイルが作成されました。必要に応じてトークンを設定してください。"
fi

# 9. Dockerが利用可能か確認
if command -v docker &> /dev/null; then
    echo "Dockerが利用可能です。"
    read -p "Docker環境をビルドしますか？ (y/n): " build_docker
    if [ "$build_docker" = "y" ]; then
        echo "Docker環境をビルドしています..."
        cd docker
        docker-compose build
        cd ..
        echo "Docker環境のビルドが完了しました。"
    fi
else
    echo "Dockerがインストールされていません。手動でインストールしてください。"
fi

# 10. Ollamaが利用可能か確認
if command -v ollama &> /dev/null; then
    echo "Ollamaが利用可能です。"
else
    echo "Ollamaがインストールされていません。"
    read -p "Ollamaをインストールしますか？ (y/n): " install_ollama
    if [ "$install_ollama" = "y" ]; then
        echo "Ollamaをインストールしています..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "Ollamaのインストールが完了しました。"
    fi
fi

echo ""
echo "=========================================="
echo "セットアップが完了しました！"
echo "=========================================="
echo ""
echo "次のステップ："
echo "1. .envファイルを編集して必要なトークンを設定"
echo "2. Docker環境を起動: cd docker && docker-compose up -d"
echo "3. Webインターフェースにアクセス: http://localhost:8050"
echo ""
echo "詳細な使用方法はREADME.mdを参照してください。"