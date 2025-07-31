#!/bin/bash
"""
Docker RAG統合ビルドスクリプト
RAG依存関係を含むDockerイメージをビルドし、統合テストを実行
"""

set -e  # エラー時に停止

echo "🐳 Docker RAG統合ビルド開始"
echo "=================================="

# スクリプトが実行されているディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 プロジェクトルート: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 必要ファイルの存在確認
echo "📋 必要ファイルの存在確認..."

required_files=(
    "requirements.txt"
    "requirements_rag.txt"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
    "app/main_unified.py"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file が見つかりません"
        exit 1
    fi
done

# ビルドオプションの設定
NO_CACHE=""
if [[ "$1" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
    echo "🔄 キャッシュを使わずにビルドします"
fi

# Dockerイメージビルド
echo ""
echo "🔨 Dockerイメージビルド中..."
echo "  - RAG依存関係をインストール"
echo "  - 統合Webインターフェースを設定"
echo "  - 必要なディレクトリを作成"
echo ""

docker build $NO_CACHE -f docker/Dockerfile -t ai-ft-rag:latest .

if [[ $? -eq 0 ]]; then
    echo "✅ Dockerイメージビルド成功"
else
    echo "❌ Dockerイメージビルド失敗"
    exit 1
fi

# イメージサイズ確認
echo ""
echo "📊 イメージサイズ:"
docker images ai-ft-rag:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Docker Composeサービス起動
echo ""
echo "🚀 Docker Composeサービス起動..."
cd docker
docker-compose up -d

if [[ $? -eq 0 ]]; then
    echo "✅ Docker Composeサービス起動成功"
else
    echo "❌ Docker Composeサービス起動失敗"
    exit 1
fi

# サービス状態確認
echo ""
echo "📊 サービス状態:"
docker-compose ps

# RAG統合テスト実行
echo ""
echo "🧪 RAG統合テスト実行..."
echo "コンテナ起動を待機中（10秒）..."
sleep 10

docker-compose exec -T ai-ft python test_docker_rag.py

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Docker RAG統合完了！"
    echo ""
    echo "📍 アクセス情報:"
    echo "  - 統合Webインターフェース: http://localhost:8050/"
    echo "    - ファインチューニング: http://localhost:8050/finetune"
    echo "    - RAG機能: http://localhost:8050/rag"
    echo "  - RAG API: http://localhost:8050/rag/*"
    echo "  - 開発用RAG API: http://localhost:8051/"
    echo "  - Jupyter Lab: http://localhost:8888/"
    echo "  - TensorBoard: http://localhost:6006/"
    echo ""
    echo "🔧 管理コマンド:"
    echo "  - ログ確認: docker-compose logs -f ai-ft"
    echo "  - コンテナ内アクセス: docker-compose exec ai-ft bash"
    echo "  - サービス停止: docker-compose down"
    echo ""
else
    echo ""
    echo "⚠️ RAG統合テストが一部失敗しました"
    echo "🔧 詳細ログを確認してください:"
    echo "  docker-compose logs ai-ft"
    echo ""
    echo "💡 トラブルシューティング:"
    echo "  1. docker-compose exec ai-ft bash でコンテナ内に入る"
    echo "  2. python test_docker_rag.py で再テスト"
    echo "  3. python app/main_unified.py で手動起動テスト"
fi

echo ""
echo "🏁 ビルドスクリプト完了"