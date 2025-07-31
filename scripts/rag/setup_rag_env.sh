#!/bin/bash

# RAGシステム環境セットアップスクリプト

set -e

echo "🚀 RAGシステム環境セットアップを開始します..."

# プロジェクトルートに移動
cd "$(dirname "$0")/../.."

# 1. 必要なディレクトリを作成
echo "📁 ディレクトリ構造を作成中..."
mkdir -p data/rag_documents/{road_standards,regulations,technical_guides}
mkdir -p qdrant_data
mkdir -p cache/embeddings
mkdir -p outputs/rag_models

# 2. Python環境の確認
echo "🐍 Python環境を確認中..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3がインストールされていません"
    exit 1
fi

python3 --version

# 3. 仮想環境の作成（オプション）
if [ "$1" = "--create-venv" ]; then
    echo "📦 仮想環境を作成中..."
    if [ ! -d "venv_rag" ]; then
        python3 -m venv venv_rag
    fi
    source venv_rag/bin/activate
    echo "✅ 仮想環境を有効化しました"
fi

# 4. RAG用依存関係のインストール
echo "📦 RAG用依存関係をインストール中..."
pip install -r requirements_rag.txt

# 5. Spacy日本語モデルのダウンロード
echo "🌐 Spacy日本語モデルをダウンロード中..."
python -m spacy download ja_core_news_lg

# 6. 埋め込みモデルの事前ダウンロード
echo "🤖 埋め込みモデルを事前ダウンロード中..."
python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# multilingual-e5-large
print('Downloading multilingual-e5-large...')
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
print('✅ multilingual-e5-large downloaded')

# 日本語SentenceBERT
print('Downloading Japanese SentenceBERT...')
model_st = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
print('✅ Japanese SentenceBERT downloaded')
"

# 7. Qdrantローカルサーバーの準備
echo "🗄️ Qdrantローカルサーバーを準備中..."
if [ ! -d "qdrant_data" ]; then
    mkdir -p qdrant_data
fi

# 8. サンプル設定ファイルのコピー
echo "⚙️ 設定ファイルを準備中..."
if [ ! -f "src/rag/config/rag_config.local.yaml" ]; then
    cp src/rag/config/rag_config.yaml src/rag/config/rag_config.local.yaml
    echo "✅ ローカル設定ファイルを作成しました"
fi

# 9. 権限設定
echo "🔐 権限を設定中..."
chmod -R 755 qdrant_data
chmod -R 755 data/rag_documents
chmod -R 755 cache

# 10. テスト実行
echo "🧪 基本テストを実行中..."
python -c "
import sys
sys.path.append('.')

try:
    from src.rag.indexing.embedding_model import EmbeddingModelFactory
    from src.rag.indexing.vector_store import QdrantVectorStore
    
    # 埋め込みモデルのテスト
    print('Testing embedding model...')
    model = EmbeddingModelFactory.create('multilingual-e5-large')
    test_embeddings = model.encode(['テスト文書'])
    print(f'✅ Embedding model OK (dim: {test_embeddings.shape[1]})')
    
    # ベクトルストレージのテスト
    print('Testing vector store...')
    vector_store = QdrantVectorStore(collection_name='test_collection')
    vector_store.add_documents(
        texts=['テスト文書'],
        embeddings=test_embeddings,
        metadatas=[{'type': 'test'}],
        ids=['test_1']
    )
    print('✅ Vector store OK')
    
    # テストコレクションをクリア
    vector_store.clear_collection()
    
except Exception as e:
    print(f'❌ テストエラー: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 RAGシステム環境セットアップが完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. data/rag_documents/ に道路設計文書（PDF）を配置"
echo "2. python scripts/rag/index_documents.py を実行してインデックス作成"
echo "3. RAG Web UIを起動: streamlit run app/rag_interface.py"
echo ""
echo "📚 ディレクトリ構造:"
echo "├── data/rag_documents/     # 文書置き場"
echo "├── qdrant_data/           # ベクトルDB"
echo "├── cache/embeddings/      # 埋め込みキャッシュ"
echo "└── outputs/rag_models/    # RAG出力"
echo ""