#!/usr/bin/env python3
"""
ベクトル次元の統一を実行するスクリプト
安全にバックアップを作成してから修正を適用
"""

import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# 標準的なモデルと次元のマッピング
MODEL_DIMENSIONS = {
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-small": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "openai/text-embedding-ada-002": 1536,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}

def backup_data():
    """データのバックアップを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/rag_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"バックアップを作成中: {backup_dir}")
    
    # Qdrantデータのバックアップ
    qdrant_paths = [
        Path("data/qdrant"),
        Path("qdrant_data")
    ]
    
    for qdrant_path in qdrant_paths:
        if qdrant_path.exists():
            backup_path = backup_dir / qdrant_path.name
            shutil.copytree(qdrant_path, backup_path)
            print(f"  ✅ {qdrant_path} -> {backup_path}")
    
    # 設定ファイルのバックアップ
    config_files = [
        Path("src/rag/config/rag_config.yaml"),
        Path("src/rag/config/rag_config.py")
    ]
    
    for config_file in config_files:
        if config_file.exists():
            backup_path = backup_dir / config_file.name
            shutil.copy2(config_file, backup_path)
            print(f"  ✅ {config_file} -> {backup_path}")
    
    return backup_dir

def fix_rag_config(target_model: str, target_dim: int):
    """RAG設定ファイルを修正"""
    config_path = Path("src/rag/config/rag_config.yaml")
    
    if not config_path.exists():
        print(f"⚠️  設定ファイルが見つかりません: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 埋め込み設定を更新
    if 'embedding' not in config:
        config['embedding'] = {}
    
    config['embedding']['model_name'] = target_model
    config['embedding']['embedding_dim'] = target_dim  # 明示的に次元を設定
    
    # ベクトルストア設定も更新
    if 'vector_store' not in config:
        config['vector_store'] = {}
    if 'qdrant' not in config['vector_store']:
        config['vector_store']['qdrant'] = {}
    
    config['vector_store']['qdrant']['vector_dim'] = target_dim
    
    # ファイルに書き戻し
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"✅ RAG設定ファイルを更新: {target_model} ({target_dim}次元)")
    return True

def fix_vector_store_code(target_dim: int):
    """vector_store.pyのデフォルト値を修正"""
    vector_store_path = Path("src/rag/indexing/vector_store.py")
    
    if not vector_store_path.exists():
        print(f"⚠️  vector_store.pyが見つかりません: {vector_store_path}")
        return False
    
    with open(vector_store_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # embedding_dim のデフォルト値を置換
    import re
    
    # パターン1: __init__メソッドのデフォルト値
    pattern1 = r'(embedding_dim:\s*int\s*=\s*)\d+'
    content = re.sub(pattern1, f'\\g<1>{target_dim}', content)
    
    # パターン2: VectorStoreクラスのデフォルト値
    pattern2 = r'(def __init__\(self,.*?embedding_dim:\s*int\s*=\s*)\d+'
    content = re.sub(pattern2, f'\\g<1>{target_dim}', content, flags=re.DOTALL)
    
    with open(vector_store_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ vector_store.pyのデフォルト値を更新: {target_dim}次元")
    return True

def create_recreation_script(target_model: str, target_dim: int):
    """Qdrantコレクションを再作成するスクリプトを生成"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Qdrantコレクションを再作成するスクリプト
次元: {target_dim}
モデル: {target_model}
"""

import sys
sys.path.append(".")

from src.rag.indexing.vector_store import QdrantVectorStore
from src.rag.indexing.embedding_model import EmbeddingModelFactory
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_collection():
    """コレクションを再作成"""
    
    # 1. 新しいベクトルストアの初期化
    vector_store = QdrantVectorStore(
        collection_name="road_design_docs",
        embedding_dim={target_dim},
        path="./data/qdrant"
    )
    
    # 2. 既存のコレクションを削除（存在する場合）
    try:
        vector_store.client.delete_collection("road_design_docs")
        logger.info("既存のコレクションを削除しました")
    except Exception as e:
        logger.info(f"既存のコレクションは存在しません: {{e}}")
    
    # 3. 新しいコレクションを作成
    vector_store._ensure_collection()
    logger.info(f"新しいコレクションを作成しました (次元: {target_dim})")
    
    # 4. 確認
    info = vector_store.get_collection_info()
    print(f"\\nコレクション情報:")
    print(f"  ベクトル数: {{info.get('vectors_count', 0)}}")
    print(f"  ステータス: {{info.get('status', 'unknown')}}")
    
    # 5. テストデータの追加
    print("\\nテストデータを追加中...")
    
    # 埋め込みモデルの初期化
    embedding_model = EmbeddingModelFactory.create(
        model_name="{target_model}",
        device="cuda"
    )
    
    # サンプルテキスト
    test_texts = [
        "道路設計における最小曲線半径の決定要因について",
        "設計速度60km/hの場合の安全基準",
        "縦断勾配の制限値と適用条件"
    ]
    
    # 埋め込みベクトルの生成
    embeddings = embedding_model.encode_batch(test_texts)
    
    # メタデータ
    metadatas = [
        {{"doc_id": f"test_{{i}}", "title": f"テスト文書{{i}}", "category": "test"}}
        for i in range(len(test_texts))
    ]
    
    # IDの生成
    import uuid
    ids = [str(uuid.uuid4()) for _ in test_texts]
    
    # データの追加
    vector_store.add_documents(
        texts=test_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ {{len(test_texts)}}件のテストデータを追加しました")
    
    # 6. 検索テスト
    print("\\n検索テストを実行中...")
    query = "設計速度"
    query_embedding = embedding_model.encode(query)
    
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3
    )
    
    print(f"クエリ: '{{query}}'")
    print(f"検索結果: {{len(results)}}件")
    for i, result in enumerate(results, 1):
        print(f"  {{i}}. スコア={{result.score:.3f}}, テキスト={{result.text[:50]}}...")
    
    print("\\n✅ コレクションの再作成が完了しました")

if __name__ == "__main__":
    recreate_collection()
'''
    
    script_path = Path("scripts/rag_fixes/recreate_qdrant_collection.py")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 実行権限を付与
    os.chmod(script_path, 0o755)
    
    print(f"✅ 再作成スクリプトを生成: {script_path}")
    return script_path

def main():
    """メイン処理"""
    print("=" * 60)
    print("ベクトル次元の統一処理")
    print("=" * 60)
    
    # 1. ターゲットモデルと次元の選択
    target_model = "intfloat/multilingual-e5-large"
    target_dim = 1024
    
    print(f"\n目標設定:")
    print(f"  モデル: {target_model}")
    print(f"  次元数: {target_dim}")
    
    # ユーザー確認
    response = input("\n続行しますか？ (y/n): ")
    if response.lower() != 'y':
        print("処理を中止しました")
        return
    
    # 2. バックアップの作成
    print(f"\n1. バックアップの作成")
    backup_dir = backup_data()
    print(f"   バックアップ完了: {backup_dir}")
    
    # 3. 設定ファイルの修正
    print(f"\n2. 設定ファイルの修正")
    
    # RAG設定の修正
    if not fix_rag_config(target_model, target_dim):
        print("❌ RAG設定の修正に失敗しました")
        return
    
    # ソースコードの修正
    if not fix_vector_store_code(target_dim):
        print("❌ ソースコードの修正に失敗しました")
        return
    
    # 4. 再作成スクリプトの生成
    print(f"\n3. Qdrantコレクション再作成スクリプトの生成")
    script_path = create_recreation_script(target_model, target_dim)
    
    # 5. 完了メッセージ
    print("\n" + "=" * 60)
    print("✅ ベクトル次元の統一処理が完了しました")
    print("=" * 60)
    
    print(f"\n次のステップ:")
    print(f"1. Dockerコンテナ内で以下のコマンドを実行:")
    print(f"   docker exec -it ai-ft-container bash")
    print(f"   cd /workspace")
    print(f"   python {script_path}")
    print(f"\n2. RAGシステムの再起動:")
    print(f"   docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh")
    
    print(f"\n問題が発生した場合のロールバック:")
    print(f"   cp -r {backup_dir}/* .")

if __name__ == "__main__":
    main()
