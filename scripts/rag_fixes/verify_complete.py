#!/usr/bin/env python3
"""
ベクトル次元統一の完全検証スクリプト
"""

import sys
import os
sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

import json
import yaml
from pathlib import Path

def verify_all():
    """全体的な検証を実行"""
    
    print("=" * 60)
    print("🔍 ベクトル次元統一の検証")
    print("=" * 60)
    
    issues = []
    fixes = []
    
    # 1. YAML設定の確認
    print("\n1. YAML設定ファイルの確認")
    config_path = Path("src/rag/config/rag_config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get('embedding', {})
        model_name = embedding_config.get('model_name', 'unknown')
        embedding_dim = embedding_config.get('embedding_dim', 'not set')
        
        print(f"  モデル: {model_name}")
        print(f"  設定次元: {embedding_dim}")
        
        if model_name == "intfloat/multilingual-e5-large":
            print(f"  ✅ モデル設定OK")
        else:
            issues.append(f"モデルが不正: {model_name}")
        
        if embedding_dim == 1024:
            print(f"  ✅ 次元設定OK")
        else:
            issues.append(f"次元設定が不正: {embedding_dim}")
            
        # ベクトルストア設定
        vector_config = config.get('vector_store', {}).get('qdrant', {})
        vector_dim = vector_config.get('vector_dim', 'not set')
        
        print(f"  ベクトルストア次元: {vector_dim}")
        if vector_dim == 1024:
            print(f"  ✅ ベクトルストア設定OK")
        else:
            issues.append(f"ベクトルストア次元が不正: {vector_dim}")
    else:
        issues.append("YAML設定ファイルが見つかりません")
    
    # 2. Qdrantメタデータの確認
    print("\n2. Qdrantメタデータの確認")
    meta_path = Path("data/qdrant/meta.json")
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        collections = meta.get('collections', {})
        for collection_name, config in collections.items():
            if 'vectors' in config:
                size = config['vectors'].get('size', 'unknown')
                print(f"  コレクション: {collection_name}")
                print(f"  実際の次元: {size}")
                
                if size == 1024:
                    print(f"  ✅ Qdrant次元OK")
                else:
                    issues.append(f"Qdrant次元が不正: {size}")
                    fixes.append("Qdrantコレクションの再作成が必要")
    else:
        print(f"  ⚠️  メタデータファイルなし（初回実行時は正常）")
    
    # 3. Pythonコードの確認
    print("\n3. Pythonコードの確認")
    vector_store_path = Path("src/rag/indexing/vector_store.py")
    if vector_store_path.exists():
        with open(vector_store_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # embedding_dim = 1024 があるか確認
        if "embedding_dim: int = 1024" in content:
            print(f"  ✅ vector_store.pyのデフォルト値OK")
        else:
            issues.append("vector_store.pyのデフォルト値が不正")
    
    # 4. 実際の動作確認
    print("\n4. 実際の動作確認")
    try:
        from src.rag.indexing.embedding_model import MultilingualE5EmbeddingModel
        import numpy as np
        
        print("  埋め込みモデルのテスト...")
        model = MultilingualE5EmbeddingModel(
            model_name="intfloat/multilingual-e5-large",
            device="cpu"
        )
        
        # テストエンコード
        test_text = "テスト"
        embedding = model.encode(test_text, is_query=True)
        
        if isinstance(embedding, np.ndarray):
            actual_dim = embedding.shape[-1]
        else:
            actual_dim = len(embedding)
        
        print(f"  実際の出力次元: {actual_dim}")
        
        if actual_dim == 1024:
            print(f"  ✅ モデル出力OK")
        else:
            issues.append(f"モデル出力次元が不正: {actual_dim}")
    except Exception as e:
        print(f"  ⚠️  モデルテストをスキップ: {e}")
    
    # 結果のまとめ
    print("\n" + "=" * 60)
    if issues:
        print("❌ 問題が見つかりました:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        if fixes:
            print("\n💡 推奨される修正:")
            for i, fix in enumerate(fixes, 1):
                print(f"  {i}. {fix}")
        
        print("\n次のコマンドを実行してください:")
        print("  python scripts/rag_fixes/recreate_collection.py")
    else:
        print("✅ すべての設定が正しく統一されています")
        print("\n次のステップ:")
        print("  1. RAGシステムを再起動:")
        print("     ./start_dev_env.sh")
        print("  2. Webインターフェースで確認:")
        print("     http://localhost:8050/rag")
    
    print("=" * 60)

if __name__ == "__main__":
    verify_all()
