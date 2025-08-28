#!/usr/bin/env python3
"""
ベクトル次元の不整合を検証するスクリプト
"""

import json
import yaml
from pathlib import Path
import sys

def check_dimensions():
    """ベクトル次元の設定を確認"""
    
    print("=" * 60)
    print("ベクトル次元の検証")
    print("=" * 60)
    
    issues = []
    
    # 1. RAG設定ファイルの確認
    config_path = Path("src/rag/config/rag_config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            embedding_config = config.get('embedding', {})
            model_name = embedding_config.get('model_name', 'unknown')
            print(f"\n1. RAG設定ファイル (rag_config.yaml):")
            print(f"   モデル名: {model_name}")
            
            # モデルごとの次元数
            model_dimensions = {
                "intfloat/multilingual-e5-large": 1024,
                "intfloat/multilingual-e5-base": 768,
                "sentence-transformers/all-MiniLM-L12-v2": 384,
                "openai/text-embedding-ada-002": 1536,
                "BAAI/bge-large-en-v1.5": 1024,
            }
            
            expected_dim = model_dimensions.get(model_name, "不明")
            print(f"   期待される次元数: {expected_dim}")
    else:
        print(f"\n1. RAG設定ファイルが見つかりません: {config_path}")
        issues.append("RAG設定ファイルが見つかりません")
    
    # 2. Qdrantメタデータの確認
    qdrant_meta_paths = [
        Path("data/qdrant/meta.json"),
        Path("qdrant_data/meta.json")
    ]
    
    actual_dim = None
    for meta_path in qdrant_meta_paths:
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                collections = meta.get('collections', {})
                
                print(f"\n2. Qdrantメタデータ ({meta_path}):")
                for collection_name, config in collections.items():
                    if 'vectors' in config:
                        actual_dim = config['vectors'].get('size', '不明')
                        print(f"   コレクション: {collection_name}")
                        print(f"   実際の次元数: {actual_dim}")
                        print(f"   距離メトリック: {config['vectors'].get('distance', '不明')}")
                        
                        # 不整合チェック
                        if expected_dim != "不明" and actual_dim != expected_dim:
                            issues.append(f"次元の不整合: 期待値={expected_dim}, 実際={actual_dim}")
                break
    
    if actual_dim is None:
        print("\n2. Qdrantメタデータファイルが見つかりません")
        issues.append("Qdrantメタデータファイルが見つかりません")
    
    # 3. ソースコードのデフォルト値確認
    vector_store_path = Path("src/rag/indexing/vector_store.py")
    if vector_store_path.exists():
        with open(vector_store_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # embedding_dim のデフォルト値を探す
            import re
            pattern = r'embedding_dim:\s*int\s*=\s*(\d+)'
            matches = re.findall(pattern, content)
            if matches:
                code_dim = int(matches[0])
                print(f"\n3. ソースコード (vector_store.py):")
                print(f"   デフォルト次元数: {code_dim}")
                
                if expected_dim != "不明" and code_dim != expected_dim:
                    issues.append(f"コードのデフォルト値が不整合: {code_dim} != {expected_dim}")
    
    # 4. 埋め込みモデルの実際の次元数を確認
    print(f"\n4. 埋め込みモデルの実際の次元数確認:")
    try:
        from sentence_transformers import SentenceTransformer
        
        # multilingual-e5-largeをロード
        if model_name == "intfloat/multilingual-e5-large":
            print(f"   {model_name} をロード中...")
            model = SentenceTransformer(model_name)
            test_embedding = model.encode("test")
            real_dim = len(test_embedding)
            print(f"   実際の出力次元数: {real_dim}")
            
            if real_dim != expected_dim:
                issues.append(f"モデルの実際の出力次元が異なる: {real_dim} != {expected_dim}")
    except Exception as e:
        print(f"   モデルのロードに失敗: {e}")
    
    # 結果まとめ
    print("\n" + "=" * 60)
    if issues:
        print("❌ 検証結果: 問題が見つかりました")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False
    else:
        print("✅ 検証結果: 次元の設定は一致しています")
        return True

if __name__ == "__main__":
    success = check_dimensions()
    sys.exit(0 if success else 1)
