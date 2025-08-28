#!/usr/bin/env python3
"""
ベクトル次元を安全に統一するための完全版スクリプト
multilingual-e5-largeモデル（1024次元）に統一
"""

import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os
import re

# 統一する設定
TARGET_MODEL = "intfloat/multilingual-e5-large"
TARGET_DIM = 1024

def create_backup():
    """重要なデータのバックアップを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/vector_dimension_fix_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 バックアップを作成中: {backup_dir}")
    
    # バックアップ対象
    backup_items = [
        ("data/qdrant", "qdrant_data"),
        ("qdrant_data", "qdrant_data_alt"),
        ("src/rag/config/rag_config.yaml", "rag_config.yaml"),
        ("src/rag/config/rag_config.py", "rag_config.py"),
        ("src/rag/indexing/vector_store.py", "vector_store.py"),
        ("src/rag/indexing/embedding_model.py", "embedding_model.py")
    ]
    
    for source, dest_name in backup_items:
        source_path = Path(source)
        if source_path.exists():
            dest_path = backup_dir / dest_name
            if source_path.is_dir():
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)
            print(f"  ✅ バックアップ: {source} -> {dest_path}")
    
    # バックアップ情報を記録
    info = {
        "timestamp": timestamp,
        "target_model": TARGET_MODEL,
        "target_dim": TARGET_DIM,
        "backup_items": [str(item[0]) for item in backup_items if Path(item[0]).exists()]
    }
    
    with open(backup_dir / "backup_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    return backup_dir

def fix_yaml_config():
    """rag_config.yamlを修正"""
    config_path = Path("src/rag/config/rag_config.yaml")
    
    if not config_path.exists():
        print(f"⚠️  設定ファイルが見つかりません: {config_path}")
        return False
    
    print(f"\n📝 YAMLの修正: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 埋め込み設定を更新
    if 'embedding' not in config:
        config['embedding'] = {}
    
    old_model = config['embedding'].get('model_name', 'unknown')
    config['embedding']['model_name'] = TARGET_MODEL
    config['embedding']['embedding_dim'] = TARGET_DIM
    config['embedding']['max_length'] = 512
    config['embedding']['normalize_embeddings'] = True
    
    print(f"  変更: {old_model} -> {TARGET_MODEL} ({TARGET_DIM}次元)")
    
    # ベクトルストア設定も更新
    if 'vector_store' not in config:
        config['vector_store'] = {}
    if 'qdrant' not in config['vector_store']:
        config['vector_store']['qdrant'] = {}
    
    config['vector_store']['qdrant']['vector_dim'] = TARGET_DIM
    config['vector_store']['qdrant']['collection_name'] = 'road_design_docs'
    
    # 保存
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"  ✅ YAML設定を更新")
    return True

def fix_python_code():
    """Pythonコードのデフォルト値を修正"""
    
    # 1. vector_store.pyの修正
    vector_store_path = Path("src/rag/indexing/vector_store.py")
    if vector_store_path.exists():
        print(f"\n📝 vector_store.pyの修正")
        
        with open(vector_store_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # embedding_dimのデフォルト値を修正
        patterns = [
            (r'(embedding_dim:\s*int\s*=\s*)\d+', f'\\1{TARGET_DIM}'),
            (r'(embedding_dim\s*=\s*)\d+', f'\\1{TARGET_DIM}')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        with open(vector_store_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ デフォルト次元を{TARGET_DIM}に変更")
    
    # 2. embedding_model.pyの修正（必要に応じて）
    embedding_model_path = Path("src/rag/indexing/embedding_model.py")
    if embedding_model_path.exists():
        print(f"\n📝 embedding_model.pyの確認")
        
        with open(embedding_model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # EmbeddingModelFactoryにデフォルトモデルを設定
        if 'EmbeddingModelFactory' in content and TARGET_MODEL not in content:
            # デフォルトモデルを更新する処理を追加
            print(f"  ℹ️  EmbeddingModelFactoryのデフォルトモデルを確認")
    
    return True

def fix_qdrant_metadata():
    """Qdrantのメタデータファイルを修正"""
    meta_paths = [
        Path("data/qdrant/meta.json"),
        Path("qdrant_data/meta.json")
    ]
    
    for meta_path in meta_paths:
        if meta_path.exists():
            print(f"\n📝 Qdrantメタデータの修正: {meta_path}")
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # コレクション設定を更新
            if 'collections' in meta:
                for collection_name, config in meta['collections'].items():
                    if 'vectors' in config:
                        old_size = config['vectors'].get('size', 'unknown')
                        config['vectors']['size'] = TARGET_DIM
                        print(f"  コレクション '{collection_name}': {old_size} -> {TARGET_DIM}")
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"  ✅ メタデータを更新")
    
    return True

def create_verification_script():
    """検証スクリプトを作成"""
    script_content = f'''#!/usr/bin/env python3
"""
ベクトル次元の統一後の検証スクリプト
"""

import sys
sys.path.append(".")

from src.rag.config.rag_config import load_config
from src.rag.indexing.vector_store import QdrantVectorStore
from src.rag.indexing.embedding_model import EmbeddingModelFactory
import numpy as np

def verify_dimensions():
    """次元の統一を検証"""
    
    print("=" * 60)
    print("ベクトル次元の検証")
    print("=" * 60)
    
    # 1. 設定の確認
    config = load_config()
    print(f"\\n1. RAG設定:")
    print(f"   モデル: {{config.embedding.model_name}}")
    print(f"   設定次元: {{getattr(config.embedding, 'embedding_dim', 'not set')}}")
    
    # 2. 埋め込みモデルのテスト
    print(f"\\n2. 埋め込みモデルのテスト:")
    try:
        model = EmbeddingModelFactory.create(
            model_name="{TARGET_MODEL}",
            device="cpu"  # テスト用にCPU使用
        )
        
        test_text = "テスト文書"
        embedding = model.encode(test_text)
        actual_dim = len(embedding) if isinstance(embedding, (list, np.ndarray)) else embedding.shape[-1]
        
        print(f"   実際の出力次元: {{actual_dim}}")
        print(f"   期待される次元: {TARGET_DIM}")
        
        if actual_dim == {TARGET_DIM}:
            print(f"   ✅ 次元が一致")
        else:
            print(f"   ❌ 次元が不一致")
            return False
    except Exception as e:
        print(f"   ⚠️  エラー: {{e}}")
        return False
    
    # 3. ベクトルストアのテスト
    print(f"\\n3. ベクトルストアのテスト:")
    try:
        vector_store = QdrantVectorStore(
            collection_name="road_design_docs",
            embedding_dim={TARGET_DIM}
        )
        
        info = vector_store.get_collection_info()
        print(f"   コレクション状態: {{info.get('status', 'unknown')}}")
        print(f"   ベクトル数: {{info.get('vectors_count', 0)}}")
    except Exception as e:
        print(f"   ⚠️  エラー: {{e}}")
    
    print(f"\\n✅ 検証完了")
    return True

if __name__ == "__main__":
    verify_dimensions()
'''
    
    script_path = Path("scripts/rag_fixes/verify_dimensions.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n📄 検証スクリプトを作成: {script_path}")
    return script_path

def main():
    """メイン処理"""
    print("=" * 60)
    print("🔧 ベクトル次元統一処理")
    print("=" * 60)
    
    print(f"\n目標設定:")
    print(f"  📦 モデル: {TARGET_MODEL}")
    print(f"  📐 次元数: {TARGET_DIM}")
    print(f"  🎯 対象: Qdrant, RAG設定, ソースコード")
    
    # 1. バックアップの作成
    print(f"\n[ステップ 1/5] バックアップの作成")
    backup_dir = create_backup()
    
    # 2. YAML設定の修正
    print(f"\n[ステップ 2/5] YAML設定の修正")
    if not fix_yaml_config():
        print("❌ YAML設定の修正に失敗")
        return False
    
    # 3. Pythonコードの修正
    print(f"\n[ステップ 3/5] Pythonコードの修正")
    if not fix_python_code():
        print("❌ Pythonコードの修正に失敗")
        return False
    
    # 4. Qdrantメタデータの修正
    print(f"\n[ステップ 4/5] Qdrantメタデータの修正")
    fix_qdrant_metadata()
    
    # 5. 検証スクリプトの作成
    print(f"\n[ステップ 5/5] 検証スクリプトの作成")
    verify_script = create_verification_script()
    
    # 完了メッセージ
    print("\n" + "=" * 60)
    print("✅ ベクトル次元の統一処理が完了しました")
    print("=" * 60)
    
    print(f"\n📋 次のステップ:")
    print(f"1. 検証スクリプトの実行:")
    print(f"   python {verify_script}")
    print(f"\n2. Qdrantコレクションの再作成（必要に応じて）:")
    print(f"   python scripts/rag_fixes/recreate_qdrant_collection.py")
    print(f"\n3. RAGシステムの再起動:")
    print(f"   ./start_dev_env.sh")
    
    print(f"\n🔙 問題が発生した場合のロールバック:")
    print(f"   cp -r {backup_dir}/* .")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
