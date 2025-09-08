#!/usr/bin/env python3
"""
RAG設定の読み込みをデバッグするスクリプト
"""

import sys
import os
sys.path.insert(0, '/home/kjifu/MoE_RAG')
os.chdir('/home/kjifu/MoE_RAG')

from src.rag.config.rag_config import load_config
import yaml
from pathlib import Path

def debug_config():
    """設定ファイルの内容を確認"""
    print("=" * 60)
    print("📋 RAG設定デバッグ")
    print("=" * 60)
    
    # 1. YAMLファイルを直接読み込み
    config_path = Path("src/rag/config/rag_config.yaml")
    print(f"\n1. YAMLファイル直接読み込み: {config_path}")
    print("-" * 40)
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            
        llm_config = yaml_config.get('llm', {})
        print(f"provider: {llm_config.get('provider')}")
        print(f"model_name: {llm_config.get('model_name')}")
        print(f"ollama_model: {llm_config.get('ollama_model')}")
        
        if 'ollama' in llm_config:
            print(f"ollama.model: {llm_config['ollama'].get('model')}")
            print(f"ollama.base_url: {llm_config['ollama'].get('base_url')}")
    else:
        print(f"❌ ファイルが見つかりません: {config_path}")
    
    # 2. load_config()関数で読み込み
    print(f"\n2. load_config()関数での読み込み")
    print("-" * 40)
    
    try:
        config = load_config()
        
        if hasattr(config, 'llm'):
            print(f"config.llm.provider: {getattr(config.llm, 'provider', 'なし')}")
            print(f"config.llm.model_name: {getattr(config.llm, 'model_name', 'なし')}")
            print(f"config.llm.ollama_model: {getattr(config.llm, 'ollama_model', 'なし')}")
            
            if hasattr(config.llm, 'ollama'):
                print(f"config.llm.ollama.model: {getattr(config.llm.ollama, 'model', 'なし')}")
                print(f"config.llm.ollama.base_url: {getattr(config.llm.ollama, 'base_url', 'なし')}")
            else:
                print("config.llm.ollama: なし")
        else:
            print("❌ config.llmが存在しません")
            
    except Exception as e:
        print(f"❌ load_config()エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. QueryEngineでの読み込みをシミュレート
    print(f"\n3. QueryEngineでのモデル選択ロジック")
    print("-" * 40)
    
    try:
        config = load_config()
        
        # QueryEngineと同じロジック
        ollama_model = 'llama3.2:3b'  # デフォルト
        
        if hasattr(config.llm, 'ollama') and hasattr(config.llm.ollama, 'model'):
            ollama_model = config.llm.ollama.model
            print(f"✅ config.llm.ollama.modelから取得: {ollama_model}")
        elif hasattr(config.llm, 'ollama_model'):
            ollama_model = config.llm.ollama_model
            print(f"✅ config.llm.ollama_modelから取得: {ollama_model}")
        elif hasattr(config.llm, 'model_name') and config.llm.model_name.startswith('ollama:'):
            ollama_model = config.llm.model_name[7:]
            print(f"✅ config.llm.model_nameから取得: {ollama_model}")
        else:
            print(f"⚠️ デフォルト値を使用: {ollama_model}")
        
        print(f"\n最終的に使用されるモデル: {ollama_model}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()