#!/usr/bin/env python3
"""
RAGシステムのモデルを直接更新するスクリプト
"""

import yaml
import sys
from pathlib import Path

def update_rag_model(model_name):
    """RAG設定ファイルのモデルを更新"""
    config_path = Path("/home/kjifu/MoE_RAG/src/rag/config/rag_config.yaml")
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return False
    
    try:
        # 設定を読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # LLMセクションが存在しない場合は作成
        if 'llm' not in config:
            config['llm'] = {}
        
        # Ollamaセクションが存在しない場合は作成
        if 'ollama' not in config['llm']:
            config['llm']['ollama'] = {}
        
        # モデル設定を更新
        config['llm']['provider'] = 'ollama'
        config['llm']['model_name'] = f'ollama:{model_name}'
        config['llm']['ollama_model'] = model_name
        config['llm']['ollama']['model'] = model_name
        config['llm']['ollama']['base_url'] = 'http://localhost:11434'
        config['llm']['use_ollama_fallback'] = True
        
        # 設定を保存
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✅ RAGモデル設定を更新しました: {model_name}")
        print(f"   設定ファイル: {config_path}")
        
        # 更新された設定を表示
        print("\n📊 更新後の設定:")
        print(f"   provider: {config['llm']['provider']}")
        print(f"   model_name: {config['llm']['model_name']}")
        print(f"   ollama_model: {config['llm']['ollama_model']}")
        print(f"   ollama.model: {config['llm']['ollama']['model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("使用方法: python update_rag_model.py <model_name>")
        print("例: python update_rag_model.py deepseek-32b-finetuned:latest")
        print("    python update_rag_model.py llama3.2:3b")
        sys.exit(1)
    
    model_name = sys.argv[1]
    print(f"🔄 RAGモデルを '{model_name}' に更新中...")
    
    if update_rag_model(model_name):
        print("\n✅ 更新完了！")
        print("💡 Webインターフェースを再起動するか、新しいクエリを実行してください")
    else:
        print("\n❌ 更新に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()