#!/usr/bin/env python3
"""
RAGシステムのモデル設定を強制的に更新
"""

import yaml
import json
import os

def update_rag_config():
    """RAG設定ファイルを更新"""
    config_path = '/workspace/src/rag/config/rag_config.yaml'
    
    if not os.path.exists(config_path):
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return False
    
    print(f"設定ファイルを更新中: {config_path}")
    
    try:
        # 現在の設定を読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # バックアップを作成
        with open(config_path + '.backup', 'w') as f:
            yaml.dump(config, f)
        
        # Ollamaモデルを設定
        if 'llm' not in config:
            config['llm'] = {}
        
        config['llm']['provider'] = 'ollama'
        config['llm']['ollama'] = {
            'model': 'deepseek-32b-finetuned',
            'base_url': 'http://localhost:11434',
            'temperature': 0.6,
            'top_p': 0.9,
            'max_tokens': 2048
        }
        
        # 使用可能なモデルリストも更新
        config['llm']['available_models'] = [
            'deepseek-32b-finetuned',
            'llama3.2:3b',
            'deepseek-r1:32b'
        ]
        
        # 更新した設定を保存
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print("✅ RAG設定を更新しました")
        print("\n現在の設定:")
        print(f"  プロバイダー: {config['llm']['provider']}")
        print(f"  モデル: {config['llm']['ollama']['model']}")
        print(f"  温度: {config['llm']['ollama']['temperature']}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def verify_ollama_connection():
    """Ollamaサービスへの接続を確認"""
    import subprocess
    
    print("\nOllama接続確認中...")
    
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            data = json.loads(result.stdout)
            if 'models' in data:
                print("✅ Ollamaサービスに接続成功")
                print(f"  登録モデル数: {len(data['models'])}")
                
                # deepseek-32b-finetunedを探す
                for model in data['models']:
                    if 'deepseek-32b-finetuned' in model.get('name', ''):
                        print("✅ deepseek-32b-finetuned が登録されています")
                        return True
                
                print("⚠️ deepseek-32b-finetuned が見つかりません")
                print("\n再登録するには:")
                print("ollama create deepseek-32b-finetuned -f /workspace/models/Modelfile_finetuned")
        else:
            print("❌ Ollamaサービスからレスポンスがありません")
            
    except Exception as e:
        print(f"❌ Ollama接続エラー: {e}")
        print("\nOllamaサービスを起動してください:")
        print("ollama serve &")
    
    return False

def main():
    print("RAGシステムモデル設定の強制更新")
    print("="*60)
    
    # Ollama接続確認
    ollama_ok = verify_ollama_connection()
    
    # RAG設定更新
    if update_rag_config():
        print("\n" + "="*60)
        print("✅ 設定更新完了")
        print("\n次のステップ:")
        print("1. Webインターフェースをリロード (Ctrl+F5)")
        print("2. RAGページで確認")
        
        if not ollama_ok:
            print("\n⚠️ 注意: Ollamaにモデルが登録されていません")
            print("以下のコマンドで登録してください:")
            print("ollama create deepseek-32b-finetuned -f /workspace/models/Modelfile_finetuned")
    else:
        print("\n❌ 設定更新に失敗しました")

if __name__ == "__main__":
    main()