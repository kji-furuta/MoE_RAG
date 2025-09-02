#!/usr/bin/env python3
"""
RAGシステムを強制的にOllamaモードに設定
"""

import yaml
import os
import sys

def force_ollama_mode():
    """設定ファイルを強制的にOllamaモードに変更"""
    
    config_path = '/workspace/src/rag/config/rag_config.yaml'
    
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '../src/rag/config/rag_config.yaml')
    
    print(f"設定ファイル: {config_path}")
    
    try:
        # 現在の設定を読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # LLM設定を強制的にOllamaモードに
        if 'llm' not in config:
            config['llm'] = {}
        
        # 重要な設定を全て設定
        config['llm']['provider'] = 'ollama'
        config['llm']['use_ollama_fallback'] = True
        config['llm']['use_finetuned'] = False  # ローカルモデルを無効化
        config['llm']['ollama_model'] = 'deepseek-32b-finetuned'
        config['llm']['ollama_host'] = 'http://localhost:11434'
        config['llm']['model_name'] = 'ollama:deepseek-32b-finetuned'
        
        # Ollamaセクションも追加
        config['llm']['ollama'] = {
            'model': 'deepseek-32b-finetuned',
            'base_url': 'http://localhost:11434',
            'temperature': 0.6,
            'top_p': 0.9,
            'max_tokens': 2048
        }
        
        # ベースモデルを無効化（ローカルモデルのロードを防ぐ）
        config['llm']['base_model'] = None
        config['llm']['model_path'] = None
        config['llm']['finetuned_model_path'] = None
        
        # 更新した設定を保存
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print("✅ 設定を強制的にOllamaモードに更新しました")
        print("\n現在の設定:")
        print(f"  プロバイダー: {config['llm']['provider']}")
        print(f"  モデル: {config['llm']['ollama_model']}")
        print(f"  use_finetuned: {config['llm']['use_finetuned']}")
        print(f"  use_ollama_fallback: {config['llm']['use_ollama_fallback']}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def verify_ollama_model():
    """Ollamaモデルが登録されているか確認"""
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "deepseek-32b-finetuned" in result.stdout:
            print("\n✅ deepseek-32b-finetuned モデルが登録されています")
            return True
        else:
            print("\n⚠️ deepseek-32b-finetuned モデルが見つかりません")
            print("登録コマンド:")
            print("cd /workspace/models && ollama create deepseek-32b-finetuned -f Modelfile_finetuned")
            return False
    except:
        # コンテナ内で実行
        try:
            result = subprocess.run(
                ["docker", "exec", "ai-ft-container", "ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "deepseek-32b-finetuned" in result.stdout:
                print("\n✅ deepseek-32b-finetuned モデルが登録されています")
                return True
            else:
                print("\n⚠️ deepseek-32b-finetuned モデルが見つかりません")
                return False
        except Exception as e:
            print(f"\n❌ Ollama確認エラー: {e}")
            return False

def main():
    print("RAGシステムを強制的にOllamaモードに設定")
    print("="*60)
    
    # 設定を強制更新
    if force_ollama_mode():
        # Ollamaモデルを確認
        verify_ollama_model()
        
        print("\n" + "="*60)
        print("✅ 設定完了")
        print("\n次のステップ:")
        print("1. Webサーバーを再起動（自動リロードされる場合は不要）")
        print("2. RAGページをリロード（Ctrl+F5）")
        print("3. クエリを実行してdeepseek-32b-finetunedが使用されることを確認")
    else:
        print("\n❌ 設定更新に失敗しました")

if __name__ == "__main__":
    main()