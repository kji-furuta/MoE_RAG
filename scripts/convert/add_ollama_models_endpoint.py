#!/usr/bin/env python3
"""
OllamaモデルリストAPIエンドポイントを追加
"""

import subprocess
import json

# Ollamaモデルリストを取得する関数
def get_ollama_models():
    """Ollamaに登録されているモデルを取得"""
    try:
        # ollama listコマンドを実行
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 出力を解析
        lines = result.stdout.strip().split('\n')
        models = []
        
        for line in lines[1:]:  # ヘッダー行をスキップ
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0].replace(':latest', '')
                    models.append(model_name)
        
        return models
        
    except Exception as e:
        print(f"エラー: {e}")
        return []

# APIで取得する方法
def get_ollama_models_via_api():
    """Ollama APIを使用してモデルリストを取得"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            check=True
        )
        
        data = json.loads(result.stdout)
        models = []
        
        if 'models' in data:
            for model in data['models']:
                # モデル名を抽出
                name = model.get('name', '').replace(':latest', '')
                if name:
                    models.append(name)
        
        return models
        
    except Exception as e:
        print(f"APIエラー: {e}")
        return []

def main():
    print("Ollamaモデルリストを取得中...")
    
    # コマンドラインから取得
    print("\n方法1: ollama listコマンド")
    models_cmd = get_ollama_models()
    print(f"検出されたモデル: {models_cmd}")
    
    # APIから取得
    print("\n方法2: Ollama API")
    models_api = get_ollama_models_via_api()
    print(f"検出されたモデル: {models_api}")
    
    # 統合リスト
    all_models = list(set(models_cmd + models_api))
    
    print("\n統合モデルリスト:")
    for model in all_models:
        print(f"  - {model}")
    
    # deepseek-32b-finetunedが含まれているか確認
    if 'deepseek-32b-finetuned' in all_models:
        print("\n✅ deepseek-32b-finetuned が検出されました！")
    else:
        print("\n⚠️ deepseek-32b-finetuned が見つかりません")
        print("以下のコマンドで再登録してください:")
        print("ollama create deepseek-32b-finetuned -f /workspace/models/Modelfile_finetuned")

if __name__ == "__main__":
    main()