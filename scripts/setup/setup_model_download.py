#!/usr/bin/env python3
"""
モデルダウンロード設定スクリプト
"""

import os
import json
from pathlib import Path

def setup_model_download():
    """モデルダウンロード設定"""
    print("=== モデルダウンロード設定 ===")
    print()
    
    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    print(f"プロジェクトルート: {project_root}")
    print(f"モデルディレクトリ: {models_dir}")
    print()
    
    # modelsディレクトリが存在しない場合は作成
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
        print(f"✅ モデルディレクトリを作成しました: {models_dir}")
    else:
        print(f"✅ モデルディレクトリが既に存在します: {models_dir}")
    
    # 環境変数を設定
    cache_dir = str(models_dir)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    print(f"✅ キャッシュディレクトリを設定しました: {cache_dir}")
    print()
    
    # 設定をファイルに保存
    config = {
        'models_directory': str(models_dir),
        'cache_directory': cache_dir,
        'hf_home': cache_dir,
        'transformers_cache': cache_dir
    }
    
    config_file = project_root / "model_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 設定を保存しました: {config_file}")
    print()
    
    # 既存のモデルを確認
    existing_models = []
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                existing_models.append(item.name)
    
    if existing_models:
        print("=== 既存のモデル ===")
        for model in existing_models:
            print(f"• {model}")
    else:
        print("既存のモデルはありません。")
    
    print()
    print("=== 使用方法 ===")
    print("1. 環境変数を設定:")
    print(f"   Windows: set HF_HOME={cache_dir}")
    print(f"   Linux/Mac: export HF_HOME={cache_dir}")
    print()
    print("2. モデルをダウンロード:")
    print("   from transformers import AutoTokenizer, AutoModelForCausalLM")
    print("   model_name = 'Qwen/Qwen2.5-14B-Instruct'")
    print("   tokenizer = AutoTokenizer.from_pretrained(model_name)")
    print("   model = AutoModelForCausalLM.from_pretrained(model_name)")
    print()
    print("3. または、download_model.pyスクリプトを使用")
    
    return str(models_dir)

def download_specific_model(model_name):
    """特定のモデルをダウンロード"""
    print(f"\n=== モデルダウンロード: {model_name} ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=os.environ.get('HF_HOME', './models')
        )
        print("✅ トークナイザーのダウンロード完了")
        
        print("モデルをダウンロード中...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=os.environ.get('HF_HOME', './models')
        )
        print("✅ モデルのダウンロード完了")
        
        return True
        
    except Exception as e:
        print(f"❌ ダウンロードに失敗しました: {e}")
        return False

if __name__ == "__main__":
    models_dir = setup_model_download()
    
    # 利用可能なモデルから選択してダウンロード
    print("\n=== モデルダウンロード ===")
    available_models = [
        "Qwen/Qwen2.5-14B-Instruct",
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "cyberagent/calm3-22b-chat"
    ]
    
    print("ダウンロード可能なモデル:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    choice = input("\nダウンロードするモデルの番号を入力してください (Enterでスキップ): ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_models):
            download_specific_model(available_models[idx])
        else:
            print("無効な選択です。")
    else:
        print("ダウンロードをスキップしました。") 