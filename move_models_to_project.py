#!/usr/bin/env python3
"""
既存のモデルをプロジェクト内に移動するスクリプト
"""

import os
import shutil
import json
from pathlib import Path

def move_models_to_project():
    """既存のモデルをプロジェクト内に移動"""
    print("=== モデル移動スクリプト ===")
    print()
    
    # パス設定
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    cache_dir = Path("/home/kjifu/.cache/huggingface/hub")
    
    print(f"プロジェクトルート: {project_root}")
    print(f"モデルディレクトリ: {models_dir}")
    print(f"キャッシュディレクトリ: {cache_dir}")
    print()
    
    # modelsディレクトリを作成
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
        print(f"✅ モデルディレクトリを作成しました: {models_dir}")
    
    # 利用可能なモデルを確認
    available_models = [
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "cyberagent/calm3-22b-chat",
        "Qwen/Qwen2.5-32B-Instruct"
    ]
    
    moved_models = []
    
    for model_name in available_models:
        # キャッシュ内のモデルパス
        cache_model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
        
        if cache_model_dir.exists():
            print(f"移動中: {model_name}")
            print(f"  元: {cache_model_dir}")
            
            # プロジェクト内のモデルパス
            project_model_dir = models_dir / model_name.replace('/', '/')
            
            try:
                # モデルをコピー
                if project_model_dir.exists():
                    shutil.rmtree(project_model_dir)
                
                shutil.copytree(cache_model_dir, project_model_dir)
                moved_models.append(model_name)
                print(f"  ✅ 移動完了: {project_model_dir}")
                
            except Exception as e:
                print(f"  ❌ 移動失敗: {e}")
        else:
            print(f"❌ モデルが見つかりません: {model_name}")
    
    print()
    print("=== 移動結果 ===")
    print(f"移動済み: {len(moved_models)}/{len(available_models)}")
    
    for model in moved_models:
        print(f"• {model}")
    
    # 環境変数を設定
    cache_dir_str = str(models_dir)
    os.environ['HF_HOME'] = cache_dir_str
    os.environ['TRANSFORMERS_CACHE'] = cache_dir_str
    
    print()
    print("✅ 環境変数を設定しました:")
    print(f"HF_HOME={cache_dir_str}")
    print(f"TRANSFORMERS_CACHE={cache_dir_str}")
    
    # 設定をファイルに保存
    config = {
        'models_directory': str(models_dir),
        'cache_directory': cache_dir_str,
        'moved_models': moved_models
    }
    
    config_file = project_root / "model_migration_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 設定を保存しました: {config_file}")
    
    return moved_models

def create_symlink():
    """シンボリックリンクを作成（オプション）"""
    print("\n=== シンボリックリンク作成 ===")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    cache_dir = Path("/home/kjifu/.cache/huggingface/hub")
    
    # 既存のキャッシュをバックアップ
    backup_dir = cache_dir.parent / "hub_backup"
    if not backup_dir.exists():
        shutil.move(cache_dir, backup_dir)
        print(f"✅ 既存キャッシュをバックアップ: {backup_dir}")
    
    # プロジェクト内のmodelsをキャッシュとしてシンボリックリンク
    try:
        os.symlink(models_dir, cache_dir)
        print(f"✅ シンボリックリンクを作成: {models_dir} -> {cache_dir}")
        return True
    except Exception as e:
        print(f"❌ シンボリックリンク作成失敗: {e}")
        return False

if __name__ == "__main__":
    print("モデル移動を開始しますか？ (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        moved_models = move_models_to_project()
        
        print("\nシンボリックリンクを作成しますか？ (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            create_symlink()
    else:
        print("移動をキャンセルしました。") 