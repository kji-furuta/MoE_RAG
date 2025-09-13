#!/usr/bin/env python3
"""
継続学習タスクを削除するスクリプト
不要なタスクとその関連ファイルを完全に削除します
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

def load_tasks_state(file_path: str) -> dict:
    """タスク状態ファイルを読み込む"""
    if not os.path.exists(file_path):
        print(f"タスク状態ファイルが見つかりません: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_tasks_state(file_path: str, tasks_state: dict):
    """タスク状態ファイルを保存"""
    # バックアップを作成
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        print(f"バックアップを作成しました: {backup_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(tasks_state, f, indent=2, ensure_ascii=False)

def delete_task_files(task_id: str, base_dir: str = "/workspace"):
    """タスク関連ファイルを削除"""
    deleted_files = []
    
    # EWCデータの削除
    ewc_files = [
        f"{base_dir}/outputs/ewc_data/fisher_{task_id}.pt",
        f"{base_dir}/outputs/ewc_data/optimal_params_{task_id}.pt",
    ]
    
    for file_path in ewc_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_files.append(file_path)
            print(f"削除: {file_path}")
    
    # タスクモデルディレクトリの削除
    model_dir = f"{base_dir}/outputs/continual_{task_id}"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        deleted_files.append(model_dir)
        print(f"削除: {model_dir}")
    
    # GGUFファイルの削除
    models_dir = Path(f"{base_dir}/models")
    if models_dir.exists():
        for gguf_file in models_dir.glob(f"{task_id}_*.gguf"):
            gguf_file.unlink()
            deleted_files.append(str(gguf_file))
            print(f"削除: {gguf_file}")
    
    # Ollamaモデルの削除
    import subprocess
    try:
        # Ollamaから登録済みモデルを削除
        model_name = f"{task_id}_deepseek-32b-finetuned:latest"
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Ollamaモデルを削除: {model_name}")
        elif "not found" not in result.stderr:
            print(f"Ollamaモデル削除エラー: {result.stderr}")
    except Exception as e:
        print(f"Ollamaモデル削除をスキップ: {e}")
    
    return deleted_files

def update_task_history(task_id: str, base_dir: str = "/workspace"):
    """タスク履歴を更新"""
    history_file = f"{base_dir}/outputs/ewc_data/task_history.json"
    
    if not os.path.exists(history_file):
        print(f"タスク履歴ファイルが見つかりません: {history_file}")
        return
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # タスクを履歴から削除
    original_count = len(history.get('tasks', []))
    history['tasks'] = [t for t in history.get('tasks', []) if t.get('task_id') != task_id]
    removed_count = original_count - len(history['tasks'])
    
    if removed_count > 0:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"タスク履歴から{removed_count}件のエントリを削除しました")

def list_tasks(tasks_state: dict):
    """タスク一覧を表示"""
    if not tasks_state:
        print("タスクが見つかりません")
        return
    
    print("\n=== 継続学習タスク一覧 ===")
    for task_id, task_info in tasks_state.items():
        status = task_info.get('status', 'unknown')
        created_at = task_info.get('created_at', 'N/A')
        model_name = task_info.get('model_name', 'N/A')
        
        print(f"\nタスクID: {task_id}")
        print(f"  状態: {status}")
        print(f"  作成日時: {created_at}")
        print(f"  モデル: {model_name}")
        
        if 'training_config' in task_info:
            config = task_info['training_config']
            print(f"  エポック数: {config.get('num_epochs', 'N/A')}")
            print(f"  学習率: {config.get('learning_rate', 'N/A')}")

def delete_tasks(task_ids: List[str], base_dir: str = "/workspace", dry_run: bool = False):
    """複数のタスクを削除"""
    tasks_state_file = f"{base_dir}/data/continual_learning/tasks_state.json"
    tasks_state = load_tasks_state(tasks_state_file)
    
    if not tasks_state:
        print("タスクが見つかりません")
        return
    
    for task_id in task_ids:
        if task_id not in tasks_state:
            print(f"タスクが見つかりません: {task_id}")
            continue
        
        print(f"\n=== タスク {task_id} を削除中 ===")
        
        if dry_run:
            print("(Dry run - 実際には削除されません)")
        else:
            # ファイルを削除
            deleted_files = delete_task_files(task_id, base_dir)
            
            # タスク履歴を更新
            update_task_history(task_id, base_dir)
            
            # タスク状態から削除
            del tasks_state[task_id]
            
            print(f"タスク {task_id} を削除しました（{len(deleted_files)}個のファイル/ディレクトリ）")
    
    if not dry_run and task_ids:
        # 更新された状態を保存
        save_tasks_state(tasks_state_file, tasks_state)
        print(f"\nタスク状態ファイルを更新しました: {tasks_state_file}")

def main():
    parser = argparse.ArgumentParser(description="継続学習タスクを削除")
    parser.add_argument('--list', action='store_true', help='タスク一覧を表示')
    parser.add_argument('--delete', nargs='+', help='削除するタスクID（複数指定可能）')
    parser.add_argument('--delete-all', action='store_true', help='すべてのタスクを削除')
    parser.add_argument('--dry-run', action='store_true', help='実際には削除せず、削除対象を表示')
    parser.add_argument('--base-dir', default='/workspace', help='ベースディレクトリ（デフォルト: /workspace）')
    
    args = parser.parse_args()
    
    tasks_state_file = f"{args.base_dir}/data/continual_learning/tasks_state.json"
    tasks_state = load_tasks_state(tasks_state_file)
    
    if args.list:
        list_tasks(tasks_state)
    elif args.delete:
        delete_tasks(args.delete, args.base_dir, args.dry_run)
    elif args.delete_all:
        if not args.dry_run:
            response = input("すべてのタスクを削除しますか？ (yes/no): ")
            if response.lower() != 'yes':
                print("キャンセルしました")
                return
        
        all_task_ids = list(tasks_state.keys())
        delete_tasks(all_task_ids, args.base_dir, args.dry_run)
    else:
        # デフォルト: タスク一覧を表示
        list_tasks(tasks_state)
        print("\n使用方法:")
        print("  --list              : タスク一覧を表示")
        print("  --delete task_001   : 特定のタスクを削除")
        print("  --delete-all        : すべてのタスクを削除")
        print("  --dry-run           : 削除せずに対象を表示")

if __name__ == "__main__":
    main()