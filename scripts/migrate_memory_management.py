#!/usr/bin/env python3
"""
メモリ管理システムマイグレーションスクリプト
既存のコードを新しい統一メモリ管理システムに移行
"""

import os
import sys
import logging
from pathlib import Path
import shutil
from datetime import datetime
import re

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

from src.core.memory_manager import get_memory_manager
from src.core.quantization_manager import get_quantization_config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backup_file(filepath: Path) -> Path:
    """ファイルをバックアップ"""
    backup_dir = filepath.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
    
    shutil.copy2(filepath, backup_path)
    logger.info(f"Backed up: {filepath} -> {backup_path}")
    
    return backup_path


def update_memory_optimized_loader():
    """memory_optimized_loader.pyを更新"""
    filepath = Path("app/memory_optimized_loader.py")
    
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return
    
    # バックアップを作成
    backup_file(filepath)
    
    # 新しいバージョンへのシンボリックリンクを作成（Windowsの場合はコピー）
    new_filepath = Path("app/memory_optimized_loader_v2.py")
    
    if os.name == 'nt':
        # Windows: コピー
        shutil.copy2(new_filepath, filepath)
        logger.info(f"Copied: {new_filepath} -> {filepath}")
    else:
        # Unix/Linux: シンボリックリンク
        filepath.unlink()
        filepath.symlink_to(new_filepath)
        logger.info(f"Created symlink: {filepath} -> {new_filepath}")


def update_import_statements():
    """インポート文を更新"""
    files_to_update = [
        "app/main_unified.py",
        "app/routers/finetuning.py",
        "app/routers/continual.py",
        "src/training/lora_finetuning.py",
        "src/training/full_finetuning.py",
        "src/training/ewc_full_finetuning.py"
    ]
    
    old_imports = [
        (r"from app\.memory_optimized_loader import", 
         "from app.memory_optimized_loader_v2 import"),
        (r"import app\.memory_optimized_loader",
         "import app.memory_optimized_loader_v2 as memory_optimized_loader"),
        (r"from memory_optimized_loader import",
         "from app.memory_optimized_loader_v2 import"),
    ]
    
    for file_path in files_to_update:
        filepath = Path(file_path)
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        
        # ファイルを読み込み
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # インポート文を更新
        for old_pattern, new_import in old_imports:
            content = re.sub(old_pattern, new_import, content)
        
        # 変更があった場合のみ更新
        if content != original_content:
            backup_file(filepath)
            filepath.write_text(content, encoding='utf-8')
            logger.info(f"Updated imports in: {filepath}")


def update_environment_variables():
    """環境変数の設定を更新"""
    files_to_update = [
        "app/main_unified.py",
        "app/memory_optimized_loader.py",
        "src/training/lora_finetuning.py",
        "src/training/full_finetuning.py"
    ]
    
    # 削除すべき環境変数設定
    old_env_patterns = [
        r'os\.environ\["CUDA_LAUNCH_BLOCKING"\]\s*=\s*"1"',
        r'os\.environ\["PYTORCH_CUDA_ALLOC_CONF"\]\s*=\s*"[^"]*"',
    ]
    
    for file_path in files_to_update:
        filepath = Path(file_path)
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # 古い環境変数設定を削除
        for pattern in old_env_patterns:
            content = re.sub(pattern, "# Removed: Environment variable now managed by memory_manager", content)
        
        if content != original_content:
            backup_file(filepath)
            filepath.write_text(content, encoding='utf-8')
            logger.info(f"Updated environment variables in: {filepath}")


def update_quantization_config_usage():
    """量子化設定の使用箇所を更新"""
    
    # get_optimal_quantization_config関数の置き換え
    files_to_check = [
        "app/main_unified.py",
        "app/model_utils.py",
        "src/training/lora_finetuning.py"
    ]
    
    for file_path in files_to_check:
        filepath = Path(file_path)
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # 関数呼び出しの更新
        patterns = [
            (r"get_optimal_quantization_config\(",
             "get_memory_manager().get_optimal_quantization("),
            (r"BitsAndBytesConfig\(",
             "UnifiedQuantizationConfig("),
        ]
        
        for old_pattern, new_pattern in patterns:
            content = re.sub(old_pattern, new_pattern, content)
        
        # 必要なインポートを追加
        if "get_memory_manager" in content and "from src.core.memory_manager import" not in content:
            # インポート部分を見つけて追加
            import_section = re.search(r"(import .*?\n)+", content)
            if import_section:
                insert_pos = import_section.end()
                new_import = "from src.core.memory_manager import get_memory_manager\n"
                content = content[:insert_pos] + new_import + content[insert_pos:]
        
        if content != original_content:
            backup_file(filepath)
            filepath.write_text(content, encoding='utf-8')
            logger.info(f"Updated quantization config usage in: {filepath}")


def create_migration_report():
    """マイグレーションレポートを作成"""
    report_path = Path("migration_report.md")
    
    report_content = f"""# メモリ管理システムマイグレーションレポート

実行日時: {datetime.now().isoformat()}

## 実行された変更

### 1. メモリ最適化ローダーの更新
- `app/memory_optimized_loader.py` -> `app/memory_optimized_loader_v2.py`
- 新しい統一メモリ管理システムを使用

### 2. インポート文の更新
- すべての関連ファイルで新しいモジュールをインポート

### 3. 環境変数の管理
- `CUDA_LAUNCH_BLOCKING=1` を削除（本番環境用）
- メモリ管理が `src.core.memory_manager` で一元化

### 4. 量子化設定の統一
- `BitsAndBytesConfig` の直接使用を `UnifiedQuantizationConfig` に置き換え
- モデルサイズとGPUメモリに基づく自動設定

## 新機能

### メモリマネージャー (`src.core.memory_manager`)
- GPUメモリの監視とレポート
- 動的な量子化設定の選択
- メモリクリア機能の改善

### 量子化マネージャー (`src.core.quantization_manager`)
- プリセット設定のサポート
- 設定の保存と読み込み
- モデルごとの最適化

## 使用方法

```python
from src.core.memory_manager import get_memory_manager
from app.memory_optimized_loader_v2 import create_model_loader

# メモリマネージャーの初期化（本番環境）
memory_manager = get_memory_manager(debug_mode=False)

# モデルローダーの作成
loader = create_model_loader()

# モデルの読み込み（自動最適化）
model, tokenizer = loader.load_base_model(
    "model_name",
    for_training=True
)

# メモリ状況の監視
status = memory_manager.monitor_memory_usage()
print(status)
```

## バックアップ

すべての変更されたファイルは `backups/` ディレクトリにバックアップされています。

## 注意事項

1. **環境変数**: `CUDA_LAUNCH_BLOCKING=1` は削除されました。デバッグが必要な場合は `debug_mode=True` を使用してください。

2. **量子化設定**: 自動設定が推奨されますが、必要に応じてプリセットを使用できます。

3. **メモリ監視**: 新しいシステムは継続的なメモリ監視を提供します。

## トラブルシューティング

問題が発生した場合：
1. バックアップファイルから復元
2. `debug_mode=True` で詳細ログを確認
3. `memory_manager.monitor_memory_usage()` でメモリ状況を確認

"""
    
    report_path.write_text(report_content, encoding='utf-8')
    logger.info(f"Migration report created: {report_path}")


def verify_migration():
    """マイグレーションの検証"""
    logger.info("Verifying migration...")
    
    try:
        # メモリマネージャーのテスト
        memory_manager = get_memory_manager(debug_mode=False)
        memory_status = memory_manager.monitor_memory_usage()
        logger.info(f"Memory manager working: {memory_status}")
        
        # 量子化マネージャーのテスト
        quant_manager = get_quantization_config_manager()
        presets = quant_manager.list_saved_configs()
        logger.info(f"Quantization manager working. Saved configs: {presets}")
        
        # 新しいローダーのインポートテスト
        from app.memory_optimized_loader_v2 import create_model_loader
        loader = create_model_loader()
        logger.info("New model loader imported successfully")
        
        logger.info("Migration verification successful!")
        return True
        
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False


def main():
    """メインのマイグレーション処理"""
    logger.info("Starting memory management system migration...")
    
    # 1. メモリ最適化ローダーの更新
    logger.info("Step 1: Updating memory optimized loader...")
    update_memory_optimized_loader()
    
    # 2. インポート文の更新
    logger.info("Step 2: Updating import statements...")
    update_import_statements()
    
    # 3. 環境変数の更新
    logger.info("Step 3: Updating environment variables...")
    update_environment_variables()
    
    # 4. 量子化設定の更新
    logger.info("Step 4: Updating quantization config usage...")
    update_quantization_config_usage()
    
    # 5. マイグレーションレポートの作成
    logger.info("Step 5: Creating migration report...")
    create_migration_report()
    
    # 6. 検証
    logger.info("Step 6: Verifying migration...")
    success = verify_migration()
    
    if success:
        logger.info("Migration completed successfully!")
        logger.info("Please review the migration_report.md for details.")
    else:
        logger.error("Migration completed with errors. Please check the logs.")
        logger.info("You can restore from backups if needed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
