#!/bin/bash
"""
ディスク容量管理スクリプト
不要ファイルのクリーンアップと容量確保
"""

import os
import shutil
import psutil
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiskSpaceManager:
    """ディスク容量管理クラス"""
    
    def __init__(self, workspace_path="/home/kjifu/AI_FT_7"):
        self.workspace_path = Path(workspace_path)
        self.min_required_gb = 85  # 32Bモデル用の最小必要容量
        
    def check_disk_usage(self):
        """現在のディスク使用状況を確認"""
        usage = psutil.disk_usage(str(self.workspace_path))
        
        info = {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "percent": usage.percent
        }
        
        logger.info(f"Disk usage for {self.workspace_path}:")
        logger.info(f"  Total: {info['total_gb']:.1f} GB")
        logger.info(f"  Used: {info['used_gb']:.1f} GB ({info['percent']:.1f}%)")
        logger.info(f"  Free: {info['free_gb']:.1f} GB")
        
        return info
    
    def find_large_files(self, min_size_gb=1):
        """大きなファイルを検索"""
        large_files = []
        min_size_bytes = min_size_gb * (1024**3)
        
        for root, dirs, files in os.walk(self.workspace_path):
            # .gitディレクトリをスキップ
            if '.git' in root:
                continue
                
            for file in files:
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    if size > min_size_bytes:
                        large_files.append({
                            "path": str(file_path),
                            "size_gb": size / (1024**3)
                        })
                except:
                    continue
        
        # サイズでソート
        large_files.sort(key=lambda x: x['size_gb'], reverse=True)
        
        logger.info(f"\nFound {len(large_files)} files larger than {min_size_gb} GB:")
        for file_info in large_files[:10]:  # Top 10
            logger.info(f"  {file_info['size_gb']:.2f} GB: {file_info['path']}")
        
        return large_files
    
    def cleanup_cache(self):
        """キャッシュファイルのクリーンアップ"""
        cache_dirs = [
            Path.home() / ".cache/huggingface/hub",
            self.workspace_path / "temp_uploads",
            self.workspace_path / ".pytest_cache",
            Path("/tmp"),
        ]
        
        freed_space = 0
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                logger.info(f"Cleaning cache: {cache_dir}")
                
                # 古いファイルを削除（7日以上）
                try:
                    if cache_dir == Path("/tmp"):
                        # /tmpは慎重に扱う
                        cmd = f"find {cache_dir} -type f -name 'tmp*' -mtime +7 -delete 2>/dev/null"
                    else:
                        cmd = f"find {cache_dir} -type f -mtime +7 -delete 2>/dev/null"
                    
                    subprocess.run(cmd, shell=True)
                    logger.info(f"  ✓ Cleaned old files from {cache_dir}")
                except Exception as e:
                    logger.warning(f"  Failed to clean {cache_dir}: {e}")
        
        return freed_space
    
    def cleanup_old_checkpoints(self):
        """古いチェックポイントの削除"""
        outputs_dir = self.workspace_path / "outputs"
        
        if not outputs_dir.exists():
            return 0
        
        checkpoints = []
        for checkpoint_dir in outputs_dir.glob("*/checkpoint-*"):
            if checkpoint_dir.is_dir():
                # 最終更新時刻を取得
                mtime = checkpoint_dir.stat().st_mtime
                size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
                checkpoints.append({
                    "path": checkpoint_dir,
                    "mtime": mtime,
                    "size_gb": size / (1024**3)
                })
        
        # 古い順にソート
        checkpoints.sort(key=lambda x: x['mtime'])
        
        # 最新の3つを除いて削除
        freed_space = 0
        for checkpoint in checkpoints[:-3]:
            logger.info(f"Removing old checkpoint: {checkpoint['path']} ({checkpoint['size_gb']:.2f} GB)")
            shutil.rmtree(checkpoint['path'])
            freed_space += checkpoint['size_gb']
        
        logger.info(f"Freed {freed_space:.2f} GB from old checkpoints")
        return freed_space
    
    def suggest_alternatives(self):
        """容量不足時の代替案を提案"""
        usage = self.check_disk_usage()
        
        if usage['free_gb'] < self.min_required_gb:
            logger.warning(f"\n⚠️ Insufficient space for full fine-tuning")
            logger.info("\n💡 Recommended alternatives:")
            
            suggestions = [
                "1. Use LoRA/DoRA instead of full fine-tuning:",
                "   - Saves only adapter weights (~200MB vs 64GB)",
                "   - Command: python train_with_lora.py",
                "",
                "2. Use AWQ quantization first:",
                "   - Reduces model size by 75%",
                "   - Command: python scripts/quantize_model.py",
                "",
                "3. Mount external storage:",
                "   - sudo mount /dev/sdb1 /mnt/external",
                "   - ln -s /mnt/external/models /workspace/models_external",
                "",
                "4. Use model sharding:",
                "   - Save with max_shard_size='2GB'",
                "   - Distribute across multiple directories",
            ]
            
            for suggestion in suggestions:
                print(suggestion)
        else:
            logger.info(f"✅ Sufficient space available for training")
    
    def create_efficient_config(self):
        """効率的なトレーニング設定を作成"""
        config = {
            "training": {
                "gradient_checkpointing": True,
                "fp16": True,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            },
            "model_saving": {
                "max_shard_size": "2GB",
                "safe_serialization": True,
            },
            "optimization": {
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
        }
        
        config_path = self.workspace_path / "configs" / "efficient_training.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created efficient config at: {config_path}")
        return config_path


def main():
    """メイン実行"""
    manager = DiskSpaceManager()
    
    print("=" * 70)
    print("💾 Disk Space Management Tool")
    print("=" * 70)
    
    # 1. 現在の状況確認
    usage = manager.check_disk_usage()
    
    # 2. 大きなファイルの検索
    print("\n🔍 Searching for large files...")
    large_files = manager.find_large_files(min_size_gb=1)
    
    # 3. クリーンアップ実行
    if usage['free_gb'] < manager.min_required_gb:
        print(f"\n🧹 Cleaning up (need {manager.min_required_gb} GB, have {usage['free_gb']:.1f} GB)...")
        
        # キャッシュクリーンアップ
        manager.cleanup_cache()
        
        # 古いチェックポイント削除
        manager.cleanup_old_checkpoints()
        
        # 再度確認
        usage = manager.check_disk_usage()
    
    # 4. 代替案の提案
    manager.suggest_alternatives()
    
    # 5. 効率的な設定の作成
    manager.create_efficient_config()
    
    print("\n" + "=" * 70)
    print("✅ Disk management completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
