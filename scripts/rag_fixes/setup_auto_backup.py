#!/usr/bin/env python3
"""
自動バックアップシステムの実装
定期的なバックアップとクリーンアップを管理
"""

import os
import sys
import time
import json
import schedule
import threading
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, "/workspace" if os.path.exists("/workspace") else ".")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoBackupManager:
    """自動バックアップマネージャー"""
    
    def __init__(self, config_path: str = "./config/backup_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 永続化アダプターの初期化
        from scripts.rag_fixes.integrate_persistence import PersistentRAGAdapter
        self.adapter = PersistentRAGAdapter()
        
        # バックアップ統計
        self.stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'last_backup': None,
            'total_size_bytes': 0
        }
        
        # スケジューラーの設定
        self._setup_schedule()
        
        logger.info(f"AutoBackupManager initialized with config: {config_path}")
    
    def _load_config(self) -> dict:
        """設定ファイルの読み込み"""
        default_config = {
            'enabled': True,
            'backup_schedule': {
                'hourly': False,
                'daily': True,
                'weekly': True,
                'time': "03:00"  # 毎日3時
            },
            'retention': {
                'keep_daily': 7,    # 7日分の日次バックアップ
                'keep_weekly': 4,   # 4週分の週次バックアップ
                'keep_monthly': 3   # 3ヶ月分の月次バックアップ
            },
            'incremental': {
                'enabled': True,
                'threshold': 100  # 100文書ごとに増分バックアップ
            },
            'compression': {
                'enabled': True,
                'format': 'gzip'
            },
            'notification': {
                'enabled': False,
                'webhook_url': None
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # デフォルト設定とマージ
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        else:
            # デフォルト設定を保存
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_schedule(self):
        """スケジュールの設定"""
        if not self.config['enabled']:
            logger.info("Auto backup is disabled")
            return
        
        schedule_config = self.config['backup_schedule']
        
        # 日次バックアップ
        if schedule_config.get('daily'):
            schedule.every().day.at(schedule_config['time']).do(
                self._run_backup, backup_type='daily'
            )
            logger.info(f"Daily backup scheduled at {schedule_config['time']}")
        
        # 週次バックアップ（日曜日）
        if schedule_config.get('weekly'):
            schedule.every().sunday.at(schedule_config['time']).do(
                self._run_backup, backup_type='weekly'
            )
            logger.info(f"Weekly backup scheduled on Sunday at {schedule_config['time']}")
        
        # 時間ごとバックアップ
        if schedule_config.get('hourly'):
            schedule.every().hour.do(
                self._run_backup, backup_type='hourly'
            )
            logger.info("Hourly backup scheduled")
    
    def _run_backup(self, backup_type: str = 'manual'):
        """バックアップの実行"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{backup_type}_{timestamp}"
            
            logger.info(f"Starting {backup_type} backup: {backup_name}")
            
            # バックアップ実行
            backup_path = self.adapter.create_backup(backup_name)
            
            # 統計更新
            self.stats['total_backups'] += 1
            self.stats['successful_backups'] += 1
            self.stats['last_backup'] = datetime.now().isoformat()
            
            # バックアップサイズの計算
            backup_dir = Path(backup_path)
            if backup_dir.exists():
                size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
                self.stats['total_size_bytes'] += size
                
                # 圧縮が有効な場合
                if self.config['compression']['enabled']:
                    self._compress_backup(backup_dir)
            
            logger.info(f"Backup completed successfully: {backup_name}")
            
            # クリーンアップ
            self._cleanup_old_backups()
            
            # 通知
            if self.config['notification']['enabled']:
                self._send_notification(f"Backup completed: {backup_name}", 'success')
            
            return True
            
        except Exception as e:
            self.stats['failed_backups'] += 1
            logger.error(f"Backup failed: {e}")
            
            if self.config['notification']['enabled']:
                self._send_notification(f"Backup failed: {e}", 'error')
            
            return False
    
    def _compress_backup(self, backup_dir: Path):
        """バックアップの圧縮"""
        import gzip
        import tarfile
        
        try:
            format = self.config['compression']['format']
            
            if format == 'gzip':
                # tar.gzファイルの作成
                archive_path = backup_dir.with_suffix('.tar.gz')
                
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(backup_dir, arcname=backup_dir.name)
                
                # 元のディレクトリを削除
                import shutil
                shutil.rmtree(backup_dir)
                
                logger.info(f"Backup compressed: {archive_path}")
                
        except Exception as e:
            logger.warning(f"Failed to compress backup: {e}")
    
    def _cleanup_old_backups(self):
        """古いバックアップのクリーンアップ"""
        retention = self.config['retention']
        
        try:
            # バックアップディレクトリの取得
            backup_base = Path("data/rag_persistent/backups")
            if not backup_base.exists():
                return
            
            now = datetime.now()
            
            # バックアップファイルの分類
            daily_backups = []
            weekly_backups = []
            monthly_backups = []
            
            for backup in backup_base.iterdir():
                if backup.is_dir() or backup.suffix in ['.gz', '.tar']:
                    # バックアップの日時を取得
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    age_days = (now - mtime).days
                    
                    # バックアップタイプの判定
                    if 'daily' in backup.name:
                        daily_backups.append((backup, mtime, age_days))
                    elif 'weekly' in backup.name:
                        weekly_backups.append((backup, mtime, age_days))
                    elif 'monthly' in backup.name:
                        monthly_backups.append((backup, mtime, age_days))
            
            # 古いバックアップの削除
            self._remove_old_backups(daily_backups, retention['keep_daily'], 'daily')
            self._remove_old_backups(weekly_backups, retention['keep_weekly'] * 7, 'weekly')
            self._remove_old_backups(monthly_backups, retention['keep_monthly'] * 30, 'monthly')
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _remove_old_backups(self, backups: list, keep_days: int, backup_type: str):
        """指定された期間より古いバックアップを削除"""
        import shutil
        
        for backup_path, mtime, age_days in backups:
            if age_days > keep_days:
                try:
                    if backup_path.is_dir():
                        shutil.rmtree(backup_path)
                    else:
                        backup_path.unlink()
                    
                    logger.info(f"Removed old {backup_type} backup: {backup_path.name} ({age_days} days old)")
                    
                except Exception as e:
                    logger.warning(f"Failed to remove {backup_path}: {e}")
    
    def _send_notification(self, message: str, level: str = 'info'):
        """通知の送信"""
        if not self.config['notification']['enabled']:
            return
        
        webhook_url = self.config['notification'].get('webhook_url')
        if not webhook_url:
            return
        
        try:
            import requests
            
            payload = {
                'text': f"[RAG Backup {level.upper()}] {message}",
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code != 200:
                logger.warning(f"Notification failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    def start_scheduler(self):
        """スケジューラーを開始"""
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1分ごとにチェック
        
        # バックグラウンドスレッドで実行
        scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
        scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def get_status(self) -> dict:
        """バックアップシステムの状態を取得"""
        return {
            'enabled': self.config['enabled'],
            'stats': self.stats,
            'next_backup': self._get_next_backup_time(),
            'config': self.config
        }
    
    def _get_next_backup_time(self) -> str:
        """次のバックアップ時刻を取得"""
        jobs = schedule.get_jobs()
        if jobs:
            # 次の実行時刻を計算（簡略版）
            return self.config['backup_schedule']['time']
        return "Not scheduled"
    
    def perform_manual_backup(self) -> bool:
        """手動バックアップの実行"""
        return self._run_backup('manual')
    
    def restore_latest_backup(self) -> bool:
        """最新のバックアップから復元"""
        backup_base = Path("data/rag_persistent/backups")
        
        if not backup_base.exists():
            logger.error("No backup directory found")
            return False
        
        # 最新のバックアップを検索
        backups = sorted(
            backup_base.iterdir(),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not backups:
            logger.error("No backups found")
            return False
        
        latest_backup = backups[0]
        logger.info(f"Restoring from: {latest_backup.name}")
        
        return self.adapter.restore_backup(latest_backup.name)


def main():
    """メイン処理"""
    print("=" * 60)
    print("🔧 自動バックアップシステムのセットアップ")
    print("=" * 60)
    
    # 1. バックアップマネージャーの初期化
    print("\n1. バックアップマネージャーの初期化")
    manager = AutoBackupManager()
    print("  ✅ マネージャー初期化完了")
    
    # 2. 現在の状態表示
    print("\n2. 現在の状態")
    status = manager.get_status()
    print(f"  有効: {status['enabled']}")
    print(f"  総バックアップ数: {status['stats']['total_backups']}")
    print(f"  成功: {status['stats']['successful_backups']}")
    print(f"  失敗: {status['stats']['failed_backups']}")
    print(f"  最終バックアップ: {status['stats']['last_backup'] or 'なし'}")
    print(f"  次回バックアップ: {status['next_backup']}")
    
    # 3. テストバックアップの実行
    print("\n3. テストバックアップの実行")
    success = manager.perform_manual_backup()
    if success:
        print("  ✅ テストバックアップ成功")
    else:
        print("  ❌ テストバックアップ失敗")
    
    # 4. スケジューラーの起動
    print("\n4. スケジューラーの起動")
    manager.start_scheduler()
    print("  ✅ スケジューラー起動完了")
    
    print("\n✅ 自動バックアップシステムのセットアップ完了")
    print("\nバックアップスケジュール:")
    config = manager.config['backup_schedule']
    if config['daily']:
        print(f"  日次: 毎日 {config['time']}")
    if config['weekly']:
        print(f"  週次: 毎週日曜日 {config['time']}")
    if config['hourly']:
        print(f"  時間: 毎時")
    
    print("\n保持期間:")
    retention = manager.config['retention']
    print(f"  日次バックアップ: {retention['keep_daily']}日")
    print(f"  週次バックアップ: {retention['keep_weekly']}週")
    print(f"  月次バックアップ: {retention['keep_monthly']}ヶ月")
    
    return manager


if __name__ == "__main__":
    manager = main()
    
    # デーモンモードで実行する場合
    if '--daemon' in sys.argv:
        print("\n🔄 デーモンモードで実行中... (Ctrl+Cで終了)")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 バックアップシステムを終了します")
