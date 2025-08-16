"""
モデルパス解決器
ファインチューニング済みモデルの動的検出とパス解決
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class ModelPathResolver:
    """モデルパス解決器"""
    
    def __init__(self, base_output_dir: str = "./outputs"):
        """
        Args:
            base_output_dir: ファインチューニング済みモデルの基本ディレクトリ
        """
        self.base_output_dir = Path(base_output_dir)
        
    def find_latest_model(self, model_type: Optional[str] = None) -> Optional[str]:
        """最新のファインチューニング済みモデルを検索
        
        Args:
            model_type: モデルタイプ（lora, qlora, フルファインチューニング等）
            
        Returns:
            最新モデルのパス（存在しない場合はNone）
        """
        
        if not self.base_output_dir.exists():
            logger.warning(f"Output directory not found: {self.base_output_dir}")
            return None
            
        # モデルディレクトリを検索
        model_dirs = self._find_model_directories()
        
        if not model_dirs:
            logger.warning("No fine-tuned models found")
            return None
            
        # モデルタイプでフィルタリング
        if model_type:
            filtered_dirs = [d for d in model_dirs if model_type.lower() in d['name'].lower()]
            if filtered_dirs:
                model_dirs = filtered_dirs
            else:
                logger.warning(f"No models found for type: {model_type}")
                
        # 最新のモデルを選択（作成時刻順）
        latest_model = max(model_dirs, key=lambda x: x['created_time'])
        
        logger.info(f"Found latest model: {latest_model['name']} ({latest_model['path']})")
        return str(latest_model['path'])
        
    def _find_model_directories(self) -> List[Dict[str, Any]]:
        """モデルディレクトリを検索してメタデータと共に返す"""
        
        model_dirs = []
        
        for item in self.base_output_dir.iterdir():
            if not item.is_dir():
                continue
                
            # 隠しディレクトリをスキップ
            if item.name.startswith('.'):
                continue
                
            # モデルファイルが存在するかチェック
            if self._is_valid_model_directory(item):
                # 作成時刻を取得
                created_time = self._extract_creation_time(item.name)
                if created_time is None:
                    created_time = datetime.fromtimestamp(item.stat().st_mtime)
                    
                model_info = {
                    'name': item.name,
                    'path': item,
                    'created_time': created_time,
                    'model_type': self._detect_model_type(item.name),
                    'size': self._calculate_directory_size(item)
                }
                
                model_dirs.append(model_info)
                
        return model_dirs
        
    def _is_valid_model_directory(self, dir_path: Path) -> bool:
        """ディレクトリが有効なモデルディレクトリかチェック"""
        
        # 必須ファイルのパターン（LoRA/QLoRA）
        lora_patterns = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        # 必須ファイルのパターン（フルファインチューニング）
        full_patterns = [
            "config.json",
            "pytorch_model.bin"
        ]
        
        # その他のモデルファイル
        other_patterns = [
            "model.safetensors",
            "pytorch_model.safetensors",
            "adapter_model.bin"
        ]
        
        # LoRAモデルのチェック
        lora_files = sum(1 for pattern in lora_patterns if any(dir_path.glob(pattern)))
        if lora_files >= 1:  # adapter_config.json または adapter_model.safetensors があればOK
            return True
        
        # フルファインチューニングモデルのチェック
        full_files = sum(1 for pattern in full_patterns if any(dir_path.glob(pattern)))
        if full_files >= 1:  # config.json または pytorch_model.bin があればOK
            return True
            
        # その他のモデルファイル
        for pattern in other_patterns:
            if any(dir_path.glob(pattern)):
                return True
                
        return False
        
    def _extract_creation_time(self, dir_name: str) -> Optional[datetime]:
        """ディレクトリ名から作成時刻を抽出"""
        
        # タイムスタンプパターン（例: 20250725_061715）
        timestamp_patterns = [
            r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
            r'(\d{14})',       # YYYYMMDDHHMMSS
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, dir_name)
            if match:
                timestamp_str = match.group(1)
                
                try:
                    # 形式に応じて解析
                    if '_' in timestamp_str and len(timestamp_str) == 15:  # YYYYMMDD_HHMMSS
                        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    elif len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                        return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    elif '-' in timestamp_str:  # YYYY-MM-DD_HH-MM-SS
                        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                        
                except ValueError:
                    continue
                    
        return None
        
    def _detect_model_type(self, dir_name: str) -> str:
        """ディレクトリ名からモデルタイプを推定"""
        
        dir_name_lower = dir_name.lower()
        
        if 'qlora' in dir_name_lower or '4bit' in dir_name_lower:
            return 'qlora'
        elif 'lora' in dir_name_lower:
            return 'lora'  
        elif 'フルファインチューニング' in dir_name or 'full' in dir_name_lower:
            return 'full_finetuning'
        else:
            return 'unknown'
            
    def _calculate_directory_size(self, dir_path: Path) -> int:
        """ディレクトリのサイズを計算（バイト）"""
        
        total_size = 0
        try:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except OSError:
            pass
            
        return total_size
        
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """モデルの詳細情報を取得"""
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {'exists': False}
            
        info = {
            'exists': True,
            'path': str(model_path),
            'name': model_path.name,
            'model_type': self._detect_model_type(model_path.name),
            'size': self._calculate_directory_size(model_path),
            'created_time': datetime.fromtimestamp(model_path.stat().st_mtime),
            'files': []
        }
        
        # 含まれるファイル一覧  
        try:
            for file_path in model_path.iterdir():
                if file_path.is_file():
                    info['files'].append({
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
        except OSError as e:
            logger.warning(f"Could not list files in {model_path}: {e}")
            
        return info
        
    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """モデルパスの有効性を検証"""
        
        model_path = Path(model_path)
        
        result = {
            'is_valid': False,
            'exists': model_path.exists(),
            'is_directory': model_path.is_dir() if model_path.exists() else False,
            'has_model_files': False,
            'model_type': 'unknown',
            'issues': []
        }
        
        if not result['exists']:
            result['issues'].append(f"Path does not exist: {model_path}")
            return result
            
        if not result['is_directory']:
            result['issues'].append(f"Path is not a directory: {model_path}")
            return result
            
        # モデルファイルの存在チェック
        result['has_model_files'] = self._is_valid_model_directory(model_path)
        if not result['has_model_files']:
            result['issues'].append("No valid model files found")
            
        # モデルタイプの検出
        result['model_type'] = self._detect_model_type(model_path.name)
        
        # 総合的な有効性判定
        result['is_valid'] = result['exists'] and result['is_directory'] and result['has_model_files']
        
        return result
        
    def create_latest_symlink(self, target_path: Optional[str] = None) -> Optional[str]:
        """最新モデルへのシンボリックリンクを作成"""
        
        if target_path is None:
            target_path = self.find_latest_model()
            
        if target_path is None:
            logger.warning("No model found to create symlink")
            return None
            
        latest_link = self.base_output_dir / "latest"
        
        try:
            # 既存のシンボリックリンクを削除
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
                
            # 新しいシンボリックリンクを作成（相対パスで）
            target_relative = Path(target_path).relative_to(self.base_output_dir)
            latest_link.symlink_to(target_relative)
            
            logger.info(f"Created symlink: {latest_link} -> {target_relative}")
            return str(latest_link)
            
        except OSError as e:
            logger.error(f"Failed to create symlink: {e}")
            return None
            
    def list_available_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧を取得"""
        
        models = self._find_model_directories()
        
        # 作成時刻でソート（新しい順）
        models.sort(key=lambda x: x['created_time'], reverse=True)
        
        # 情報を整理
        for model in models:
            model['created_time_str'] = model['created_time'].strftime('%Y-%m-%d %H:%M:%S')
            model['size_mb'] = round(model['size'] / (1024 * 1024), 1)
            model['path_str'] = str(model['path'])
            
        return models


def resolve_model_path(config_path: str, 
                      preferred_type: Optional[str] = None,
                      base_output_dir: str = "./outputs") -> str:
    """モデルパスを解決（便利関数）
    
    Args:
        config_path: 設定で指定されたパス
        preferred_type: 優先するモデルタイプ
        base_output_dir: 出力ディレクトリ
        
    Returns:
        解決されたモデルパス
    """
    
    resolver = ModelPathResolver(base_output_dir)
    
    # 設定パスが有効かチェック
    validation = resolver.validate_model_path(config_path)
    
    if validation['is_valid']:
        logger.info(f"Using configured model path: {config_path}")
        return config_path
        
    # 設定パスが無効な場合は自動検出
    logger.warning(f"Configured path invalid: {config_path}")
    logger.info("Attempting to find alternative model...")
    
    detected_path = resolver.find_latest_model(preferred_type)
    
    if detected_path:
        logger.info(f"Using detected model: {detected_path}")
        
        # latest シンボリックリンクを作成/更新
        resolver.create_latest_symlink(detected_path)
        
        return detected_path
    else:
        logger.error("No valid model found")
        raise FileNotFoundError(f"No valid model found. Configured path: {config_path}")


def get_model_resolver(base_output_dir: str = "./outputs") -> ModelPathResolver:
    """モデルパス解決器を取得（便利関数）"""
    return ModelPathResolver(base_output_dir)