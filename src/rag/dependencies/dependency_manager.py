"""
RAG システムの依存関係マネージャー

このモジュールは、RAGシステムが必要とする全ての依存関係を管理し、
チェック、インストール、レポート生成などの機能を提供します。
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib
import importlib.metadata
import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging

# ロガーの設定
logger = logging.getLogger(__name__)


class DependencyLevel(Enum):
    """依存関係のレベル"""
    CORE = "core"                    # 必須：これがなければ動作しない
    INFRASTRUCTURE = "infrastructure" # 基盤：基本機能に必要
    OPTIONAL = "optional"            # オプション：拡張機能用


@dataclass
class Dependency:
    """依存関係の定義"""
    name: str                                    # 表示名
    package_name: str                            # Pythonパッケージ名
    level: DependencyLevel                      # 依存レベル
    version_spec: Optional[str] = None          # バージョン指定 (例: ">=1.0.0")
    alternatives: List[str] = field(default_factory=list)  # 代替パッケージ
    check_function: Optional[Callable] = None   # カスタムチェック関数
    install_command: Optional[str] = None       # カスタムインストールコマンド
    description: Optional[str] = None           # 説明
    
    
@dataclass 
class DependencyCheckResult:
    """依存関係チェックの結果"""
    is_satisfied: bool              # 全ての依存関係が満たされているか
    missing_core: List[str]         # 不足しているコア依存関係
    missing_infrastructure: List[str]  # 不足しているインフラ依存関係
    missing_optional: List[str]     # 不足しているオプション依存関係
    warnings: List[str]             # 警告メッセージ
    can_run: bool                   # 最小限の機能で動作可能か
    alternatives_used: Dict[str, str]  # 使用された代替パッケージ
    installed_versions: Dict[str, str]  # インストール済みバージョン
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def missing_dependencies(self) -> List[str]:
        """不足している必須依存関係のリスト"""
        return self.missing_core + self.missing_infrastructure
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'is_satisfied': self.is_satisfied,
            'missing_core': self.missing_core,
            'missing_infrastructure': self.missing_infrastructure,
            'missing_optional': self.missing_optional,
            'warnings': self.warnings,
            'can_run': self.can_run,
            'alternatives_used': self.alternatives_used,
            'installed_versions': self.installed_versions,
            'timestamp': self.timestamp.isoformat()
        }


class RAGDependencyManager:
    """RAGシステムの依存関係を管理"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: キャッシュディレクトリ（チェック結果の保存用）
        """
        self.dependencies = self._define_dependencies()
        self.check_results = {}
        
        # キャッシュディレクトリの設定（権限エラー対策）
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # 複数の候補から書き込み可能なディレクトリを選択
            cache_candidates = [
                Path('/tmp/ai_ft_cache/dependencies'),  # Docker環境用
                Path('/workspace/.cache/ai_ft/dependencies'),  # ワークスペース内
                Path.home() / '.cache' / 'ai_ft' / 'dependencies',  # ホームディレクトリ
            ]
            
            for candidate in cache_candidates:
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    # 書き込みテスト
                    test_file = candidate / '.test'
                    test_file.touch()
                    test_file.unlink()
                    self.cache_dir = candidate
                    break
                except (PermissionError, OSError):
                    continue
            else:
                # どこにも書き込めない場合はメモリ内キャッシュのみ使用
                self.cache_dir = None
                logger.warning("No writable cache directory found, using memory cache only")
        
    def _define_dependencies(self) -> Dict[str, Dependency]:
        """依存関係の定義"""
        return {
            # ===== コア依存関係 =====
            "transformers": Dependency(
                name="Transformers",
                package_name="transformers",
                level=DependencyLevel.CORE,
                version_spec=">=4.30.0",
                description="Hugging Face Transformersライブラリ（LLMの基盤）"
            ),
            "torch": Dependency(
                name="PyTorch", 
                package_name="torch",
                level=DependencyLevel.CORE,
                version_spec=">=2.0.0",
                description="深層学習フレームワーク"
            ),
            "sentence_transformers": Dependency(
                name="Sentence Transformers",
                package_name="sentence_transformers",
                level=DependencyLevel.CORE,
                version_spec=">=2.3.1",
                description="文埋め込みモデル用ライブラリ"
            ),
            "pydantic": Dependency(
                name="Pydantic",
                package_name="pydantic",
                level=DependencyLevel.CORE,
                version_spec=">=2.5.0",
                description="データバリデーション用ライブラリ"
            ),
            
            # ===== インフラストラクチャ依存関係 =====
            "qdrant": Dependency(
                name="Qdrant Client",
                package_name="qdrant_client",
                level=DependencyLevel.INFRASTRUCTURE,
                version_spec=">=1.7.3",
                alternatives=["chromadb", "faiss-cpu"],
                check_function=self._check_qdrant_service,
                description="ベクトルデータベースクライアント"
            ),
            "pdf_processor": Dependency(
                name="PyMuPDF",
                package_name="fitz",
                level=DependencyLevel.INFRASTRUCTURE,
                version_spec=">=1.23.16",
                alternatives=["pdfplumber"],
                install_command="pip install PyMuPDF",
                description="PDF処理用ライブラリ"
            ),
            "pandas": Dependency(
                name="Pandas",
                package_name="pandas",
                level=DependencyLevel.INFRASTRUCTURE,
                version_spec=">=2.1.4",
                description="データ分析ライブラリ"
            ),
            "numpy": Dependency(
                name="NumPy",
                package_name="numpy",
                level=DependencyLevel.INFRASTRUCTURE,
                version_spec=">=1.24.4",
                description="数値計算ライブラリ"
            ),
            "loguru": Dependency(
                name="Loguru",
                package_name="loguru",
                level=DependencyLevel.INFRASTRUCTURE,
                version_spec=">=0.7.2",
                description="ログ管理ライブラリ"
            ),
            
            # ===== オプション依存関係 =====
            "easyocr": Dependency(
                name="EasyOCR",
                package_name="easyocr",
                level=DependencyLevel.OPTIONAL,
                version_spec=">=1.7.1",
                description="OCR（光学文字認識）ライブラリ"
            ),
            "spacy": Dependency(
                name="spaCy",
                package_name="spacy",
                level=DependencyLevel.OPTIONAL,
                version_spec=">=3.7.2",
                check_function=self._check_spacy_model,
                description="自然言語処理ライブラリ"
            ),
            "streamlit": Dependency(
                name="Streamlit",
                package_name="streamlit",
                level=DependencyLevel.OPTIONAL,
                version_spec=">=1.29.0",
                description="Web UI フレームワーク"
            ),
            "plotly": Dependency(
                name="Plotly",
                package_name="plotly",
                level=DependencyLevel.OPTIONAL,
                version_spec=">=5.18.0",
                description="インタラクティブ可視化ライブラリ"
            )
        }
    
    def check_all_dependencies(self, use_cache: bool = True) -> DependencyCheckResult:
        """
        全依存関係をチェック
        
        Args:
            use_cache: キャッシュを使用するか
            
        Returns:
            DependencyCheckResult: チェック結果
        """
        # キャッシュの確認
        if use_cache:
            cached_result = self._load_cache()
            if cached_result:
                logger.info("Using cached dependency check result")
                return cached_result
        
        logger.info("Checking all dependencies...")
        
        missing_core = []
        missing_infrastructure = []
        missing_optional = []
        warnings = []
        alternatives_used = {}
        installed_versions = {}
        
        for dep_name, dep in self.dependencies.items():
            logger.debug(f"Checking {dep_name}...")
            
            # パッケージのチェック
            is_available, version = self._check_package_with_version(dep.package_name)
            
            if is_available:
                installed_versions[dep_name] = version or "unknown"
                
                # バージョンチェック
                if dep.version_spec and version:
                    if not self._check_version_spec(version, dep.version_spec):
                        warnings.append(
                            f"{dep_name}: Installed version {version} may not meet requirement {dep.version_spec}"
                        )
            else:
                # 代替手段をチェック
                alt_found = False
                if dep.alternatives:
                    for alt in dep.alternatives:
                        if self._check_package(alt):
                            alt_found = True
                            alternatives_used[dep_name] = alt
                            warnings.append(
                                f"{dep_name} not found, using {alt} as alternative"
                            )
                            break
                
                if not alt_found:
                    if dep.level == DependencyLevel.CORE:
                        missing_core.append(dep_name)
                    elif dep.level == DependencyLevel.INFRASTRUCTURE:
                        missing_infrastructure.append(dep_name)
                    else:
                        missing_optional.append(dep_name)
                    
            # カスタムチェック関数がある場合は実行
            if dep.check_function and is_available:
                try:
                    service_ok = dep.check_function()
                    if not service_ok:
                        warnings.append(f"Service check failed for {dep_name}")
                except Exception as e:
                    warnings.append(f"Service check error for {dep_name}: {str(e)}")
        
        # 動作可能かどうかを判定
        can_run = len(missing_core) == 0 and len(missing_infrastructure) <= len(alternatives_used)
        
        result = DependencyCheckResult(
            is_satisfied=(len(missing_core) == 0 and 
                         len(missing_infrastructure) == 0 and 
                         len(missing_optional) == 0),
            missing_core=missing_core,
            missing_infrastructure=missing_infrastructure,
            missing_optional=missing_optional,
            warnings=warnings,
            can_run=can_run,
            alternatives_used=alternatives_used,
            installed_versions=installed_versions
        )
        
        # キャッシュに保存
        self._save_cache(result)
        
        return result
    
    def _check_package_with_version(self, package_name: str) -> Tuple[bool, Optional[str]]:
        """
        パッケージの存在とバージョンをチェック
        
        Returns:
            Tuple[bool, Optional[str]]: (存在するか, バージョン)
        """
        try:
            # パッケージをインポート
            module = importlib.import_module(package_name)
            
            # バージョンの取得を試みる
            version = None
            
            # __version__ 属性を確認
            if hasattr(module, "__version__"):
                version = module.__version__
            else:
                # importlib.metadataを使用してバージョンを取得
                try:
                    version = importlib.metadata.version(package_name)
                except:
                    pass
            
            return True, version
            
        except ImportError:
            return False, None
    
    def _check_package(self, package_name: str) -> bool:
        """パッケージの存在をチェック"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def _check_version_spec(self, version: str, spec: str) -> bool:
        """
        バージョン指定を満たしているかチェック
        
        簡易実装：>= のみをサポート
        """
        if spec.startswith(">="):
            required_version = spec[2:].strip()
            return self._compare_versions(version, required_version) >= 0
        return True  # 他の指定子は一旦trueを返す
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        バージョンを比較
        Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        try:
            from packaging import version
            return -1 if version.parse(v1) < version.parse(v2) else (
                0 if version.parse(v1) == version.parse(v2) else 1
            )
        except:
            # packagingが使えない場合は簡易比較
            v1_parts = [int(x) for x in v1.split('.')]
            v2_parts = [int(x) for x in v2.split('.')]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                p1 = v1_parts[i] if i < len(v1_parts) else 0
                p2 = v2_parts[i] if i < len(v2_parts) else 0
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            return 0
    
    def _check_qdrant_service(self) -> bool:
        """Qdrantサービスの稼働状況をチェック"""
        try:
            from qdrant_client import QdrantClient
            
            # 環境変数から設定を取得
            host = os.environ.get("QDRANT_HOST", "localhost")
            port = int(os.environ.get("QDRANT_PORT", "6333"))
            
            client = QdrantClient(host=host, port=port, timeout=5.0)
            collections = client.get_collections()
            logger.debug(f"Qdrant service is running with {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.debug(f"Qdrant service check failed: {e}")
            return False
    
    def _check_spacy_model(self) -> bool:
        """spaCyの日本語モデルがインストールされているかチェック"""
        try:
            import spacy
            spacy.load("ja_core_news_lg")
            return True
        except:
            return False
    
    def install_missing_dependencies(
        self, 
        level: Optional[DependencyLevel] = None,
        dry_run: bool = False
    ) -> Dict[str, bool]:
        """
        不足している依存関係をインストール
        
        Args:
            level: インストールするレベル（Noneの場合は全レベル）
            dry_run: 実際にインストールせずにコマンドを表示
            
        Returns:
            Dict[str, bool]: パッケージ名とインストール成功/失敗
        """
        result = self.check_all_dependencies(use_cache=False)
        install_results = {}
        
        to_install = []
        
        # インストール対象を収集
        if level is None or level == DependencyLevel.CORE:
            for dep_name in result.missing_core:
                if dep_name in self.dependencies:
                    to_install.append(self.dependencies[dep_name])
        
        if level is None or level == DependencyLevel.INFRASTRUCTURE:
            for dep_name in result.missing_infrastructure:
                if dep_name in self.dependencies:
                    to_install.append(self.dependencies[dep_name])
        
        if level is None or level == DependencyLevel.OPTIONAL:
            for dep_name in result.missing_optional:
                if dep_name in self.dependencies:
                    to_install.append(self.dependencies[dep_name])
        
        # インストール実行
        for dep in to_install:
            if dep.install_command:
                cmd = dep.install_command
            else:
                package = dep.name if dep.name != dep.package_name else dep.package_name
                if dep.version_spec:
                    package = f"{package}{dep.version_spec}"
                cmd = f"{sys.executable} -m pip install {package}"
            
            logger.info(f"Installing {dep.name}: {cmd}")
            
            if dry_run:
                install_results[dep.name] = None
            else:
                try:
                    subprocess.check_call(cmd.split())
                    install_results[dep.name] = True
                    logger.info(f"Successfully installed {dep.name}")
                except subprocess.CalledProcessError as e:
                    install_results[dep.name] = False
                    logger.error(f"Failed to install {dep.name}: {e}")
        
        # キャッシュをクリア
        self._clear_cache()
        
        return install_results
    
    def get_dependency_report(self, format: str = "text") -> str:
        """
        依存関係レポートを生成
        
        Args:
            format: 出力形式 ("text", "json", "markdown")
            
        Returns:
            str: レポート文字列
        """
        result = self.check_all_dependencies()
        
        if format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            report = []
            report.append("# RAG System Dependency Report")
            report.append(f"\n**Generated at:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"\n**Status:** {'✅ All satisfied' if result.is_satisfied else '⚠️ Some missing'}")
            report.append(f"\n**Can Run:** {'Yes' if result.can_run else 'No'}")
            
            if result.missing_core:
                report.append("\n## ❌ Missing Core Dependencies")
                for dep in result.missing_core:
                    if dep in self.dependencies:
                        d = self.dependencies[dep]
                        report.append(f"- **{d.name}** ({d.package_name}): {d.description or 'N/A'}")
            
            if result.missing_infrastructure:
                report.append("\n## ⚠️ Missing Infrastructure Dependencies")
                for dep in result.missing_infrastructure:
                    if dep in self.dependencies:
                        d = self.dependencies[dep]
                        report.append(f"- **{d.name}** ({d.package_name}): {d.description or 'N/A'}")
            
            if result.missing_optional:
                report.append("\n## ℹ️ Missing Optional Dependencies")
                for dep in result.missing_optional:
                    if dep in self.dependencies:
                        d = self.dependencies[dep]
                        report.append(f"- **{d.name}** ({d.package_name}): {d.description or 'N/A'}")
            
            if result.alternatives_used:
                report.append("\n## 🔄 Alternative Packages Used")
                for original, alternative in result.alternatives_used.items():
                    report.append(f"- {original} → {alternative}")
            
            if result.warnings:
                report.append("\n## ⚠️ Warnings")
                for warning in result.warnings:
                    report.append(f"- {warning}")
            
            if result.installed_versions:
                report.append("\n## 📦 Installed Versions")
                for name, version in result.installed_versions.items():
                    report.append(f"- {name}: {version}")
            
            return "\n".join(report)
        
        else:  # text format
            report = []
            report.append("=" * 50)
            report.append("RAG System Dependency Report")
            report.append("=" * 50)
            report.append(f"Generated at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            if result.is_satisfied:
                report.append("✅ All dependencies are satisfied")
            else:
                report.append("⚠️ Some dependencies are missing")
            
            report.append(f"🔧 System can run: {'Yes' if result.can_run else 'No'}")
            report.append("")
            
            if result.missing_core:
                report.append("❌ Missing Core Dependencies:")
                for dep in result.missing_core:
                    report.append(f"  - {dep}")
            
            if result.missing_infrastructure:
                report.append("⚠️ Missing Infrastructure Dependencies:")
                for dep in result.missing_infrastructure:
                    report.append(f"  - {dep}")
            
            if result.missing_optional:
                report.append("ℹ️ Missing Optional Dependencies:")
                for dep in result.missing_optional:
                    report.append(f"  - {dep}")
            
            if result.alternatives_used:
                report.append("🔄 Alternative Packages Used:")
                for original, alternative in result.alternatives_used.items():
                    report.append(f"  - {original} → {alternative}")
            
            if result.warnings:
                report.append("⚠️ Warnings:")
                for warning in result.warnings:
                    report.append(f"  - {warning}")
            
            report.append("")
            report.append("=" * 50)
            
            return "\n".join(report)
    
    def _load_cache(self) -> Optional[DependencyCheckResult]:
        """キャッシュから結果を読み込み"""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / "dependency_check.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # キャッシュファイルの更新時刻を確認（1時間以内なら有効）
            import time
            if time.time() - cache_file.stat().st_mtime > 3600:
                return None
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            return DependencyCheckResult(
                is_satisfied=data['is_satisfied'],
                missing_core=data['missing_core'],
                missing_infrastructure=data['missing_infrastructure'],
                missing_optional=data['missing_optional'],
                warnings=data['warnings'],
                can_run=data['can_run'],
                alternatives_used=data['alternatives_used'],
                installed_versions=data['installed_versions'],
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None
    
    def _save_cache(self, result: DependencyCheckResult):
        """結果をキャッシュに保存"""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / "dependency_check.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def _clear_cache(self):
        """キャッシュをクリア"""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / "dependency_check.json"
        if cache_file.exists():
            cache_file.unlink()


# ユーティリティ関数
def check_and_install_dependencies(auto_install: bool = False) -> bool:
    """
    依存関係をチェックし、必要に応じてインストール
    
    Args:
        auto_install: 自動的にインストールするか
        
    Returns:
        bool: システムが実行可能か
    """
    manager = RAGDependencyManager()
    result = manager.check_all_dependencies()
    
    print(manager.get_dependency_report())
    
    if not result.can_run:
        if auto_install or (
            input("\nInstall missing dependencies? (y/n): ").lower() == 'y'
        ):
            install_results = manager.install_missing_dependencies()
            
            # 再チェック
            result = manager.check_all_dependencies(use_cache=False)
            print("\n" + manager.get_dependency_report())
    
    return result.can_run


if __name__ == "__main__":
    # CLIとして実行された場合
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Dependency Manager")
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check all dependencies"
    )
    parser.add_argument(
        "--install", 
        action="store_true", 
        help="Install missing dependencies"
    )
    parser.add_argument(
        "--report", 
        choices=["text", "json", "markdown"], 
        default="text",
        help="Generate dependency report"
    )
    parser.add_argument(
        "--level",
        choices=["core", "infrastructure", "optional"],
        help="Dependency level to install"
    )
    
    args = parser.parse_args()
    
    manager = RAGDependencyManager()
    
    if args.check or args.report:
        result = manager.check_all_dependencies()
        print(manager.get_dependency_report(format=args.report))
        sys.exit(0 if result.can_run else 1)
    
    if args.install:
        level = DependencyLevel[args.level.upper()] if args.level else None
        install_results = manager.install_missing_dependencies(level=level)
        
        print("\nInstallation Results:")
        for package, success in install_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {package}")
        
        # 最終チェック
        result = manager.check_all_dependencies(use_cache=False)
        sys.exit(0 if result.can_run else 1)
