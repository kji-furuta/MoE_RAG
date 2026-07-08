"""
依存関係マネージャーのテスト

このテストは、RAGシステムの依存関係管理機能が
正しく動作することを確認します。
"""

import pytest
import sys
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# プロジェクトのルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.dependencies.dependency_manager import (
    DependencyLevel,
    Dependency,
    DependencyCheckResult,
    RAGDependencyManager,
    check_and_install_dependencies
)


class TestDependency:
    """Dependencyクラスのテスト"""
    
    def test_dependency_creation(self):
        """依存関係オブジェクトの作成テスト"""
        dep = Dependency(
            name="Test Package",
            package_name="test_package",
            level=DependencyLevel.CORE,
            version_spec=">=1.0.0",
            alternatives=["alt_package"],
            description="Test description"
        )
        
        assert dep.name == "Test Package"
        assert dep.package_name == "test_package"
        assert dep.level == DependencyLevel.CORE
        assert dep.version_spec == ">=1.0.0"
        assert "alt_package" in dep.alternatives
        assert dep.description == "Test description"
    
    def test_dependency_levels(self):
        """依存レベルの確認テスト"""
        assert DependencyLevel.CORE.value == "core"
        assert DependencyLevel.INFRASTRUCTURE.value == "infrastructure"
        assert DependencyLevel.OPTIONAL.value == "optional"


class TestDependencyCheckResult:
    """DependencyCheckResultクラスのテスト"""
    
    def test_result_creation(self):
        """チェック結果オブジェクトの作成テスト"""
        result = DependencyCheckResult(
            is_satisfied=True,
            missing_core=[],
            missing_infrastructure=[],
            missing_optional=["optional_pkg"],
            warnings=["Test warning"],
            can_run=True,
            alternatives_used={"qdrant": "chromadb"},
            installed_versions={"torch": "2.0.1"}
        )
        
        assert result.is_satisfied == True
        assert result.can_run == True
        assert len(result.missing_optional) == 1
        assert "optional_pkg" in result.missing_optional
        assert "Test warning" in result.warnings
        assert result.alternatives_used["qdrant"] == "chromadb"
    
    def test_missing_dependencies_property(self):
        """missing_dependenciesプロパティのテスト"""
        result = DependencyCheckResult(
            is_satisfied=False,
            missing_core=["core1"],
            missing_infrastructure=["infra1", "infra2"],
            missing_optional=["opt1"],
            warnings=[],
            can_run=False,
            alternatives_used={},
            installed_versions={}
        )
        
        missing = result.missing_dependencies
        assert len(missing) == 3
        assert "core1" in missing
        assert "infra1" in missing
        assert "infra2" in missing
        assert "opt1" not in missing  # オプションは含まれない
    
    def test_to_dict(self):
        """辞書変換のテスト"""
        result = DependencyCheckResult(
            is_satisfied=True,
            missing_core=[],
            missing_infrastructure=[],
            missing_optional=[],
            warnings=[],
            can_run=True,
            alternatives_used={},
            installed_versions={"torch": "2.0.1"}
        )
        
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["is_satisfied"] == True
        assert data["can_run"] == True
        assert "timestamp" in data
        assert "installed_versions" in data


class TestRAGDependencyManager:
    """RAGDependencyManagerクラスのテスト"""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """テスト用マネージャーのフィクスチャ"""
        cache_dir = tmp_path / "cache"
        return RAGDependencyManager(cache_dir=cache_dir)
    
    def test_manager_initialization(self, manager):
        """マネージャーの初期化テスト"""
        assert manager is not None
        assert len(manager.dependencies) > 0
        assert "torch" in manager.dependencies
        assert "transformers" in manager.dependencies
    
    def test_define_dependencies(self, manager):
        """依存関係定義のテスト"""
        deps = manager._define_dependencies()
        
        # コア依存関係の確認
        assert "torch" in deps
        assert deps["torch"].level == DependencyLevel.CORE
        
        # インフラ依存関係の確認
        assert "qdrant" in deps
        assert deps["qdrant"].level == DependencyLevel.INFRASTRUCTURE
        assert len(deps["qdrant"].alternatives) > 0
        
        # オプション依存関係の確認
        assert "easyocr" in deps
        assert deps["easyocr"].level == DependencyLevel.OPTIONAL
    
    @patch('importlib.import_module')
    def test_check_package(self, mock_import, manager):
        """パッケージチェックのテスト"""
        # 成功ケース
        mock_import.return_value = MagicMock()
        assert manager._check_package("test_package") == True
        
        # 失敗ケース
        mock_import.side_effect = ImportError()
        assert manager._check_package("missing_package") == False
    
    @patch('importlib.import_module')
    @patch('importlib.metadata.version')
    def test_check_package_with_version(self, mock_version, mock_import, manager):
        """バージョン付きパッケージチェックのテスト"""
        # モジュールに__version__がある場合
        mock_module = MagicMock()
        mock_module.__version__ = "1.2.3"
        mock_import.return_value = mock_module
        
        exists, version = manager._check_package_with_version("test_package")
        assert exists == True
        assert version == "1.2.3"
        
        # モジュールに__version__がない場合
        mock_module = MagicMock(spec=[])  # __version__なし
        mock_import.return_value = mock_module
        mock_version.return_value = "2.0.0"
        
        exists, version = manager._check_package_with_version("test_package")
        assert exists == True
        assert version == "2.0.0"
    
    def test_compare_versions(self, manager):
        """バージョン比較のテスト"""
        assert manager._compare_versions("2.0.0", "1.0.0") == 1
        assert manager._compare_versions("1.0.0", "2.0.0") == -1
        assert manager._compare_versions("1.0.0", "1.0.0") == 0
        assert manager._compare_versions("1.2.3", "1.2.0") == 1
        assert manager._compare_versions("1.2.0", "1.2.3") == -1
    
    def test_check_version_spec(self, manager):
        """バージョン指定チェックのテスト"""
        assert manager._check_version_spec("2.0.0", ">=1.0.0") == True
        assert manager._check_version_spec("0.9.0", ">=1.0.0") == False
        assert manager._check_version_spec("1.0.0", ">=1.0.0") == True
    
    @patch.object(RAGDependencyManager, '_check_package_with_version')
    def test_check_all_dependencies(self, mock_check, manager):
        """全依存関係チェックのテスト"""
        # 全て満たされている場合
        mock_check.return_value = (True, "1.0.0")
        
        result = manager.check_all_dependencies(use_cache=False)
        
        assert result.is_satisfied == True
        assert result.can_run == True
        assert len(result.missing_core) == 0
        assert len(result.missing_infrastructure) == 0
    
    @patch.object(RAGDependencyManager, '_check_package_with_version')
    @patch.object(RAGDependencyManager, '_check_package')
    def test_check_with_alternatives(self, mock_check_pkg, mock_check_ver, manager):
        """代替パッケージのチェックテスト"""
        def check_side_effect(pkg):
            if pkg == "qdrant_client":
                return (False, None)
            elif pkg == "chromadb":
                return (True, "0.4.0")
            else:
                return (True, "1.0.0")
        
        mock_check_ver.side_effect = lambda pkg: check_side_effect(pkg)
        mock_check_pkg.side_effect = lambda pkg: pkg == "chromadb"
        
        result = manager.check_all_dependencies(use_cache=False)
        
        # Qdrantが無くても代替のchromadbがあるので動作可能
        assert result.can_run == True
        assert "qdrant" in result.alternatives_used
        assert result.alternatives_used["qdrant"] == "chromadb"
    
    def test_get_dependency_report_text(self, manager):
        """テキスト形式レポートのテスト"""
        report = manager.get_dependency_report(format="text")
        
        assert isinstance(report, str)
        assert "RAG System Dependency Report" in report
        assert "System can run:" in report
    
    def test_get_dependency_report_json(self, manager):
        """JSON形式レポートのテスト"""
        report = manager.get_dependency_report(format="json")
        
        data = json.loads(report)
        assert isinstance(data, dict)
        assert "is_satisfied" in data
        assert "can_run" in data
        assert "timestamp" in data
    
    def test_get_dependency_report_markdown(self, manager):
        """Markdown形式レポートのテスト"""
        report = manager.get_dependency_report(format="markdown")
        
        assert isinstance(report, str)
        assert "# RAG System Dependency Report" in report
        assert "**Status:**" in report
    
    @patch('subprocess.check_call')
    def test_install_missing_dependencies(self, mock_subprocess, manager):
        """依存関係インストールのテスト"""
        # いくつかの依存関係が不足している状況を設定
        with patch.object(manager, 'check_all_dependencies') as mock_check:
            mock_check.return_value = DependencyCheckResult(
                is_satisfied=False,
                missing_core=["torch"],
                missing_infrastructure=[],
                missing_optional=[],
                warnings=[],
                can_run=False,
                alternatives_used={},
                installed_versions={}
            )
            
            # ドライラン
            results = manager.install_missing_dependencies(dry_run=True)
            assert "torch" in results
            assert results["torch"] is None  # ドライランなのでNone
            
            # 実際のインストール（モック）
            mock_subprocess.return_value = None
            results = manager.install_missing_dependencies(dry_run=False)
            assert mock_subprocess.called
    
    def test_cache_operations(self, manager, tmp_path):
        """キャッシュ操作のテスト"""
        result = DependencyCheckResult(
            is_satisfied=True,
            missing_core=[],
            missing_infrastructure=[],
            missing_optional=[],
            warnings=[],
            can_run=True,
            alternatives_used={},
            installed_versions={"torch": "2.0.1"}
        )
        
        # キャッシュ保存
        manager._save_cache(result)
        
        # キャッシュ読み込み
        loaded = manager._load_cache()
        assert loaded is not None
        assert loaded.is_satisfied == True
        assert loaded.installed_versions["torch"] == "2.0.1"
        
        # キャッシュクリア
        manager._clear_cache()
        loaded = manager._load_cache()
        assert loaded is None


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    @patch('builtins.input')
    @patch.object(RAGDependencyManager, 'check_all_dependencies')
    def test_check_and_install_dependencies(self, mock_check, mock_input):
        """check_and_install_dependencies関数のテスト"""
        # 全て満たされている場合
        mock_check.return_value = DependencyCheckResult(
            is_satisfied=True,
            missing_core=[],
            missing_infrastructure=[],
            missing_optional=[],
            warnings=[],
            can_run=True,
            alternatives_used={},
            installed_versions={}
        )
        
        result = check_and_install_dependencies(auto_install=False)
        assert result == True
        
        # 依存関係が不足しているがインストールしない場合
        mock_check.return_value = DependencyCheckResult(
            is_satisfied=False,
            missing_core=["torch"],
            missing_infrastructure=[],
            missing_optional=[],
            warnings=[],
            can_run=False,
            alternatives_used={},
            installed_versions={}
        )
        mock_input.return_value = "n"
        
        result = check_and_install_dependencies(auto_install=False)
        assert result == False


# 統合テスト
class TestIntegration:
    """統合テスト"""
    
    @pytest.fixture
    def real_manager(self):
        """実際のマネージャー（モックなし）"""
        return RAGDependencyManager()
    
    def test_real_dependency_check(self, real_manager):
        """実際の依存関係チェック（インストール済み環境でのテスト）"""
        result = real_manager.check_all_dependencies(use_cache=False)
        
        # 少なくともいくつかのパッケージはインストールされているはず
        assert len(result.installed_versions) > 0
        
        # レポートが生成できることを確認
        report = real_manager.get_dependency_report()
        assert len(report) > 0
        
        # 各形式のレポートが生成できることを確認
        for format in ["text", "json", "markdown"]:
            report = real_manager.get_dependency_report(format=format)
            assert len(report) > 0
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists(),
        reason="Docker not available"
    )
    def test_service_checks(self, real_manager):
        """サービスチェックのテスト（Docker環境）"""
        # Qdrantサービスチェック
        qdrant_running = real_manager._check_qdrant_service()
        # 結果に関わらず、エラーなく実行できることを確認
        assert isinstance(qdrant_running, bool)


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v", "--tb=short"])
