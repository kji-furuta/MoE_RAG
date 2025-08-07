"""
RAGシステムのテストコード（リファクタリング版）
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
import json

from src.rag.utils import (
    validate_query, validate_top_k, validate_search_type,
    format_citations, format_sources, format_metadata,
    ValidationError, RAGException
)
from src.rag.config.config_manager import ConfigManager
from src.rag.core.query_engine_refactored import (
    QueryResult, LLMGenerator, RoadDesignQueryEngine
)


class TestValidation:
    """バリデーションユーティリティのテスト"""
    
    def test_validate_query_valid(self):
        """正常なクエリのバリデーション"""
        query = "道路の設計速度について教えてください"
        result = validate_query(query)
        assert result == query
    
    def test_validate_query_empty(self):
        """空クエリのバリデーション"""
        with pytest.raises(ValidationError):
            validate_query("")
    
    def test_validate_query_too_long(self):
        """長すぎるクエリのバリデーション"""
        long_query = "a" * 3000
        with pytest.raises(ValidationError):
            validate_query(long_query, max_length=2000)
    
    def test_validate_top_k_valid(self):
        """正常なtop_kのバリデーション"""
        assert validate_top_k(5) == 5
        assert validate_top_k(10) == 10
    
    def test_validate_top_k_invalid(self):
        """不正なtop_kのバリデーション"""
        with pytest.raises(ValidationError):
            validate_top_k(0)
        with pytest.raises(ValidationError):
            validate_top_k(101)
    
    def test_validate_search_type_valid(self):
        """正常な検索タイプのバリデーション"""
        assert validate_search_type("vector") == "vector"
        assert validate_search_type("HYBRID") == "hybrid"
    
    def test_validate_search_type_invalid(self):
        """不正な検索タイプのバリデーション"""
        with pytest.raises(ValidationError):
            validate_search_type("invalid_type")


class TestFormatting:
    """フォーマッティングユーティリティのテスト"""
    
    def test_format_citations(self):
        """引用のフォーマット"""
        citations = [
            {"text": "引用1", "source": "文書A"},
            {"text": "引用2"},
            "引用3"
        ]
        
        formatted = format_citations(citations)
        
        assert len(formatted) == 3
        assert formatted[0]["source"] == "文書A"
        assert formatted[1]["source"] == "Unknown"
        assert formatted[2]["text"] == "引用3"
    
    def test_format_sources(self):
        """ソースのフォーマット"""
        sources = [
            {"text": "ソース1", "score": 0.9, "title": "タイトル1"},
            {"text": "ソース2", "score": 1.5},  # スコアが範囲外
            {"text": "ソース3"}  # スコアなし
        ]
        
        formatted = format_sources(sources)
        
        assert len(formatted) == 3
        assert formatted[0]["score"] == 0.9
        assert formatted[1]["score"] == 1.0  # 正規化される
        assert formatted[2]["score"] == 0.0  # デフォルト値
    
    def test_format_metadata(self):
        """メタデータのフォーマット"""
        metadata = {
            "processing_time": 1.234567,
            "memory_usage": 1024 * 1024 * 512,  # 512MB
            "count": 10
        }
        
        formatted = format_metadata(metadata, readable_format=True)
        
        assert formatted["processing_time"] == "1.23s"
        assert formatted["memory_usage"] == "512.00 MB"
        assert formatted["count"] == 10


class TestConfigManager:
    """設定管理のテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        manager = ConfigManager()
        
        assert manager.get("embedding.model_name") == "intfloat/multilingual-e5-large"
        assert manager.get("search.vector_weight") == 0.7
        assert manager.get("generation.temperature") == 0.7
    
    def test_config_from_dict(self):
        """辞書から設定を読み込み"""
        config_dict = {
            "embedding": {
                "model_name": "custom-model"
            },
            "search": {
                "top_k": 20
            }
        }
        
        manager = ConfigManager(config_dict=config_dict)
        
        assert manager.get("embedding.model_name") == "custom-model"
        assert manager.get("search.top_k") == 20
        assert manager.get("search.vector_weight") == 0.7  # デフォルト値
    
    def test_config_from_yaml(self):
        """YAMLファイルから設定を読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
embedding:
  model_name: yaml-model
  dimension: 768
search:
  vector_weight: 0.8
  keyword_weight: 0.2
""")
            yaml_path = f.name
        
        try:
            manager = ConfigManager(config_path=yaml_path)
            
            assert manager.get("embedding.model_name") == "yaml-model"
            assert manager.get("embedding.dimension") == 768
            assert manager.get("search.vector_weight") == 0.8
            
        finally:
            Path(yaml_path).unlink()
    
    def test_config_validation(self):
        """設定検証のテスト"""
        # 不正な重み
        with pytest.raises(Exception):  # ConfigurationError
            ConfigManager(config_dict={
                "search": {
                    "vector_weight": 1.5  # 範囲外
                }
            })
    
    def test_get_set_config(self):
        """設定の取得と設定"""
        manager = ConfigManager()
        
        # 設定
        manager.set("custom.key", "custom_value")
        manager.set("embedding.batch_size", 64)
        
        # 取得
        assert manager.get("custom.key") == "custom_value"
        assert manager.get("embedding.batch_size") == 64
        assert manager.get("nonexistent.key", "default") == "default"


class TestQueryResult:
    """クエリ結果のテスト"""
    
    def test_query_result_creation(self):
        """QueryResultの作成"""
        result = QueryResult(
            query="テストクエリ",
            answer="テスト回答",
            citations=[{"text": "引用"}],
            sources=[{"text": "ソース", "score": 0.9}],
            confidence_score=0.85,
            processing_time=1.5
        )
        
        assert result.query == "テストクエリ"
        assert result.answer == "テスト回答"
        assert len(result.citations) == 1
        assert len(result.sources) == 1
        assert result.confidence_score == 0.85
        assert result.processing_time == 1.5
    
    def test_query_result_defaults(self):
        """QueryResultのデフォルト値"""
        result = QueryResult(
            query="テストクエリ",
            answer="テスト回答"
        )
        
        assert result.citations == []
        assert result.sources == []
        assert result.confidence_score == 0.0
        assert result.processing_time == 0.0
        assert result.metadata == {}


class TestQueryEngine:
    """クエリエンジンのテスト"""
    
    @pytest.fixture
    def temp_config(self):
        """テスト用の一時設定ファイル"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
embedding:
  model_name: test-model
  dimension: 384
search:
  top_k: 5
generation:
  model_name: test-llm
  temperature: 0.5
""")
            yield Path(f.name)
        
        Path(f.name).unlink()
    
    def test_engine_initialization(self, temp_config):
        """エンジンの初期化"""
        engine = RoadDesignQueryEngine(config_path=temp_config)
        
        assert not engine.is_initialized
        assert engine.config is not None
        
        # 最小モードで初期化（テスト用）
        engine.initialize(mode="minimal")
        
        assert engine.is_initialized
        assert engine.llm_generator is not None
    
    def test_engine_system_info(self, temp_config):
        """システム情報の取得"""
        engine = RoadDesignQueryEngine(config_path=temp_config)
        engine.initialize(mode="minimal")
        
        info = engine.get_system_info()
        
        assert info["initialized"] == True
        assert "components" in info
        assert "config" in info
        assert "gpu" in info
    
    def test_error_result_creation(self, temp_config):
        """エラー結果の作成"""
        engine = RoadDesignQueryEngine(config_path=temp_config)
        
        result = engine._create_error_result(
            "テストクエリ",
            "テストエラー",
            1.0
        )
        
        assert "エラーが発生しました" in result.answer
        assert result.confidence_score == 0.0
        assert result.processing_time == 1.0
        assert result.metadata["error"] == "テストエラー"


@pytest.mark.integration
class TestIntegration:
    """統合テスト"""
    
    def test_end_to_end_workflow(self):
        """エンドツーエンドのワークフロー"""
        # 設定の作成
        config = ConfigManager(config_dict={
            "embedding": {"model_name": "test-model"},
            "generation": {"temperature": 0.5}
        })
        
        # エンジンの初期化
        engine = RoadDesignQueryEngine()
        engine.config = config.to_dict()
        
        # クエリの検証
        query = "道路設計の基準について"
        validated_query = validate_query(query)
        
        assert validated_query == query
        
        # 結果の作成（モック）
        result = QueryResult(
            query=validated_query,
            answer="道路設計基準についての回答",
            citations=[{"text": "基準1", "source": "文書A"}],
            sources=[{"text": "ソース1", "score": 0.9}],
            confidence_score=0.85,
            processing_time=2.0
        )
        
        # フォーマット
        formatted_citations = format_citations(result.citations)
        formatted_sources = format_sources(result.sources)
        formatted_metadata = format_metadata({
            "processing_time": result.processing_time,
            "confidence": result.confidence_score
        })
        
        assert len(formatted_citations) == 1
        assert len(formatted_sources) == 1
        assert "processing_time" in formatted_metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])