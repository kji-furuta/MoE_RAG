"""RAGアプリケーションクラス.

検索エンジン、メタデータマネージャー、検索履歴管理を統合する
RAGシステムのファサードクラス。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.schemas.rag import (
    QueryResponse,
    SavedSearchResult,
    SearchHistoryResponse,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


class RAGApplication:
    """RAGアプリケーション"""

    def __init__(self):
        self.query_engine = None
        self.metadata_manager = None
        self.is_initialized = False
        self.initialization_error = None
        self.config: Dict[str, Any] = {}

        # Optional RAG imports
        try:
            from dataclasses import asdict
            from src.rag.config.rag_config import load_config
            self.config = asdict(load_config())
        except Exception as config_error:
            logger.warning(f"RAG設定の読み込みに失敗しました: {config_error}")
            self.config = {}

        # Docker環境に対応した永続化ディレクトリの設定
        if os.path.exists("/workspace"):
            self.search_history_dir = Path("/workspace/data/search_history")
        else:
            project_root = Path(__file__).parent.parent
            self.search_history_dir = project_root / "data" / "search_history"

        self.search_history_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Search history directory: {self.search_history_dir}")

    async def initialize(self):
        """非同期でシステムを初期化"""
        try:
            from src.rag.core.query_engine import RoadDesignQueryEngine
            from src.rag.indexing.metadata_manager import MetadataManager
        except ImportError as e:
            self.initialization_error = f"RAG system components not available: {e}"
            return

        try:
            import asyncio
            logger.info("Initializing RAG system...")

            self.query_engine = RoadDesignQueryEngine()
            await asyncio.get_event_loop().run_in_executor(
                None, self.query_engine.initialize
            )

            metadata_db_path = "/workspace/metadata/metadata.db" if os.path.exists("/workspace") else "./metadata/metadata.db"
            self.metadata_manager = MetadataManager(db_path=metadata_db_path)
            logger.info(f"MetadataManager initialized with path: {metadata_db_path}")

            self.is_initialized = True
            logger.info("RAG system initialized successfully")

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Failed to initialize RAG system: {e}")

    def check_initialized(self):
        """初期化チェック"""
        if not self.is_initialized:
            if self.initialization_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAG system initialization failed: {self.initialization_error}"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="RAG system is not yet initialized"
                )

    def switch_to_finetuned_model(self, model_path: str) -> bool:
        """ファインチューニングモデルに切り替え

        query engine の LLMGenerator 設定を直接更新して
        ローカルモデルをロードする。

        Returns:
            True if switch succeeded, False otherwise.
        """
        if not self.query_engine or not self.query_engine.llm_generator:
            logger.warning("Query engine or LLM generator not available")
            return False

        llm_gen = self.query_engine.llm_generator
        llm_config = llm_gen.config.llm

        # LLMGenerator の設定を更新
        llm_config.model_name = model_path
        llm_config.use_finetuned = True
        llm_config.provider = 'local'

        # 既存のモデルをアンロード
        if llm_gen.model is not None:
            del llm_gen.model
            llm_gen.model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ollama フォールバックを無効化
        llm_gen.use_ollama_fallback = False

        logger.info(f"Switched LLM config to finetuned model: {model_path}")
        return True

    def restore_default_model(self) -> None:
        """デフォルトのモデル設定に復元"""
        if not self.query_engine or not self.query_engine.llm_generator:
            return

        llm_gen = self.query_engine.llm_generator
        llm_config = llm_gen.config.llm

        # 元のOllama設定に戻す
        original_model = self.config.get('llm', {}).get('model_name', '')
        original_provider = self.config.get('llm', {}).get('provider', 'ollama')

        llm_config.model_name = original_model
        llm_config.use_finetuned = False
        llm_config.provider = original_provider

        # ローカルモデルをアンロード
        if llm_gen.model is not None:
            del llm_gen.model
            llm_gen.model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ollama フォールバックを再有効化
        if original_provider == 'ollama':
            llm_gen._enable_ollama_fallback()

        logger.info(f"Restored default model: {original_model}")

    def save_search_result(self, query_response: QueryResponse, name: Optional[str] = None, tags: Optional[List[str]] = None) -> SavedSearchResult:
        """検索結果を保存"""
        result_id = str(uuid.uuid4())
        timestamp = datetime.now(JST).isoformat()

        saved_result = SavedSearchResult(
            id=result_id,
            query=query_response.query,
            answer=query_response.answer,
            citations=query_response.citations,
            sources=query_response.sources,
            confidence_score=query_response.confidence_score,
            search_type=query_response.metadata.get("search_type", "hybrid"),
            top_k=query_response.metadata.get("top_k", 5),
            saved_at=timestamp,
            metadata={
                "name": name or f"Search_{timestamp[:10]}",
                "tags": tags or [],
                "processing_time": query_response.processing_time
            }
        )

        file_path = self.search_history_dir / f"{result_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(saved_result.dict(), f, ensure_ascii=False, indent=2)

        return saved_result

    def get_search_history(self, page: int = 1, limit: int = 10, tag: Optional[str] = None) -> SearchHistoryResponse:
        """検索履歴を取得"""
        all_results = []

        for json_file in self.search_history_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)

                file_id = json_file.stem
                result_data["id"] = file_id

                if tag and tag not in result_data.get("metadata", {}).get("tags", []):
                    continue

                all_results.append(SavedSearchResult(**result_data))
            except Exception as e:
                logger.error(f"Error loading search result {json_file}: {e}")

        all_results.sort(key=lambda x: x.saved_at, reverse=True)

        total = len(all_results)
        start = (page - 1) * limit
        end = start + limit
        paginated_results = all_results[start:end]

        return SearchHistoryResponse(
            total=total,
            results=paginated_results,
            page=page,
            limit=limit
        )

    def get_saved_result(self, result_id: str) -> Optional[SavedSearchResult]:
        """保存された検索結果を取得"""
        file_path = self.search_history_dir / f"{result_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                result_data["id"] = result_id
                return SavedSearchResult(**result_data)
        except Exception as e:
            logger.error(f"Error loading search result {result_id}: {e}")
            return None

    def export_search_results(self, result_ids: List[str], format: str = "json") -> bytes:
        """検索結果をエクスポート"""
        results = []
        for result_id in result_ids:
            result = self.get_saved_result(result_id)
            if result:
                results.append(result.dict())

        if format == "csv":
            output = io.StringIO()
            if results:
                writer = csv.DictWriter(output, fieldnames=["query", "answer", "confidence_score", "saved_at", "search_type"])
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        "query": result["query"],
                        "answer": result["answer"],
                        "confidence_score": result["confidence_score"],
                        "saved_at": result["saved_at"],
                        "search_type": result["search_type"]
                    })

            return output.getvalue().encode("utf-8")
        else:
            return json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")

    def delete_search_history_item(self, result_id: str) -> bool:
        """検索履歴の個別アイテムを削除"""
        try:
            if ".." in result_id or "/" in result_id or "\\" in result_id:
                logger.error(f"Invalid result_id: {result_id}")
                return False

            file_path = self.search_history_dir / f"{result_id}.json"

            if not file_path.exists():
                logger.warning(f"Search history item not found: {result_id}")
                return False

            if not str(file_path.resolve()).startswith(str(self.search_history_dir.resolve())):
                logger.error(f"Invalid file path: {file_path}")
                return False

            file_path.unlink()
            logger.info(f"Deleted search history item: {result_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting search history item {result_id}: {e}")
            return False
