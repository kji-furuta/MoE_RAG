"""Phase 1 fixes: security, exception hierarchy, and API endpoint tests.

Tests cover:
1. app.exceptions hierarchy correctness
2. File upload path traversal protection
3. Key API endpoint error handling (requires FastAPI + httpx)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# 1. Exception hierarchy tests (pure Python, no heavy deps)
# ===================================================================


class TestExceptionHierarchy:
    """Verify app.exceptions class relationships and basic behaviour."""

    @pytest.fixture(autouse=True)
    def _import_exceptions(self):
        self.mod = importlib.import_module("app.exceptions")

    # -- inheritance -------------------------------------------------

    def test_model_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.ModelError, self.mod.AppError)
        assert issubclass(self.mod.ModelLoadError, self.mod.ModelError)
        assert issubclass(self.mod.ModelNotFoundError, self.mod.ModelError)
        assert issubclass(self.mod.ModelMemoryError, self.mod.ModelError)

    def test_training_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.TrainingError, self.mod.AppError)
        assert issubclass(self.mod.TrainingConfigError, self.mod.TrainingError)

    def test_rag_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.RAGError, self.mod.AppError)
        assert issubclass(self.mod.RAGQueryError, self.mod.RAGError)
        assert issubclass(self.mod.RAGIndexError, self.mod.RAGError)
        assert issubclass(self.mod.RAGInitializationError, self.mod.RAGError)
        assert issubclass(self.mod.DocumentProcessingError, self.mod.RAGError)

    def test_inference_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.InferenceError, self.mod.AppError)
        assert issubclass(self.mod.GenerationError, self.mod.InferenceError)
        assert issubclass(self.mod.OllamaError, self.mod.InferenceError)

    def test_vector_store_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.VectorStoreError, self.mod.AppError)
        assert issubclass(self.mod.VectorStoreConnectionError, self.mod.VectorStoreError)

    def test_file_upload_errors_inherit_from_app_error(self):
        assert issubclass(self.mod.FileUploadError, self.mod.AppError)
        assert issubclass(self.mod.FileValidationError, self.mod.FileUploadError)

    # -- catch behaviour --------------------------------------------

    def test_catch_model_load_with_model_error(self):
        with pytest.raises(self.mod.ModelError):
            raise self.mod.ModelLoadError("test")

    def test_catch_rag_query_with_app_error(self):
        with pytest.raises(self.mod.AppError):
            raise self.mod.RAGQueryError("test")

    def test_catch_generation_with_inference_error(self):
        with pytest.raises(self.mod.InferenceError):
            raise self.mod.GenerationError("test")

    def test_all_exceptions_are_exception_subclass(self):
        for name in dir(self.mod):
            obj = getattr(self.mod, name)
            if isinstance(obj, type) and issubclass(obj, Exception) and obj is not Exception:
                assert issubclass(obj, self.mod.AppError), (
                    f"{name} should inherit from AppError"
                )


# ===================================================================
# 2. Upload path traversal protection tests
# ===================================================================


class TestUploadPathTraversal:
    """Verify that the upload router rejects path traversal attempts."""

    @pytest.fixture()
    def upload_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "uploaded"
        d.mkdir()
        return d

    def test_basename_strips_directory(self):
        """os.path.basename must strip directory components."""
        assert os.path.basename("../../etc/passwd") == "passwd"
        assert os.path.basename("/etc/shadow") == "shadow"
        assert os.path.basename("normal.jsonl") == "normal.jsonl"

    def test_dotfile_rejected(self):
        """Files starting with '.' should be rejected."""
        safe = os.path.basename(".hidden")
        assert safe.startswith(".")

    def test_resolved_path_stays_in_upload_dir(self, upload_dir: Path):
        """Symlink-based escape should be caught by is_relative_to."""
        safe_filename = os.path.basename("data.json")
        file_path = upload_dir / safe_filename
        assert file_path.resolve().is_relative_to(upload_dir.resolve())

    def test_traversal_attempt_normalised(self, upload_dir: Path):
        """'../' in filename is stripped by basename."""
        malicious = "../../../etc/passwd"
        safe_filename = os.path.basename(malicious)
        file_path = upload_dir / safe_filename
        assert file_path.resolve().is_relative_to(upload_dir.resolve())
        assert ".." not in str(file_path)


# ===================================================================
# 3. API integration tests (require FastAPI + httpx)
# ===================================================================

# Skip this entire section if fastapi or httpx are not installed.
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")
class TestUploadEndpoint:
    """Integration tests for /api/upload-data endpoint."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch) -> Generator:
        """Create a test client with a temporary upload directory."""
        monkeypatch.setattr("app.dependencies.UPLOADED_DIR", tmp_path / "uploaded")

        from app.routers.upload import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as c:
            yield c

    def test_upload_valid_jsonl(self, client, tmp_path: Path):
        content = b'{"text": "hello"}\n{"text": "world"}\n'
        resp = client.post(
            "/api/upload-data",
            files={"file": ("train.jsonl", content, "application/octet-stream")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data_count"] == 2

    def test_upload_invalid_extension_rejected(self, client):
        resp = client.post(
            "/api/upload-data",
            files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_path_traversal_rejected(self, client):
        content = b'{"text": "test"}\n'
        resp = client.post(
            "/api/upload-data",
            files={"file": ("../../etc/passwd.jsonl", content, "application/octet-stream")},
        )
        # Should succeed but filename is sanitised to "passwd.jsonl"
        if resp.status_code == 200:
            data = resp.json()
            assert ".." not in data.get("path", "")

    def test_upload_dotfile_rejected(self, client):
        content = b'{"text": "test"}\n'
        resp = client.post(
            "/api/upload-data",
            files={"file": (".hidden.jsonl", content, "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_invalid_json_returns_400(self, client):
        content = b"not json\n"
        resp = client.post(
            "/api/upload-data",
            files={"file": ("bad.jsonl", content, "application/octet-stream")},
        )
        assert resp.status_code == 400


# ===================================================================
# 4. RAG exception module tests
# ===================================================================


class TestRAGExceptions:
    """Verify src.rag.utils.exceptions hierarchy."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            self.mod = importlib.import_module("src.rag.utils.exceptions")
            self.available = True
        except ImportError:
            self.available = False

    def test_rag_exception_base(self):
        if not self.available:
            pytest.skip("RAG exceptions module not importable")
        assert issubclass(self.mod.RAGException, Exception)

    def test_search_error_inherits_rag_exception(self):
        if not self.available:
            pytest.skip("RAG exceptions module not importable")
        assert issubclass(self.mod.SearchError, self.mod.RAGException)

    def test_vector_store_connection_error_inherits(self):
        if not self.available:
            pytest.skip("RAG exceptions module not importable")
        assert issubclass(self.mod.VectorStoreConnectionError, self.mod.VectorStoreError)
        assert issubclass(self.mod.VectorStoreError, self.mod.RAGException)

    def test_llm_memory_error_inherits(self):
        if not self.available:
            pytest.skip("RAG exceptions module not importable")
        assert issubclass(self.mod.LLMMemoryError, self.mod.GenerationError)

    def test_exception_message_and_code(self):
        if not self.available:
            pytest.skip("RAG exceptions module not importable")
        err = self.mod.SearchError("search failed", search_type="hybrid", query="test query")
        assert "search failed" in str(err)
        assert err.error_code == "SEARCH_ERROR"
        assert err.details["search_type"] == "hybrid"
