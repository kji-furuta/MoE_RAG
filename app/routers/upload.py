"""Standalone data upload API endpoints.

Separated from the training endpoints so that dataset uploads remain
available even when optional fine-tuning dependencies (e.g. GPU tooling)
are not installed inside the runtime environment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.dependencies import UPLOADED_DIR


router = APIRouter(prefix="/api", tags=["dataset-upload"])

JST = timezone(timedelta(hours=9))
logger = logging.getLogger("app.routers.upload")


@router.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)) -> dict:
    """トレーニングデータをアップロード"""

    try:
        logger.info("ファイルアップロード開始: %s", file.filename)

        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が不正です")

        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます (最大100MB)")

        UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOADED_DIR / file.filename

        content = await file.read()
        logger.info("ファイル保存: %s, サイズ: %s bytes", file_path, len(content))

        with open(file_path, "wb") as f:
            f.write(content)

        sample_data: List[dict] = []
        data_count = 0

        if file.filename.endswith(".jsonl"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                data_count = len(lines)

                for i, line in enumerate(lines[:5]):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.error("JSON parse error at line %s: %s", i + 1, exc)
                        raise HTTPException(
                            status_code=400,
                            detail=f"行 {i + 1} でJSONパースエラー: {exc}",
                        ) from exc
                    sample_data.append(data)

                logger.info("JSONL解析完了: %s行, サンプル: %s件", data_count, len(sample_data))

            except UnicodeDecodeError as exc:
                logger.error("ファイルエンコーディングエラー: %s", exc)
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)") from exc

        elif file.filename.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    data_count = len(data)
                    sample_data = data[:3]
                else:
                    data_count = 1
                    sample_data = [data]

                logger.info("JSON解析完了: %s件, サンプル: %s件", data_count, len(sample_data))

            except json.JSONDecodeError as exc:
                logger.error("JSON parse error: %s", exc)
                raise HTTPException(status_code=400, detail=f"JSONパースエラー: {exc}") from exc
            except UnicodeDecodeError as exc:
                logger.error("ファイルエンコーディングエラー: %s", exc)
                raise HTTPException(status_code=400, detail="ファイルのエンコーディングが不正です (UTF-8を使用してください)") from exc
        else:
            raise HTTPException(status_code=400, detail="サポートされていないファイル形式です (.jsonl または .json を使用してください)")

        result = {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content),
            "data_count": data_count,
            "sample_data": sample_data[:3],
            "uploaded_at": datetime.now(JST).isoformat(),
        }

        logger.info("アップロード成功: %s", result)
        return result

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Upload error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"アップロードエラー: {exc}") from exc
