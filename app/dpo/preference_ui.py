#!/usr/bin/env python3
"""
DPO Preference Data収集UI
FastAPI endpoints for collecting human preference data
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# FastAPI Router
router = APIRouter(prefix="/api/dpo", tags=["DPO"])

# Preference dataのデフォルト保存先
PREFERENCE_DATA_DIR = Path("data/dpo")
PREFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)


class PreferenceData(BaseModel):
    """
    Preference dataの構造
    DPOに必要な3つのフィールド: prompt, chosen, rejected
    """
    prompt: str = Field(..., description="モデルへの入力プロンプト")
    chosen: str = Field(..., description="選好される「良い」応答")
    rejected: str = Field(..., description="選好されない「悪い」応答")
    margin: Optional[float] = Field(None, description="選好の確信度（オプション）")
    metadata: Optional[dict] = Field(None, description="追加のメタデータ（オプション）")


class PreferenceDataBatch(BaseModel):
    """複数のpreference dataを一括登録"""
    preferences: List[PreferenceData] = Field(..., description="Preference dataのリスト")
    dataset_name: str = Field("preference_dataset", description="データセット名")


@router.post("/collect-preference")
async def collect_preference(data: PreferenceData):
    """
    単一のpreference dataを収集

    Args:
        data: PreferenceData (prompt, chosen, rejected, margin)

    Returns:
        成功メッセージとデータID
    """
    try:
        # JSONLファイルのパス
        jsonl_file = PREFERENCE_DATA_DIR / "preference_dataset.jsonl"

        # JSONL形式で追記
        preference_entry = {
            "prompt": data.prompt,
            "chosen": data.chosen,
            "rejected": data.rejected,
        }

        # オプションフィールド
        if data.margin is not None:
            preference_entry["margin"] = data.margin

        if data.metadata:
            preference_entry["metadata"] = data.metadata

        # タイムスタンプ追加
        preference_entry["collected_at"] = datetime.now().isoformat()

        # JSONL追記
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(preference_entry, ensure_ascii=False) + "\n")

        logger.info(f"Preference data collected: {jsonl_file}")

        return {
            "status": "success",
            "message": "Preference data collected successfully",
            "file": str(jsonl_file),
            "data": preference_entry
        }

    except Exception as e:
        logger.error(f"Error collecting preference data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error collecting preference data: {str(e)}")


@router.post("/collect-preference-batch")
async def collect_preference_batch(batch: PreferenceDataBatch):
    """
    複数のpreference dataを一括収集

    Args:
        batch: PreferenceDataBatch (preferences list, dataset_name)

    Returns:
        成功メッセージと収集数
    """
    try:
        # JSONLファイルのパス
        jsonl_file = PREFERENCE_DATA_DIR / f"{batch.dataset_name}.jsonl"

        collected_count = 0

        with open(jsonl_file, "a", encoding="utf-8") as f:
            for pref in batch.preferences:
                preference_entry = {
                    "prompt": pref.prompt,
                    "chosen": pref.chosen,
                    "rejected": pref.rejected,
                }

                if pref.margin is not None:
                    preference_entry["margin"] = pref.margin

                if pref.metadata:
                    preference_entry["metadata"] = pref.metadata

                preference_entry["collected_at"] = datetime.now().isoformat()

                f.write(json.dumps(preference_entry, ensure_ascii=False) + "\n")
                collected_count += 1

        logger.info(f"Batch preference data collected: {collected_count} entries to {jsonl_file}")

        return {
            "status": "success",
            "message": f"Collected {collected_count} preference entries",
            "file": str(jsonl_file),
            "count": collected_count
        }

    except Exception as e:
        logger.error(f"Error collecting batch preference data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error collecting batch preference data: {str(e)}")


@router.get("/preference-datasets")
async def list_preference_datasets():
    """
    利用可能なpreference datasetsのリストを取得

    Returns:
        データセットのリストとサンプル数
    """
    try:
        datasets = []

        for jsonl_file in PREFERENCE_DATA_DIR.glob("*.jsonl"):
            # ファイルの行数をカウント（サンプル数）
            with open(jsonl_file, "r", encoding="utf-8") as f:
                sample_count = sum(1 for line in f if line.strip())

            datasets.append({
                "name": jsonl_file.stem,
                "path": str(jsonl_file),
                "sample_count": sample_count,
                "file_size_kb": jsonl_file.stat().st_size / 1024
            })

        return {
            "status": "success",
            "datasets": datasets,
            "total_datasets": len(datasets)
        }

    except Exception as e:
        logger.error(f"Error listing preference datasets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.get("/preference-dataset/{dataset_name}")
async def get_preference_dataset(dataset_name: str, limit: int = 100):
    """
    特定のpreference datasetの内容を取得

    Args:
        dataset_name: データセット名
        limit: 取得する最大サンプル数

    Returns:
        Preference dataのリスト
    """
    try:
        jsonl_file = PREFERENCE_DATA_DIR / f"{dataset_name}.jsonl"

        if not jsonl_file.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        preferences = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    preferences.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {i+1}: {str(e)}")
                    continue

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "sample_count": len(preferences),
            "preferences": preferences
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preference dataset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting dataset: {str(e)}")


@router.delete("/preference-dataset/{dataset_name}")
async def delete_preference_dataset(dataset_name: str):
    """
    Preference datasetを削除

    Args:
        dataset_name: データセット名

    Returns:
        削除成功メッセージ
    """
    try:
        jsonl_file = PREFERENCE_DATA_DIR / f"{dataset_name}.jsonl"

        if not jsonl_file.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        # ファイル削除
        jsonl_file.unlink()

        logger.info(f"Preference dataset deleted: {jsonl_file}")

        return {
            "status": "success",
            "message": f"Dataset '{dataset_name}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting preference dataset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")


@router.post("/upload-preference-file")
async def upload_preference_file(file: UploadFile = File(...)):
    """
    Preference datasetのJSONLファイルをアップロード

    Args:
        file: アップロードするJSONLファイル

    Returns:
        アップロード成功メッセージとサンプル数
    """
    try:
        # ファイル名の検証
        if not file.filename.endswith(".jsonl"):
            raise HTTPException(status_code=400, detail="Only .jsonl files are allowed")

        # 保存先パス
        save_path = PREFERENCE_DATA_DIR / file.filename

        # ファイル保存
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # サンプル数をカウント
        with open(save_path, "r", encoding="utf-8") as f:
            sample_count = sum(1 for line in f if line.strip())

        logger.info(f"Preference file uploaded: {save_path}, {sample_count} samples")

        return {
            "status": "success",
            "message": "File uploaded successfully",
            "filename": file.filename,
            "path": str(save_path),
            "sample_count": sample_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading preference file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


# 統計情報取得
@router.get("/stats")
async def get_preference_stats():
    """
    Preference dataの統計情報を取得

    Returns:
        データセット数、総サンプル数、ディレクトリサイズ
    """
    try:
        total_samples = 0
        total_size = 0
        dataset_count = 0

        for jsonl_file in PREFERENCE_DATA_DIR.glob("*.jsonl"):
            dataset_count += 1
            total_size += jsonl_file.stat().st_size

            with open(jsonl_file, "r", encoding="utf-8") as f:
                total_samples += sum(1 for line in f if line.strip())

        return {
            "status": "success",
            "dataset_count": dataset_count,
            "total_samples": total_samples,
            "total_size_mb": total_size / (1024 * 1024),
            "data_directory": str(PREFERENCE_DATA_DIR)
        }

    except Exception as e:
        logger.error(f"Error getting preference stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
