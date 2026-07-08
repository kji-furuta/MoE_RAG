"""Pydantic models for the fine-tuning subsystem."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Request payload for starting a fine-tuning job."""

    model_name: str
    training_data: List[str]
    training_method: str = Field(
        "lora",
        description="Training strategy identifier (lora, qlora, full, continual)",
    )
    lora_config: Dict[str, Any]
    training_config: Dict[str, Any]


class GenerationRequest(BaseModel):
    """Payload for inference against a fine-tuned model."""

    model_path: str
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


class TrainingStatus(BaseModel):
    """Runtime status of a fine-tuning job."""

    task_id: str
    status: str
    progress: float
    message: str
    model_path: Optional[str] = None
