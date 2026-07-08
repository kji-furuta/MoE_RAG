"""Shared runtime state for the fine-tuning subsystem."""

from __future__ import annotations

from typing import Any, Dict, Union

from .models import TrainingStatus

# NOTE: Existing code occasionally stores plain dictionaries for compatibility
# with historical endpoints. We keep the union type until those callers are
# migrated to structured models.
TrainingTaskEntry = Union[TrainingStatus, Dict[str, Any]]


# Global in-memory registry used by API handlers and background tasks.
training_tasks: Dict[str, TrainingTaskEntry] = {}


__all__ = ["TrainingTaskEntry", "training_tasks"]
