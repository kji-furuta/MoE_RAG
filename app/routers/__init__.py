"""
APIルーターモジュール
各機能のAPIエンドポイントを分離して管理
"""

from .finetuning import router as finetuning_router
from .generation import router as generation_router
from .models import router as models_router
from .moe_dataset import router as moe_dataset_router
from .monitoring import router as monitoring_router
from .pages import router as pages_router
from .rag import router as rag_router
from .rlanything import router as rlanything_router
from .upload import router as upload_router

__all__ = [
    "finetuning_router",
    "generation_router",
    "models_router",
    "moe_dataset_router",
    "monitoring_router",
    "pages_router",
    "rag_router",
    "rlanything_router",
    "upload_router",
]
