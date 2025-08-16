import torch
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_available_device() -> torch.device:
    """利用可能なデバイス（GPU/MPS/CPU）を取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def get_gpu_memory_info() -> Dict[str, Any]:
    """GPU ����1�֗"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        memory_total = props.total_memory / 1024**3  # GB
        
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": memory_total,
            "allocated_memory_gb": memory_allocated,
            "reserved_memory_gb": memory_reserved,
            "free_memory_gb": memory_total - memory_allocated,
            "multi_processor_count": props.multi_processor_count,
        })
    
    return info


def optimize_model_for_gpu(
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    enable_mixed_precision: bool = True,
    gradient_checkpointing: bool = False
) -> Tuple[torch.nn.Module, torch.device]:
    """GPU"""
    if device is None:
        device = get_available_device()
    
    # モデルをGPUに転送
    model = model.to(device)
    
    # Mixed Precision Training
    if enable_mixed_precision and device.type == "cuda":
        logger.info("Enabling mixed precision training")
        model = model.half()  # FP16
    
    # Gradient Checkpointing n-�
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # GPUキャッシュをクリア
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return model, device


def clear_gpu_memory():
    """GPU メモリをクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Cleared GPU memory cache")


def set_memory_fraction(fraction: float = 0.9):
    """GPU メモリをクリア"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"Set GPU memory fraction to {fraction * 100}%")