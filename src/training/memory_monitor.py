"""
Memory monitoring and reporting utilities
メモリ監視とレポートユーティリティ
"""
import torch
import psutil
import logging
from typing import Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """GPU and system memory monitoring"""
    
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """Get system RAM information"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent
        }
    
    @staticmethod
    def get_gpu_memory(device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information for a specific device"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            free_info = torch.cuda.mem_get_info(device_id)
            
            return {
                'device_name': props.name,
                'total_gb': total,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free_info[0] / 1024**3,
                'used_percent': (allocated / total) * 100 if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {}
    
    @staticmethod
    def get_all_gpu_memory() -> Dict[int, Dict[str, float]]:
        """Get memory information for all GPUs"""
        if not torch.cuda.is_available():
            return {}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_info[i] = MemoryMonitor.get_gpu_memory(i)
        return gpu_info
    
    @staticmethod
    def format_memory_error(error: Exception) -> str:
        """
        Format CUDA OOM error message, fixing the abnormal memory reporting bug
        
        Args:
            error: The CUDA OOM exception
            
        Returns:
            Formatted error message with correct memory values
        """
        error_str = str(error)
        
        # Fix the abnormal GiB reporting (17179869184.00 GiB bug)
        # This is caused by incorrect integer overflow in PyTorch's error message
        pattern = r'(\d{10,})\.\d+ GiB'
        
        def fix_memory_value(match):
            value = float(match.group(1))
            # If value is absurdly large (> 1000000 GiB), it's likely a bug
            if value > 1000000:
                # Try to extract the actual memory from other parts of the message
                return "unknown"
            return f"{value:.2f} GiB"
        
        fixed_error = re.sub(pattern, fix_memory_value, error_str)
        
        # Also add actual memory status
        memory_status = MemoryMonitor.get_memory_summary()
        
        return f"{fixed_error}\n\n実際のメモリ状況:\n{memory_status}"
    
    @staticmethod
    def get_memory_summary() -> str:
        """Get a summary of current memory usage"""
        lines = []
        
        # System memory
        sys_mem = MemoryMonitor.get_system_memory()
        lines.append(f"システムRAM: {sys_mem['used_gb']:.1f}/{sys_mem['total_gb']:.1f} GB ({sys_mem['percent']:.1f}%)")
        
        # GPU memory
        gpu_info = MemoryMonitor.get_all_gpu_memory()
        for gpu_id, info in gpu_info.items():
            if info:
                lines.append(
                    f"GPU {gpu_id} ({info['device_name']}): "
                    f"{info['allocated_gb']:.1f}/{info['total_gb']:.1f} GB "
                    f"({info['used_percent']:.1f}%)"
                )
        
        return "\n".join(lines)
    
    @staticmethod
    def check_memory_availability(required_gb: float = 10.0) -> Tuple[bool, str]:
        """
        Check if sufficient memory is available
        
        Args:
            required_gb: Required free memory in GB
            
        Returns:
            (is_available, message)
        """
        gpu_info = MemoryMonitor.get_all_gpu_memory()
        
        for gpu_id, info in gpu_info.items():
            if info and info.get('free_gb', 0) >= required_gb:
                return True, f"GPU {gpu_id} has {info['free_gb']:.1f} GB free"
        
        # Check if we can free up memory
        total_free = sum(info.get('free_gb', 0) for info in gpu_info.values() if info)
        
        if total_free >= required_gb:
            return False, f"合計 {total_free:.1f} GB の空きメモリがありますが、単一GPUでは不足しています。マルチGPU分散を使用してください。"
        else:
            return False, f"メモリ不足: {required_gb:.1f} GB必要ですが、{total_free:.1f} GBしか利用できません。"
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")


def monitor_memory_during_training(func):
    """
    Decorator to monitor memory during training
    
    Usage:
        @monitor_memory_during_training
        def train_model(...):
            ...
    """
    def wrapper(*args, **kwargs):
        # Log initial memory state
        logger.info(f"Memory before training:\n{MemoryMonitor.get_memory_summary()}")
        
        try:
            result = func(*args, **kwargs)
            
            # Log final memory state
            logger.info(f"Memory after training:\n{MemoryMonitor.get_memory_summary()}")
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM error with proper formatting
            formatted_error = MemoryMonitor.format_memory_error(e)
            logger.error(f"CUDA OOM Error (formatted):\n{formatted_error}")
            
            # Try to clear memory
            MemoryMonitor.clear_gpu_memory()
            
            # Re-raise with better message
            raise RuntimeError(formatted_error) from e
        
    return wrapper