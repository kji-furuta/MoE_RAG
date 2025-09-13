"""
Error recovery and resilience utilities for training pipelines
"""
import torch
import gc
import logging
import traceback
from typing import Optional, Callable, Any, Dict
from functools import wraps
import psutil
import time

logger = logging.getLogger(__name__)


class TrainingErrorRecovery:
    """Training error recovery manager"""
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleared")
    
    @staticmethod
    def check_memory_availability(required_gb: float = 8.0) -> bool:
        """Check if sufficient GPU memory is available"""
        if not torch.cuda.is_available():
            return False
        
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.mem_get_info(i)[0] / 1024**3
            if free_memory >= required_gb:
                return True
        return False
    
    @staticmethod
    def reduce_batch_size(current_batch_size: int, min_batch_size: int = 1) -> int:
        """Reduce batch size on OOM error"""
        new_batch_size = max(current_batch_size // 2, min_batch_size)
        logger.warning(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        return new_batch_size
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """Enable gradient checkpointing to save memory"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on wrapped model")


def with_error_recovery(max_retries: int = 3, 
                        reduce_batch_on_oom: bool = True,
                        clear_memory_on_error: bool = True):
    """
    Decorator for automatic error recovery in training functions
    
    Args:
        max_retries: Maximum number of retry attempts
        reduce_batch_on_oom: Whether to reduce batch size on OOM errors
        clear_memory_on_error: Whether to clear GPU memory on errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            original_batch_size = kwargs.get('batch_size', None)
            
            for attempt in range(max_retries):
                try:
                    # Clear memory before retry
                    if attempt > 0 and clear_memory_on_error:
                        TrainingErrorRecovery.clear_gpu_memory()
                        time.sleep(2)  # Wait for memory to be freed
                    
                    return func(*args, **kwargs)
                    
                except torch.cuda.OutOfMemoryError as e:
                    last_error = e
                    logger.error(f"OOM Error on attempt {attempt + 1}: {str(e)}")
                    
                    if clear_memory_on_error:
                        TrainingErrorRecovery.clear_gpu_memory()
                    
                    # Try to reduce batch size
                    if reduce_batch_on_oom and original_batch_size:
                        new_batch_size = TrainingErrorRecovery.reduce_batch_size(
                            kwargs.get('batch_size', original_batch_size)
                        )
                        kwargs['batch_size'] = new_batch_size
                    
                    # Enable gradient checkpointing if model is available
                    if 'model' in kwargs:
                        TrainingErrorRecovery.enable_gradient_checkpointing(kwargs['model'])
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    if clear_memory_on_error and torch.cuda.is_available():
                        TrainingErrorRecovery.clear_gpu_memory()
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            # All retries failed
            logger.error(f"All {max_retries} attempts failed")
            raise last_error
        
        return wrapper
    return decorator


class CheckpointManager:
    """Manager for checkpoint saving and recovery"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(self, 
                       model, 
                       optimizer, 
                       epoch: int, 
                       step: int,
                       loss: float,
                       additional_info: Optional[Dict] = None):
        """Save training checkpoint with automatic cleanup"""
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'loss': loss,
            'timestamp': time.time()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                    
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        if not self.checkpoints:
            return None
        
        latest_checkpoint = self.checkpoints[-1]
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None


class MemoryMonitor:
    """Monitor memory usage during training"""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_used_gb': psutil.virtual_memory().used / 1024**3,
            'ram_available_gb': psutil.virtual_memory().available / 1024**3,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = torch.cuda.mem_get_info(i)[0] / 1024**3
                
                stats[f'gpu{i}_allocated_gb'] = allocated
                stats[f'gpu{i}_reserved_gb'] = reserved
                stats[f'gpu{i}_free_gb'] = free
        
        return stats
    
    @staticmethod
    def log_memory_stats(prefix: str = ""):
        """Log current memory statistics"""
        stats = MemoryMonitor.get_memory_stats()
        logger.info(f"{prefix} Memory Stats: {stats}")