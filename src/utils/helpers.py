"""
Utility functions for the project
"""

import os
import json
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def save_json(data: Dict, file_path: str):
    """Save dictionary to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_yaml(data: Dict, file_path: str):
    """Save dictionary to YAML file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Saved YAML to {file_path}")


def load_yaml(file_path: str) -> Dict:
    """Load YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create a directory for experiment outputs"""
    base_dir = Path(base_dir)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_dir = base_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "predictions").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


class MetricTracker:
    """Track and save training metrics"""
    
    def __init__(self, metrics: list = None):
        self.metrics = metrics or ['loss', 'accuracy', 'f1']
        self.history = {metric: [] for metric in self.metrics}
        self.best_values = {metric: None for metric in self.metrics}
    
    def update(self, metric_dict: Dict[str, float]):
        """Update metrics with new values"""
        for metric, value in metric_dict.items():
            if metric in self.history:
                self.history[metric].append(value)
                
                # Update best value
                if self.best_values[metric] is None:
                    self.best_values[metric] = value
                else:
                    # Assume higher is better for accuracy/f1, lower for loss
                    if 'loss' in metric:
                        self.best_values[metric] = min(self.best_values[metric], value)
                    else:
                        self.best_values[metric] = max(self.best_values[metric], value)
    
    def get_best(self, metric: str) -> Optional[float]:
        """Get best value for a metric"""
        return self.best_values.get(metric)
    
    def save(self, file_path: str):
        """Save metrics history to file"""
        data = {
            'history': self.history,
            'best_values': self.best_values
        }
        save_json(data, file_path)
    
    def load(self, file_path: str):
        """Load metrics history from file"""
        data = load_json(file_path)
        self.history = data['history']
        self.best_values = data['best_values']


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 0.0001,
                 mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/f1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test experiment directory creation
    exp_dir = create_experiment_dir("./experiments")
    print(f"Experiment directory: {exp_dir}")
    
    # Test metric tracker
    tracker = MetricTracker(['loss', 'accuracy'])
    tracker.update({'loss': 0.5, 'accuracy': 0.85})
    tracker.update({'loss': 0.3, 'accuracy': 0.90})
    print(f"Best accuracy: {tracker.get_best('accuracy')}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    for loss in [0.5, 0.4, 0.35, 0.36, 0.37, 0.38]:
        should_stop = early_stopping(loss)
        print(f"Loss: {loss}, Should stop: {should_stop}")
        if should_stop:
            break
    
    print("All tests completed!")
