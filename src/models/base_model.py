"""
Base model class for training abstraction
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import PreTrainedModel


class BaseModel(nn.Module):
    """
    Base model wrapper for unified training interface
    Wraps HuggingFace PreTrainedModel instances
    """
    
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.config = model.config if hasattr(model, 'config') else None
        
    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model"""
        return self.model(*args, **kwargs)
    
    def get_model(self) -> PreTrainedModel:
        """Get the underlying PreTrainedModel"""
        return self.model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model using HuggingFace's save_pretrained method"""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_directory, **kwargs)
        else:
            # Fallback to torch.save
            torch.save(self.model.state_dict(), f"{save_directory}/pytorch_model.bin")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing if supported"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing if supported"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
    
    @property
    def device(self):
        """Get the device of the model"""
        return next(self.parameters()).device
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self