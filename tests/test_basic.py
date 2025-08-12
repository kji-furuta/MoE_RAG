"""
Test suite for the AI Fine-Tuning project
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.helpers import set_seed, MetricTracker, EarlyStopping
from data.dataloader import DataLoader, DataAugmenter


class TestUtils:
    """Test utility functions"""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility"""
        import random
        import numpy as np
        
        set_seed(42)
        random_val1 = random.random()
        np_val1 = np.random.random()
        
        set_seed(42)
        random_val2 = random.random()
        np_val2 = np.random.random()
        
        assert random_val1 == random_val2
        assert np_val1 == np_val2
    
    def test_metric_tracker(self):
        """Test metric tracking"""
        tracker = MetricTracker(['loss', 'accuracy'])
        
        tracker.update({'loss': 0.5, 'accuracy': 0.8})
        tracker.update({'loss': 0.3, 'accuracy': 0.9})
        
        assert tracker.get_best('loss') == 0.3
        assert tracker.get_best('accuracy') == 0.9
        assert len(tracker.history['loss']) == 2
    
    def test_early_stopping(self):
        """Test early stopping functionality"""
        # Test for minimizing loss
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        assert not early_stopping(0.5)  # First value
        assert not early_stopping(0.4)  # Improvement
        assert not early_stopping(0.41)  # No improvement (counter=1)
        assert not early_stopping(0.42)  # No improvement (counter=2)
        assert early_stopping(0.43)  # Should stop (counter=3)


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_data_augmentation(self):
        """Test text augmentation methods"""
        text = "This is a test sentence."
        
        # Test noise addition
        noisy_text = DataAugmenter.add_noise(text, noise_level=0.2)
        assert len(noisy_text) == len(text)
        
        # Test random deletion
        deleted_text = DataAugmenter.random_deletion(text, prob=0.2)
        assert len(deleted_text) <= len(text)
        
        # Test random insertion
        inserted_text = DataAugmenter.random_insertion(text, n=2)
        assert len(inserted_text.split()) >= len(text.split())
    
    def test_label_mapping(self):
        """Test label to integer mapping"""
        loader = DataLoader()
        labels = ['positive', 'negative', 'neutral', 'positive']
        
        label_map = loader.create_label_mapping(labels)
        
        assert len(label_map) == 3  # Three unique labels
        assert all(isinstance(v, int) for v in label_map.values())
        assert loader.label_map_reverse[label_map['positive']] == 'positive'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
