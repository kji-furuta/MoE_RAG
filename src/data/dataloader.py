"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """General data loader for various formats"""
    
    def __init__(self, 
                 text_column: str = "text",
                 label_column: str = "label",
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        Initialize DataLoader
        
        Args:
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.text_column = text_column
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.label_map = {}
        self.label_map_reverse = {}
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def load_json(self, file_path: str) -> pd.DataFrame:
        """Load data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_json(file_path, lines=True)
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def create_label_mapping(self, labels: List) -> Dict:
        """Create mapping from labels to integers"""
        unique_labels = sorted(set(labels))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.label_map_reverse = {i: label for label, i in self.label_map.items()}
        logger.info(f"Created label mapping for {len(unique_labels)} unique labels")
        return self.label_map
    
    def prepare_data(self, 
                    file_path: str,
                    split: bool = True) -> Union[Dataset, DatasetDict]:
        """
        Load and prepare data for training
        
        Args:
            file_path: Path to data file
            split: Whether to split into train/val/test
        
        Returns:
            Dataset or DatasetDict with prepared data
        """
        # Load data based on file extension
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            df = self.load_csv(str(file_path))
        elif file_path.suffix == '.json':
            df = self.load_json(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Check required columns
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in data")
        
        # Clean data
        df = df.dropna(subset=[self.text_column, self.label_column])
        df[self.text_column] = df[self.text_column].astype(str)
        
        # Create label mapping if not exists
        if not self.label_map:
            self.create_label_mapping(df[self.label_column].tolist())
        
        # Convert labels to integers
        df['labels'] = df[self.label_column].map(self.label_map)
        
        if split:
            # Split data
            train_df, test_df = train_test_split(
                df, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=df['labels']
            )
            
            train_df, val_df = train_test_split(
                train_df,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=train_df['labels']
            )
            
            # Create datasets
            dataset_dict = DatasetDict({
                'train': Dataset.from_pandas(train_df),
                'validation': Dataset.from_pandas(val_df),
                'test': Dataset.from_pandas(test_df)
            })
            
            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return dataset_dict
        else:
            # Return single dataset
            dataset = Dataset.from_pandas(df)
            logger.info(f"Created dataset with {len(df)} samples")
            return dataset
    
    def save_splits(self, 
                   dataset_dict: DatasetDict,
                   output_dir: str):
        """Save train/val/test splits to separate files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, dataset in dataset_dict.items():
            output_path = output_dir / f"{split_name}.csv"
            df = dataset.to_pandas()
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} split to {output_path}")


class DataAugmenter:
    """Data augmentation utilities for text data"""
    
    @staticmethod
    def add_noise(text: str, noise_level: float = 0.1) -> str:
        """Add random character swaps as noise"""
        import random
        
        text_list = list(text)
        num_swaps = int(len(text_list) * noise_level)
        
        for _ in range(num_swaps):
            if len(text_list) > 1:
                idx = random.randint(0, len(text_list) - 2)
                text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        
        return ''.join(text_list)
    
    @staticmethod
    def random_insertion(text: str, n: int = 1) -> str:
        """Randomly insert words into the text"""
        import random
        
        words = text.split()
        for _ in range(n):
            if words:
                random_word = random.choice(words)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, prob: float = 0.1) -> str:
        """Randomly delete words from the text"""
        import random
        
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > prob:
                new_words.append(word)
        
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)


def create_sample_data(output_path: str, num_samples: int = 1000):
    """Create sample data for testing"""
    import random
    
    # Sample texts and labels
    texts = [
        "This is a positive example of text.",
        "This is a negative example of text.",
        "Neutral text that doesn't lean either way.",
        "Another positive statement here.",
        "A clearly negative sentiment.",
    ]
    
    labels = ["positive", "negative", "neutral", "positive", "negative"]
    
    # Generate random samples
    data = []
    for _ in range(num_samples):
        idx = random.randint(0, len(texts) - 1)
        text = texts[idx]
        label = labels[idx]
        
        # Add some variation
        if random.random() > 0.5:
            text = DataAugmenter.add_noise(text, 0.05)
        
        data.append({"text": text, "label": label})
    
    # Save to file
    df = pd.DataFrame(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.json':
        df.to_json(output_path, orient='records', lines=True)
    
    logger.info(f"Created sample data with {num_samples} samples at {output_path}")
    return df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Data loading utilities")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample data")
    parser.add_argument("--output", type=str, default="data/sample.csv",
                       help="Output path for sample data")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to create")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.output, args.num_samples)
    else:
        # Test data loader
        loader = DataLoader()
        print("DataLoader initialized successfully")
