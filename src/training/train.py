"""
Main training script for fine-tuning models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class ModelTrainer:
    """Main trainer class for fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config['hardware']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name']
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=config['model']['num_labels']
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def prepare_dataset(self, file_path, is_training=True):
        """Load and prepare dataset"""
        # This is a placeholder - adjust based on your data format
        if file_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=file_path)['train']
        elif file_path.endswith('.json'):
            dataset = load_dataset('json', data_files=file_path)['train']
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config['data']['text_column']],
                padding='max_length',
                truncation=True,
                max_length=self.config['model']['max_length']
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config['data']['preprocessing_num_workers']
        )
        
        # Rename label column if necessary
        if self.config['data']['label_column'] != 'labels':
            tokenized_dataset = tokenized_dataset.rename_column(
                self.config['data']['label_column'], 'labels'
            )
        
        return tokenized_dataset
    
    def train(self):
        """Main training function"""
        logger.info("Starting training...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(
            self.config['data']['train_file'], 
            is_training=True
        )
        eval_dataset = self.prepare_dataset(
            self.config['data']['validation_file'], 
            is_training=False
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['paths']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=self.config['paths']['logging_dir'],
            logging_steps=self.config['training']['logging_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            fp16=self.config['training']['fp16'] and torch.cuda.is_available(),
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            seed=self.config['hardware']['seed'],
            report_to=["wandb"] if self.config['tracking']['use_wandb'] else [],
            run_name=f"fine-tuning-{self.config['model']['name']}",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save the best model
        best_model_path = Path(self.config['paths']['model_save_dir']) / "best_model"
        trainer.save_model(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        
        logger.info(f"Training completed. Best model saved to {best_model_path}")
        
        # Evaluate on test set if available
        if 'test_file' in self.config['data'] and self.config['data']['test_file']:
            test_dataset = self.prepare_dataset(
                self.config['data']['test_file'], 
                is_training=False
            )
            test_results = trainer.evaluate(test_dataset)
            logger.info(f"Test results: {test_results}")
        
        return trainer


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fine-tune a model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create necessary directories
    for path_key in ['output_dir', 'logging_dir', 'cache_dir', 'model_save_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run trainer
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
