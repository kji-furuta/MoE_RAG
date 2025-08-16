"""
Inference script for making predictions with fine-tuned models
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
from typing import List, Dict, Union
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Class for running inference with fine-tuned models"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model
            device: Device to use (cuda/cpu)
        """
        self.model_path = Path(model_path)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping if exists
        label_map_path = self.model_path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
                self.label_map_reverse = {v: k for k, v in self.label_map.items()}
        else:
            self.label_map = None
            self.label_map_reverse = None
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict_single(self, text: str, return_probs: bool = False) -> Dict:
        """
        Make prediction for a single text
        
        Args:
            text: Input text
            return_probs: Whether to return probability scores
        
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_idx = probs.argmax().item()
            confidence = probs[0, predicted_idx].item()
        
        # Get label
        if self.label_map_reverse:
            predicted_label = self.label_map_reverse[predicted_idx]
        else:
            predicted_label = str(predicted_idx)
        
        result = {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence
        }
        
        if return_probs:
            result['probabilities'] = probs.cpu().numpy()[0].tolist()
        
        return result
    
    def predict_batch(self, 
                     texts: List[str], 
                     batch_size: int = 32,
                     return_probs: bool = False) -> List[Dict]:
        """
        Make predictions for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            return_probs: Whether to return probability scores
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_indices = probs.argmax(dim=-1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                predicted_idx = predicted_indices[j].item()
                confidence = probs[j, predicted_idx].item()
                
                if self.label_map_reverse:
                    predicted_label = self.label_map_reverse[predicted_idx]
                else:
                    predicted_label = str(predicted_idx)
                
                result = {
                    'text': text,
                    'predicted_label': predicted_label,
                    'confidence': confidence
                }
                
                if return_probs:
                    result['probabilities'] = probs[j].cpu().numpy().tolist()
                
                results.append(result)
        
        return results
    
    def predict_file(self, 
                    input_file: str,
                    output_file: str,
                    text_column: str = 'text',
                    batch_size: int = 32,
                    return_probs: bool = False):
        """
        Make predictions for texts in a file
        
        Args:
            input_file: Path to input file (CSV or JSON)
            output_file: Path to save predictions
            text_column: Name of text column
            batch_size: Batch size for inference
            return_probs: Whether to return probability scores
        """
        # Load data
        input_path = Path(input_file)
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix == '.json':
            df = pd.read_json(input_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input file")
        
        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size, return_probs)
        
        # Add predictions to dataframe
        df['predicted_label'] = [p['predicted_label'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
        
        if return_probs:
            df['probabilities'] = [p['probabilities'] for p in predictions]
        
        # Save results
        output_path = Path(output_file)
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', lines=True)
        
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(df['predicted_label'].value_counts())
        print(f"\nAverage confidence: {df['confidence'].mean():.3f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with texts to predict"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.csv",
        help="Output file for predictions"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to predict"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in input file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--return_probs",
        action="store_true",
        help="Return probability scores"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ModelInference(args.model_path, args.device)
    
    if args.text:
        # Single text prediction
        result = inference.predict_single(args.text, args.return_probs)
        print("\nPrediction Result:")
        print(f"Text: {result['text']}")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if args.return_probs:
            print(f"Probabilities: {result['probabilities']}")
    
    elif args.input_file:
        # File prediction
        inference.predict_file(
            args.input_file,
            args.output_file,
            args.text_column,
            args.batch_size,
            args.return_probs
        )
    
    else:
        # Interactive mode
        print("\nInteractive Mode (type 'quit' to exit)")
        while True:
            text = input("\nEnter text: ")
            if text.lower() == 'quit':
                break
            
            result = inference.predict_single(text, args.return_probs)
            print(f"Predicted Label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    main()
