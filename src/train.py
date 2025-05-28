"""
Multilingual Temporal Classification Training Script

This script trains a multilingual classifier to determine whether questions
have mutable or immutable answers across 7 languages.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import Dataset, DatasetDict, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, languages: List[str]):
        self.languages = languages
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format dataframe to long format.
        
        Args:
            df: DataFrame with columns ['label'] + language columns
            
        Returns:
            DataFrame with columns ['labels', 'language', 'text']
        """
        df_long = pd.melt(
            df,
            id_vars=['is_evergreen'],
            value_vars=self.languages,
            var_name='language',
            value_name='text'
        )
        df_long = df_long.rename(columns={'is_evergreen': 'labels'})
        return df_long
    
    def create_dataset_dict(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> DatasetDict:
        """Create HuggingFace DatasetDict from dataframes."""
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })
        
        # Ensure labels are int64
        for key in dataset_dict:
            dataset = dataset_dict[key]
            new_features = dataset.features.copy()
            new_features["labels"] = Value('int64')
            dataset_dict[key] = dataset.cast(new_features)
        
        return dataset_dict


class FocalLossTrainer(Trainer):
    """Custom trainer implementing focal loss."""
    
    def __init__(self, *args, class_weights=None, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Focal loss computation
        ce_loss = F.cross_entropy(
            logits, 
            labels, 
            weight=self.class_weights,
            reduction='none'
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss


class MultilingualClassifier:
    """Main classifier class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['model']['name']
        self.languages = config['data']['languages']
        self.tokenizer = None
        self.model = None
        self.data_processor = DataProcessor(self.languages)
        self.f1_metric = evaluate.load("f1")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test data."""
        train_df = pd.read_csv(self.config['data']['train_path'])
        val_df = pd.read_csv(self.config['data']['val_path'])
        test_df = pd.read_csv(self.config['data']['test_path'])
        
        # Process additional data if specified
        if 'additional_data' in self.config['data']:
            for path in self.config['data']['additional_data']:
                additional_df = pd.read_csv(path)
                train_df = pd.concat([train_df, additional_df], ignore_index=True)
        
        return train_df, val_df, test_df
    
    def prepare_data(self) -> DatasetDict:
        """Prepare data for training."""
        train_df, val_df, test_df = self.load_data()
        
        # Process dataframes
        train_processed = self.data_processor.process_dataframe(train_df)
        val_processed = self.data_processor.process_dataframe(val_df)
        test_processed = self.data_processor.process_dataframe(test_df)
        
        # Create dataset dict
        dataset_dict = self.data_processor.create_dataset_dict(
            train_processed, val_processed, test_processed
        )
        
        return dataset_dict
    
    def tokenize_function(self, examples):
        """Tokenize examples."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length'],
            return_tensors='pt'
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics including per-language F1."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Get current evaluation dataset
        eval_dataset = self.trainer.eval_dataset
        languages = eval_dataset['language']
        unique_languages = set(languages)
        
        metrics = {}
        
        # Per-language metrics
        for lang in unique_languages:
            mask = [l == lang for l in languages]
            lang_preds = predictions[mask]
            lang_labels = labels[mask]
            if len(lang_preds) > 0:
                lang_f1 = self.f1_metric.compute(
                    predictions=lang_preds,
                    references=lang_labels,
                    average='weighted'
                )['f1']
                metrics[f'f1_{lang}'] = lang_f1
        
        # Overall F1
        overall_f1 = self.f1_metric.compute(
            predictions=predictions,
            references=labels,
            average='weighted'
        )['f1']
        metrics['f1'] = overall_f1
        
        return metrics
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        class_counts = train_df['labels'].value_counts()
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(class_weights.values, dtype=torch.float)
    
    def train(self):
        """Main training function."""
        logger.info("Starting training pipeline...")
        
        # Prepare data
        dataset_dict = self.prepare_data()
        
        # Initialize tokenizer and model
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            trust_remote_code=True
        )
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True
        )
        
        # Calculate class weights
        train_df = dataset_dict['train'].to_pandas()
        class_weights = self.calculate_class_weights(train_df)
        if torch.cuda.is_available():
            class_weights = class_weights.to("cuda")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training'].get('eval_batch_size', 128),
            num_train_epochs=self.config['training']['epochs'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_dir=f"{self.config['training']['output_dir']}/logs",
            logging_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",
            seed=self.config['training'].get('seed', 42),
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        )
        
        # Initialize trainer
        self.trainer = FocalLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=self.compute_metrics,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(tokenized_datasets['test'])
        logger.info(f"Test results: {test_results}")
        
        # Save model
        logger.info(f"Saving model to {self.config['training']['output_dir']}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])
        
        return test_results


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train multilingual classifier")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_name', type=str, help='Model name (overrides config)')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {
                'name': 'intfloat/multilingual-e5-large-instruct',
                'max_length': 64
            },
            'training': {
                'output_dir': './results/multilingual-e5-large-instruct',
                'epochs': 8,
                'batch_size': 16,
                'eval_batch_size': 128,
                'learning_rate': 4.676e-05,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'seed': 42
            },
            'data': {
                'train_path': 'data/datasets/train.csv',
                'test_path': 'data/datasets/test.csv',
                'languages': ["Russian", "English", "French", "German", 
                             "Hebrew", "Arabic", "Chinese"]
            }
        }
    
    # Override config with command line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.num_epochs:
        config['training']['epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create output directory
    Path(config['training']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize and train classifier
    classifier = MultilingualClassifier(config)
    results = classifier.train()
    
    # Save results
    results_path = Path(config['training']['output_dir']) / 'test_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"Training completed. Results saved to {results_path}")


if __name__ == "__main__":
    main()
