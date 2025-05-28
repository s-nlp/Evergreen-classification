"""
Data processing utilities for multilingual temporal classification.
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

logger = logging.getLogger(__name__)


def load_and_validate_data(file_path: str, required_columns: List[str]) -> pd.DataFrame:
    """
    Load CSV data and validate required columns exist.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(file_path)
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with missing values in required columns
    initial_rows = len(df)
    df = df.dropna(subset=required_columns)
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")
    
    return df


def analyze_class_distribution(df: pd.DataFrame, label_column: str = 'label') -> Dict:
    """
    Analyze class distribution in the dataset.
    
    Args:
        df: DataFrame with labels
        label_column: Name of label column
        
    Returns:
        Dictionary with class distribution statistics
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    
    class_counts = df[label_column].value_counts()
    total_samples = len(df)
    
    distribution = {
        'class_counts': class_counts.to_dict(),
        'class_percentages': (class_counts / total_samples * 100).to_dict(),
        'total_samples': total_samples,
        'is_balanced': max(class_counts) / min(class_counts) < 1.5
    }
    
    return distribution


def analyze_language_distribution(df: pd.DataFrame, languages: List[str]) -> Dict:
    """
    Analyze text length distribution across languages.
    
    Args:
        df: DataFrame with language columns
        languages: List of language column names
        
    Returns:
        Dictionary with language statistics
    """
    stats = {}
    
    for lang in languages:
        if lang in df.columns:
            # Calculate text lengths
            lengths = df[lang].str.len()
            
            stats[lang] = {
                'mean_length': lengths.mean(),
                'std_length': lengths.std(),
                'min_length': lengths.min(),
                'max_length': lengths.max(),
                'missing_values': df[lang].isna().sum()
            }
    
    return stats


def create_train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    stratify_column: str = 'label',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split with stratification.
    
    Args:
        df: Input DataFrame
        val_size: Validation set size (fraction)
        stratify_column: Column to stratify by
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df)
    """
    stratify = df[stratify_column] if stratify_column in df.columns else None
    
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=stratify,
        random_state=random_state
    )
    
    logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    
    return train_df, val_df


def augment_with_back_translation(
    df: pd.DataFrame,
    source_lang: str,
    target_langs: List[str],
    label_column: str = 'label'
) -> pd.DataFrame:
    """
    Simple data augmentation by using translations as additional samples.
    
    Args:
        df: Input DataFrame
        source_lang: Source language column
        target_langs: List of target language columns to use
        label_column: Label column name
        
    Returns:
        Augmented DataFrame
    """
    augmented_rows = []
    
    for _, row in df.iterrows():
        # Keep original
        augmented_rows.append(row.to_dict())
        
        # Add translations as new samples
        for target_lang in target_langs:
            if target_lang in df.columns and pd.notna(row[target_lang]):
                new_row = row.to_dict()
                # Swap source and target language texts
                new_row[source_lang] = row[target_lang]
                augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    logger.info(f"Augmented dataset size: {len(augmented_df)} (original: {len(df)})")
    
    return augmented_df


def prepare_few_shot_examples(
    df: pd.DataFrame,
    n_examples: int = 5,
    label_column: str = 'label',
    text_column: str = 'English'
) -> str:
    """
    Prepare few-shot examples for prompting.
    
    Args:
        df: DataFrame with examples
        n_examples: Number of examples per class
        label_column: Label column name
        text_column: Text column to use
        
    Returns:
        Formatted few-shot examples string
    """
    examples = []
    
    # Get balanced examples from each class
    for label in df[label_column].unique():
        class_df = df[df[label_column] == label]
        sampled = class_df.sample(n=min(n_examples, len(class_df)))
        
        for _, row in sampled.iterrows():
            class_name = "Mutable" if label == 1 else "Immutable"
            examples.append(f"Question: {row[text_column]} Classification: {class_name}")
    
    return "Examples\n\n" + "\n".join(examples)


def calculate_metrics_per_language(
    df: pd.DataFrame,
    prediction_columns: Dict[str, str],
    label_column: str = 'label'
) -> pd.DataFrame:
    """
    Calculate accuracy metrics for each language.
    
    Args:
        df: DataFrame with predictions
        prediction_columns: Dict mapping language to prediction column
        label_column: True label column
        
    Returns:
        DataFrame with metrics per language
    """
    metrics = []
    
    for language, pred_col in prediction_columns.items():
        if pred_col in df.columns:
            # Convert predictions to binary
            predictions = df[pred_col].map({'Mutable': 1, 'Immutable': 0})
            labels = df[label_column]
            
            # Calculate metrics
            mask = predictions.notna()
            if mask.sum() > 0:
                accuracy = (predictions[mask] == labels[mask]).mean()
                
                # Calculate per-class accuracy
                mutable_mask = labels == 1
                immutable_mask = labels == 0
                
                mutable_acc = (predictions[mask & mutable_mask] == 1).mean() if (mask & mutable_mask).sum() > 0 else 0
                immutable_acc = (predictions[mask & immutable_mask] == 0).mean() if (mask & immutable_mask).sum() > 0 else 0
                
                metrics.append({
                    'language': language,
                    'accuracy': accuracy,
                    'mutable_accuracy': mutable_acc,
                    'immutable_accuracy': immutable_acc,
                    'n_samples': mask.sum()
                })
    
    return pd.DataFrame(metrics)


def export_results_to_latex(metrics_df: pd.DataFrame, output_path: str):
    """
    Export metrics DataFrame to LaTeX table format.
    
    Args:
        metrics_df: DataFrame with metrics
        output_path: Path to save LaTeX file
    """
    latex_table = metrics_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Multilingual Classification Results",
        label="tab:results"
    )
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    logger.info(f"Exported LaTeX table to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Data processing utilities")
    parser.add_argument('--analyze', type=str, help='Analyze dataset')
    args = parser.parse_args()
    
    if args.analyze:
        # Load and analyze dataset
        languages = ["Russian", "English", "French", "German", "Hebrew", "Arabic", "Chinese"]
        df = load_and_validate_data(args.analyze, ['label'] + languages)
        
        # Analyze class distribution
        class_dist = analyze_class_distribution(df)
        print("\nClass Distribution:")
        for k, v in class_dist.items():
            print(f"  {k}: {v}")
        
        # Analyze language distribution
        lang_stats = analyze_language_distribution(df, languages)
        print("\nLanguage Statistics:")
        for lang, stats in lang_stats.items():
            print(f"\n  {lang}:")
            for k, v in stats.items():
                print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
