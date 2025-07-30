"""
Helper functions for the Employee Turnover Prediction project.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('employee_turnover.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns are present, False otherwise
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def save_figure(fig: plt.Figure, filename: str, dpi: int = 100, 
                bbox_inches: str = 'tight') -> None:
    """
    Save a matplotlib figure with consistent formatting.
    
    Args:
        fig: Matplotlib figure to save
        filename: Output filename
        dpi: Dots per inch for output
        bbox_inches: Bounding box setting
    """
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

def calculate_class_balance(y: pd.Series) -> dict:
    """
    Calculate class balance statistics.
    
    Args:
        y: Target variable series
        
    Returns:
        Dictionary with class counts and percentages
    """
    value_counts = y.value_counts()
    total = len(y)
    
    return {
        'counts': value_counts.to_dict(),
        'percentages': (value_counts / total * 100).to_dict(),
        'balance_ratio': value_counts.min() / value_counts.max()
    }

def print_model_summary(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Print a summary of model performance.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features  
        y_val: Validation targets
    """
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"Overfitting: {train_score - val_score:.4f}")
    print("-" * 40)

def create_directory_structure() -> None:
    """Create the necessary directory structure for the project."""
    directories = [
        'data/raw',
        'data/processed', 
        'outputs',
        'plots',
        'models',
        'docs',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_feature_importance_df(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance as a sorted DataFrame.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df