"""
Helper Utilities for Customer Churn Prediction Project
Contains general utility functions used across the project
"""

import pandas as pd
import numpy as np
import logging
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
    
    logger.info(f"Logging configured with level: {log_level}")


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_object(obj: Any, filepath: Union[str, Path], method: str = 'joblib') -> None:
    """
    Save Python object to disk
    
    Args:
        obj: Object to save
        filepath: Path to save file
        method: Serialization method ('joblib', 'pickle', 'json')
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    if method == 'joblib':
        joblib.dump(obj, filepath)
    elif method == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif method == 'json':
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    else:
        raise ValueError(f"Unknown serialization method: {method}")
    
    logger.info(f"Saved object to {filepath} using {method}")


def load_object(filepath: Union[str, Path], method: str = 'joblib') -> Any:
    """
    Load Python object from disk
    
    Args:
        filepath: Path to file
        method: Serialization method ('joblib', 'pickle', 'json')
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if method == 'joblib':
        obj = joblib.load(filepath)
    elif method == 'pickle':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif method == 'json':
        with open(filepath, 'r') as f:
            obj = json.load(f)
    else:
        raise ValueError(f"Unknown serialization method: {method}")
    
    logger.info(f"Loaded object from {filepath} using {method}")
    return obj


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate memory usage of DataFrame in MB
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by optimizing data types
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print progress
        
    Returns:
        Optimized DataFrame
    """
    start_mem = memory_usage_mb(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = memory_usage_mb(df)
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        data: Series to analyze
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method
    
    Args:
        data: Series to analyze
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calculate percentage of missing values for each column
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Series with missing value percentages
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100
    return missing_percent.sort_values(ascending=False)


def get_categorical_summary(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get summary statistics for categorical columns
    
    Args:
        df: DataFrame to analyze
        categorical_columns: List of categorical column names
        
    Returns:
        Dictionary with categorical summaries
    """
    summary = {}
    
    for col in categorical_columns:
        if col in df.columns:
            summary[col] = {
                'unique_count': df[col].nunique(),
                'unique_values': df[col].unique().tolist(),
                'value_counts': df[col].value_counts().to_dict(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
    
    return summary


def get_numerical_summary(df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get summary statistics for numerical columns
    
    Args:
        df: DataFrame to analyze
        numerical_columns: List of numerical column names
        
    Returns:
        Dictionary with numerical summaries
    """
    summary = {}
    
    for col in numerical_columns:
        if col in df.columns:
            series = df[col]
            summary[col] = {
                'count': series.count(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'missing_count': series.isnull().sum(),
                'missing_percentage': (series.isnull().sum() / len(df)) * 100,
                'outliers_iqr': detect_outliers_iqr(series).sum(),
                'outliers_zscore': detect_outliers_zscore(series).sum()
            }
    
    return summary


def create_feature_summary_report(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Create comprehensive feature summary report
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        
    Returns:
        Dictionary with complete feature summary
    """
    logger.info("Creating feature summary report")
    
    numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    
    # Remove target column from feature lists
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    report = {
        'dataset_info': {
            'shape': df.shape,
            'memory_usage_mb': memory_usage_mb(df),
            'total_missing_values': df.isnull().sum().sum(),
            'duplicated_rows': df.duplicated().sum()
        },
        'feature_types': {
            'numerical_features': numerical_cols,
            'categorical_features': categorical_cols,
            'numerical_count': len(numerical_cols),
            'categorical_count': len(categorical_cols)
        },
        'missing_values': calculate_missing_percentage(df).to_dict(),
        'numerical_summary': get_numerical_summary(df, numerical_cols),
        'categorical_summary': get_categorical_summary(df, categorical_cols)
    }
    
    # Target analysis if provided
    if target_column and target_column in df.columns:
        report['target_analysis'] = {
            'column_name': target_column,
            'data_type': str(df[target_column].dtype),
            'unique_values': df[target_column].unique().tolist(),
            'value_counts': df[target_column].value_counts().to_dict(),
            'missing_values': df[target_column].isnull().sum()
        }
        
        # If binary classification
        if df[target_column].nunique() == 2:
            positive_class = df[target_column].value_counts().index[1]
            negative_class = df[target_column].value_counts().index[0]
            
            positive_count = (df[target_column] == positive_class).sum()
            total_count = len(df)
            
            report['target_analysis']['classification_type'] = 'binary'
            report['target_analysis']['positive_class'] = positive_class
            report['target_analysis']['negative_class'] = negative_class
            report['target_analysis']['positive_rate'] = positive_count / total_count
            report['target_analysis']['class_balance_ratio'] = positive_count / (total_count - positive_count)
    
    logger.info("Feature summary report created successfully")
    return report


def print_feature_summary(report: Dict[str, Any]) -> None:
    """
    Print feature summary report in readable format
    
    Args:
        report: Feature summary report dictionary
    """
    print("="*80)
    print("FEATURE SUMMARY REPORT")
    print("="*80)
    
    # Dataset info
    info = report['dataset_info']
    print(f"\nDATASET INFORMATION:")
    print(f"  Shape: {info['shape']}")
    print(f"  Memory Usage: {info['memory_usage_mb']:.2f} MB")
    print(f"  Missing Values: {info['total_missing_values']}")
    print(f"  Duplicated Rows: {info['duplicated_rows']}")
    
    # Feature types
    types = report['feature_types']
    print(f"\nFEATURE TYPES:")
    print(f"  Numerical Features: {types['numerical_count']}")
    print(f"  Categorical Features: {types['categorical_count']}")
    
    # Missing values
    if report['missing_values']:
        print(f"\nTOP MISSING VALUES:")
        missing_sorted = sorted(report['missing_values'].items(), key=lambda x: x[1], reverse=True)
        for col, pct in missing_sorted[:5]:
            if pct > 0:
                print(f"  {col}: {pct:.2f}%")
    
    # Target analysis
    if 'target_analysis' in report:
        target = report['target_analysis']
        print(f"\nTARGET ANALYSIS:")
        print(f"  Column: {target['column_name']}")
        print(f"  Type: {target['data_type']}")
        print(f"  Unique Values: {target['unique_values']}")
        
        if 'positive_rate' in target:
            print(f"  Positive Rate: {target['positive_rate']:.3f}")
            print(f"  Class Balance Ratio: {target['class_balance_ratio']:.3f}")
    
    print("="*80)


def calculate_correlation_with_target(df: pd.DataFrame, target_column: str, 
                                    method: str = 'pearson') -> pd.Series:
    """
    Calculate correlation of all numerical features with target
    
    Args:
        df: DataFrame
        target_column: Name of target column
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Series with correlations sorted by absolute value
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != target_column]
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # For binary target, convert to numeric if needed
    if df[target_column].dtype == 'object':
        target_encoded = pd.get_dummies(df[target_column], drop_first=True).iloc[:, 0]
    else:
        target_encoded = df[target_column]
    
    correlations = df[numerical_cols].corrwith(target_encoded, method=method)
    return correlations.abs().sort_values(ascending=False)


def timer(func):
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    
    return wrapper


def validate_data_quality(df: pd.DataFrame, 
                         max_missing_percentage: float = 50.0,
                         min_unique_values: int = 2) -> Dict[str, List[str]]:
    """
    Validate data quality and identify problematic columns
    
    Args:
        df: DataFrame to validate
        max_missing_percentage: Maximum allowed missing percentage
        min_unique_values: Minimum required unique values
        
    Returns:
        Dictionary with lists of problematic columns
    """
    issues = {
        'high_missing': [],
        'low_variance': [],
        'all_missing': [],
        'single_value': []
    }
    
    for col in df.columns:
        # Check missing values
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct == 100:
            issues['all_missing'].append(col)
        elif missing_pct > max_missing_percentage:
            issues['high_missing'].append(col)
        
        # Check unique values
        unique_count = df[col].nunique()
        if unique_count == 1:
            issues['single_value'].append(col)
        elif unique_count < min_unique_values:
            issues['low_variance'].append(col)
    
    return issues


if __name__ == "__main__":
    """
    Example usage of helper functions
    """
    print("Helper Utilities Module")
    print("This module provides utility functions for the churn prediction project")
    
    # Example data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.randint(0, 10, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add some missing values
    sample_data.loc[sample_data.sample(50).index, 'feature1'] = np.nan
    
    print(f"\nExample dataset shape: {sample_data.shape}")
    print(f"Memory usage: {memory_usage_mb(sample_data):.2f} MB")
    
    # Create feature summary
    summary = create_feature_summary_report(sample_data, 'target')
    print_feature_summary(summary)
