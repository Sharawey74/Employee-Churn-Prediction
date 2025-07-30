"""
Data loading and preprocessing module for Employee Turnover Prediction.

This module handles Task 1 (Import Libraries) and data loading functionality.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import DATA_FILE, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN
from utils.helpers import setup_logging, validate_dataframe

class DataLoader:
    """
    Data loader class for handling employee turnover data.
    
    Implements Task 1: Import Libraries and basic data loading functionality.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file. If None, uses default path.
        """
        self.logger = setup_logging()
        self.data_path = data_path or DATA_FILE
        self.data = None
        self.is_loaded = False
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the employee dataset.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is invalid
        """
        try:
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Basic validation
            required_columns = FEATURE_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
            validate_dataframe(self.data, required_columns)
            
            self.is_loaded = True
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'target_distribution': self.data[TARGET_COLUMN].value_counts().to_dict()
        }
        
        return info
    
    def get_basic_statistics(self) -> dict:
        """
        Get basic statistical information about the dataset.
        
        Returns:
            Dictionary with statistical information
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        stats = {
            'numeric_stats': self.data[numeric_columns].describe().to_dict(),
            'categorical_stats': {
                col: self.data[col].value_counts().to_dict() 
                for col in categorical_columns
            },
            'correlation_matrix': self.data[numeric_columns].corr().to_dict()
        }
        
        return stats
    
    def check_data_quality(self) -> dict:
        """
        Perform data quality checks.
        
        Returns:
            Dictionary with data quality metrics
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        quality_report = {
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'data_types': self.data.dtypes.to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns},
            'potential_outliers': {}
        }
        
        # Check for potential outliers in numeric columns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            quality_report['potential_outliers'][col] = outliers
        
        return quality_report
    
    def get_feature_target_split(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get all columns except target
        feature_cols = [col for col in self.data.columns if col != TARGET_COLUMN]
        
        X = self.data[feature_cols].copy()
        y = self.data[TARGET_COLUMN].copy()
        
        return X, y
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to {output_path}")


def create_sample_data(n_samples: int = 1000, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create sample employee data for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the generated data
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(42)
    
    # Generate synthetic employee data
    data = {
        'satisfaction_level': np.random.uniform(0.1, 1.0, n_samples),
        'last_evaluation': np.random.uniform(0.3, 1.0, n_samples),
        'number_project': np.random.randint(2, 8, n_samples),
        'average_montly_hours': np.random.randint(120, 320, n_samples),
        'time_spend_company': np.random.randint(1, 11, n_samples),
        'Work_accident': np.random.binomial(1, 0.2, n_samples),
        'promotion_last_5years': np.random.binomial(1, 0.1, n_samples),
        'department': np.random.choice(
            ['sales', 'technical', 'support', 'IT', 'hr', 'accounting', 
             'marketing', 'product_mng', 'RandD', 'management'], 
            n_samples
        ),
        'salary': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable based on realistic patterns
    quit_probability = (
        0.3 * (1 - df['satisfaction_level']) +  # Lower satisfaction = higher quit probability
        0.2 * (df['average_montly_hours'] > 250).astype(int) +  # Overwork increases quit probability
        0.2 * (df['time_spend_company'] > 6).astype(int) +  # Long tenure might increase quit probability
        0.1 * (df['salary'] == 'low').astype(int) +  # Low salary increases quit probability
        0.1 * np.random.uniform(0, 1, n_samples)  # Random component
    )
    
    df['quit'] = (quit_probability > 0.5).astype(int)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Sample data saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    try:
        data = loader.load_data()
        print("Data loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        info = loader.get_data_info()
        print(f"Target distribution: {info['target_distribution']}")
        
    except FileNotFoundError:
        print("Data file not found. Creating sample data...")
        sample_data = create_sample_data(1000, "data/raw/employee_data.csv")
        print("Sample data created. Try running again.")