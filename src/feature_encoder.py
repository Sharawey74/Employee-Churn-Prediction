"""
Feature encoding module for Employee Turnover Prediction.

This module handles Task 3: Encode Categorical Features.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import CATEGORICAL_COLUMNS
from utils.helpers import setup_logging, validate_dataframe

class FeatureEncoder:
    """
    Feature encoder class for handling categorical variables.
    
    Implements Task 3: Encode Categorical Features.
    The dataset contains two categorical variables: Department and Salary.
    Creates dummy encoded variables for both categorical variables.
    """
    
    def __init__(self):
        """Initialize the feature encoder."""
        self.logger = setup_logging()
        self.encoders = {}
        self.encoded_columns = {}
        self.original_columns = None
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.
        
        Args:
            X: Input DataFrame
            categorical_columns: List of categorical columns to encode
            
        Returns:
            Transformed DataFrame with encoded categorical variables
        """
        if categorical_columns is None:
            categorical_columns = CATEGORICAL_COLUMNS
        
        self.logger.info(f"Encoding categorical columns: {categorical_columns}")
        
        # Validate input
        validate_dataframe(X, categorical_columns)
        
        # Store original columns
        self.original_columns = X.columns.tolist()
        
        # Create a copy to avoid modifying original data
        X_encoded = X.copy()
        
        # Encode each categorical column
        for col in categorical_columns:
            if col in X_encoded.columns:
                X_encoded = self._encode_column(X_encoded, col, fit=True)
        
        self.is_fitted = True
        self.logger.info(f"Encoding complete. Original shape: {X.shape}, New shape: {X_encoded.shape}")
        
        return X_encoded
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If encoder is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit_transform() first.")
        
        X_encoded = X.copy()
        
        for col in self.encoders.keys():
            if col in X_encoded.columns:
                X_encoded = self._encode_column(X_encoded, col, fit=False)
        
        return X_encoded
    
    def _encode_column(self, X: pd.DataFrame, column: str, fit: bool = True) -> pd.DataFrame:
        """
        Encode a single categorical column using one-hot encoding.
        
        Args:
            X: Input DataFrame
            column: Column name to encode
            fit: Whether to fit the encoder
            
        Returns:
            DataFrame with encoded column
        """
        if fit:
            # Get unique values and create dummy variables
            unique_values = sorted(X[column].unique())
            self.encoded_columns[column] = [f"{column}_{val}" for val in unique_values]
            
            # Create dummy variables
            dummies = pd.get_dummies(X[column], prefix=column, dtype=int)
            
            # Store the column mapping for transform
            self.encoders[column] = {
                'unique_values': unique_values,
                'dummy_columns': dummies.columns.tolist()
            }
            
            self.logger.info(f"Created dummy variables for {column}: {dummies.columns.tolist()}")
            
        else:
            # Transform using existing encoder
            if column not in self.encoders:
                raise ValueError(f"Column {column} was not seen during fitting")
            
            # Create dummy variables with same structure as training
            dummies = pd.get_dummies(X[column], prefix=column, dtype=int)
            
            # Ensure all training columns are present
            training_columns = self.encoders[column]['dummy_columns']
            for col_name in training_columns:
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Remove any extra columns not seen during training
            dummies = dummies[training_columns]
        
        # Remove original column and add dummy variables
        X_result = X.drop(columns=[column])
        X_result = pd.concat([X_result, dummies], axis=1)
        
        return X_result
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features after encoding.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit_transform() first.")
        
        feature_names = []
        
        # Add non-categorical columns
        for col in self.original_columns:
            if col not in self.encoders:
                feature_names.append(col)
        
        # Add encoded categorical columns
        for col, encoded_cols in self.encoded_columns.items():
            feature_names.extend(encoded_cols)
        
        return feature_names
    
    def get_encoding_info(self) -> Dict:
        """
        Get information about the encoding process.
        
        Returns:
            Dictionary with encoding information
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        info = {
            "status": "Fitted",
            "original_columns": self.original_columns,
            "encoded_columns": self.encoded_columns,
            "encoding_mappings": {}
        }
        
        for col, encoder_info in self.encoders.items():
            info["encoding_mappings"][col] = {
                "original_values": encoder_info['unique_values'],
                "dummy_columns": encoder_info['dummy_columns']
            }
        
        return info
    
    def reverse_transform_column(self, X: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Reverse the encoding for a specific categorical column.
        
        Args:
            X: DataFrame with encoded features
            column: Original column name to reverse
            
        Returns:
            DataFrame with original categorical column
        """
        if column not in self.encoders:
            raise ValueError(f"Column {column} was not encoded")
        
        dummy_columns = self.encoders[column]['dummy_columns']
        unique_values = self.encoders[column]['unique_values']
        
        # Check if dummy columns exist in the DataFrame
        missing_cols = [col for col in dummy_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing dummy columns: {missing_cols}")
        
        # Get the dummy variables
        dummy_data = X[dummy_columns]
        
        # Convert back to original categorical values
        original_values = []
        for idx, row in dummy_data.iterrows():
            # Find which dummy variable is 1
            active_dummies = row[row == 1].index.tolist()
            if len(active_dummies) == 1:
                # Extract original value from dummy column name
                dummy_col = active_dummies[0]
                original_val = dummy_col.replace(f"{column}_", "")
                original_values.append(original_val)
            elif len(active_dummies) == 0:
                # No dummy is 1 - this shouldn't happen with proper encoding
                original_values.append(None)
            else:
                # Multiple dummies are 1 - this shouldn't happen with one-hot encoding
                self.logger.warning(f"Multiple active dummies for row {idx}: {active_dummies}")
                original_values.append(active_dummies[0].replace(f"{column}_", ""))
        
        # Create result DataFrame
        X_result = X.drop(columns=dummy_columns).copy()
        X_result[column] = original_values
        
        return X_result


def encode_categorical_features(X: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, FeatureEncoder]:
    """
    Convenience function to encode categorical features.
    
    Args:
        X: Input DataFrame
        categorical_columns: List of categorical columns to encode
        
    Returns:
        Tuple of (encoded DataFrame, fitted encoder)
    """
    encoder = FeatureEncoder()
    X_encoded = encoder.fit_transform(X, categorical_columns)
    return X_encoded, encoder


if __name__ == "__main__":
    # Example usage
    print("Feature Encoder Example")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'satisfaction_level': [0.38, 0.80, 0.11, 0.72],
        'department': ['sales', 'IT', 'hr', 'sales'],
        'salary': ['low', 'medium', 'high', 'low'],
        'quit': [1, 0, 1, 0]
    })
    
    print("Original data:")
    print(sample_data)
    print(f"Shape: {sample_data.shape}")
    
    # Encode categorical features
    encoder = FeatureEncoder()
    X = sample_data.drop('quit', axis=1)
    X_encoded = encoder.fit_transform(X)
    
    print("\nEncoded data:")
    print(X_encoded)
    print(f"Shape: {X_encoded.shape}")
    
    print("\nEncoding info:")
    info = encoder.get_encoding_info()
    for col, mapping in info["encoding_mappings"].items():
        print(f"{col}: {mapping['original_values']} -> {mapping['dummy_columns']}")
    
    print(f"\nFeature names: {encoder.get_feature_names()}")