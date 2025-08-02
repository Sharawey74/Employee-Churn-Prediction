"""
Data Preprocessing Utilities for Customer Churn Prediction
Advanced preprocessing functions and pipeline utilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer that maintains DataFrame structure and column names
    """
    
    def __init__(self, strategy: str = 'mean', fill_value=None):
        """
        Initialize DataFrameImputer
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            fill_value: Value to use when strategy is 'constant'
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = None
        self.columns = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputer"""
        self.columns = X.columns
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imputer.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=self.columns, index=X.index)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that maintains DataFrame structure and column names
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize DataFrameScaler
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'power')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.columns = None
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'power':
            self.scaler = PowerTransformer()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the scaler"""
        self.columns = X.columns
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns, index=X.index)


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Outlier detection and treatment transformer
    """
    
    def __init__(self, method: str = 'iqr', factor: float = 1.5, treatment: str = 'clip'):
        """
        Initialize OutlierDetector
        
        Args:
            method: Detection method ('iqr', 'zscore')
            factor: Factor for outlier detection (IQR multiplier or Z-score threshold)
            treatment: Treatment method ('clip', 'remove', 'transform')
        """
        self.method = method
        self.factor = factor
        self.treatment = treatment
        self.bounds = {}
        self.columns = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the outlier detector"""
        self.columns = X.columns
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower_bound = mean - self.factor * std
                upper_bound = mean + self.factor * std
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.bounds[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        X_transformed = X.copy()
        
        for col, (lower_bound, upper_bound) in self.bounds.items():
            if col in X_transformed.columns:
                if self.treatment == 'clip':
                    X_transformed[col] = X_transformed[col].clip(lower_bound, upper_bound)
                elif self.treatment == 'remove':
                    # Mark outliers for removal
                    outlier_mask = (X_transformed[col] < lower_bound) | (X_transformed[col] > upper_bound)
                    X_transformed = X_transformed[~outlier_mask]
                elif self.treatment == 'transform':
                    # Log transformation for outliers
                    outlier_mask = (X_transformed[col] < lower_bound) | (X_transformed[col] > upper_bound)
                    X_transformed.loc[outlier_mask, col] = np.log1p(X_transformed.loc[outlier_mask, col])
        
        return X_transformed


def detect_and_handle_outliers(df: pd.DataFrame, 
                              numerical_columns: List[str],
                              method: str = 'iqr',
                              factor: float = 1.5,
                              treatment: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in numerical columns
    
    Args:
        df: Input DataFrame
        numerical_columns: List of numerical column names
        method: Detection method ('iqr', 'zscore')
        factor: Factor for outlier detection
        treatment: Treatment method ('clip', 'remove', 'transform')
        
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Handling outliers using {method} method with {treatment} treatment")
    
    df_processed = df.copy()
    outlier_counts = {}
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - factor * std
            upper_bound = mean + factor * std
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Count outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_counts[col] = outliers.sum()
        
        # Apply treatment
        if treatment == 'clip':
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        elif treatment == 'remove':
            df_processed = df_processed[~outliers]
        elif treatment == 'transform':
            # Apply log transformation to outliers
            outlier_mask = outliers & (df[col] > 0)  # Only positive values
            if outlier_mask.any():
                df_processed.loc[outlier_mask, col] = np.log1p(df_processed.loc[outlier_mask, col])
    
    # Log results
    total_outliers = sum(outlier_counts.values())
    logger.info(f"Detected {total_outliers} outliers across {len(outlier_counts)} columns")
    
    for col, count in outlier_counts.items():
        if count > 0:
            logger.info(f"  {col}: {count} outliers")
    
    return df_processed


def handle_missing_values(df: pd.DataFrame,
                         numerical_strategy: str = 'median',
                         categorical_strategy: str = 'most_frequent',
                         threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values with different strategies for different column types
    
    Args:
        df: Input DataFrame
        numerical_strategy: Strategy for numerical columns
        categorical_strategy: Strategy for categorical columns
        threshold: Drop columns with missing percentage above this threshold
        
    Returns:
        Tuple of (processed DataFrame, missing value report)
    """
    logger.info("Handling missing values")
    
    df_processed = df.copy()
    missing_report = {}
    
    # Calculate missing percentages
    missing_percentages = (df.isnull().sum() / len(df))
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    if columns_to_drop:
        logger.info(f"Dropping columns with >{threshold*100}% missing: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)
        missing_report['dropped_columns'] = columns_to_drop
    
    # Separate numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Handle numerical missing values
    if numerical_cols:
        numerical_imputer = SimpleImputer(strategy=numerical_strategy)
        df_processed[numerical_cols] = numerical_imputer.fit_transform(df_processed[numerical_cols])
        missing_report['numerical_strategy'] = numerical_strategy
        missing_report['numerical_columns_processed'] = numerical_cols
    
    # Handle categorical missing values
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
        missing_report['categorical_strategy'] = categorical_strategy
        missing_report['categorical_columns_processed'] = categorical_cols
    
    # Final missing value check
    remaining_missing = df_processed.isnull().sum().sum()
    missing_report['remaining_missing_values'] = remaining_missing
    
    logger.info(f"Missing value handling completed. Remaining missing values: {remaining_missing}")
    
    return df_processed, missing_report


def advanced_missing_value_imputation(df: pd.DataFrame,
                                     numerical_columns: List[str],
                                     categorical_columns: List[str],
                                     method: str = 'knn',
                                     n_neighbors: int = 5) -> pd.DataFrame:
    """
    Advanced missing value imputation using KNN or iterative methods
    
    Args:
        df: Input DataFrame
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        method: Imputation method ('knn', 'iterative')
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Performing advanced imputation using {method} method")
    
    df_processed = df.copy()
    
    if method == 'knn':
        # KNN imputation for numerical columns
        if numerical_columns:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df_processed[numerical_columns] = knn_imputer.fit_transform(df_processed[numerical_columns])
        
        # Simple imputation for categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in df_processed.columns:
                    mode_value = df_processed[col].mode()
                    if not mode_value.empty:
                        df_processed[col].fillna(mode_value.iloc[0], inplace=True)
    
    elif method == 'iterative':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        # Iterative imputation for numerical columns
        if numerical_columns:
            iterative_imputer = IterativeImputer(random_state=42)
            df_processed[numerical_columns] = iterative_imputer.fit_transform(df_processed[numerical_columns])
        
        # Simple imputation for categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in df_processed.columns:
                    mode_value = df_processed[col].mode()
                    if not mode_value.empty:
                        df_processed[col].fillna(mode_value.iloc[0], inplace=True)
    
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    logger.info("Advanced imputation completed")
    return df_processed


def create_feature_interactions(df: pd.DataFrame,
                               numerical_columns: List[str],
                               max_interactions: int = 10) -> pd.DataFrame:
    """
    Create interaction features between numerical columns
    
    Args:
        df: Input DataFrame
        numerical_columns: List of numerical column names
        max_interactions: Maximum number of interactions to create
        
    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating feature interactions")
    
    df_with_interactions = df.copy()
    interactions_created = 0
    
    for i, col1 in enumerate(numerical_columns):
        if interactions_created >= max_interactions:
            break
            
        for col2 in numerical_columns[i+1:]:
            if interactions_created >= max_interactions:
                break
            
            if col1 in df.columns and col2 in df.columns:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df_with_interactions[interaction_name] = df[col1] * df[col2]
                interactions_created += 1
                
                # Division interaction (avoid division by zero)
                if interactions_created < max_interactions:
                    ratio_name = f"{col1}_div_{col2}"
                    df_with_interactions[ratio_name] = df[col1] / (df[col2] + 1e-8)
                    interactions_created += 1
    
    logger.info(f"Created {interactions_created} interaction features")
    return df_with_interactions


def create_polynomial_features(df: pd.DataFrame,
                             numerical_columns: List[str],
                             degree: int = 2,
                             interaction_only: bool = False) -> pd.DataFrame:
    """
    Create polynomial features
    
    Args:
        df: Input DataFrame
        numerical_columns: List of numerical column names
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        DataFrame with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    logger.info(f"Creating polynomial features with degree {degree}")
    
    # Select only the specified numerical columns
    df_numeric = df[numerical_columns].copy()
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df_numeric)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(numerical_columns)
    
    # Create DataFrame with polynomial features
    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Combine with original DataFrame (excluding original numerical columns to avoid duplication)
    df_other = df.drop(columns=numerical_columns)
    df_result = pd.concat([df_other, df_poly], axis=1)
    
    logger.info(f"Created {len(feature_names)} polynomial features")
    return df_result


def feature_binning(df: pd.DataFrame,
                   column: str,
                   n_bins: int = 5,
                   strategy: str = 'quantile',
                   labels: List[str] = None) -> pd.Series:
    """
    Bin numerical features into categorical bins
    
    Args:
        df: Input DataFrame
        column: Column name to bin
        n_bins: Number of bins
        strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
        labels: Custom labels for bins
        
    Returns:
        Series with binned values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if strategy == 'uniform':
        binned = pd.cut(df[column], bins=n_bins, labels=labels)
    elif strategy == 'quantile':
        binned = pd.qcut(df[column], q=n_bins, labels=labels, duplicates='drop')
    elif strategy == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
        cluster_labels = kmeans.fit_predict(df[column].values.reshape(-1, 1))
        binned = pd.Series(cluster_labels, index=df.index, dtype='category')
        if labels:
            binned = binned.cat.rename_categories(labels[:len(binned.cat.categories)])
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    
    return binned


def create_preprocessing_pipeline(numerical_columns: List[str],
                                categorical_columns: List[str],
                                numerical_strategy: str = 'standard',
                                categorical_strategy: str = 'onehot',
                                handle_outliers: bool = True) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical and categorical features
    
    Args:
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        numerical_strategy: Strategy for numerical preprocessing
        categorical_strategy: Strategy for categorical preprocessing
        handle_outliers: Whether to handle outliers
        
    Returns:
        ColumnTransformer pipeline
    """
    logger.info("Creating preprocessing pipeline")
    
    # Numerical preprocessing pipeline
    numerical_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    
    if handle_outliers:
        numerical_steps.append(('outlier_detector', OutlierDetector()))
    
    if numerical_strategy == 'standard':
        numerical_steps.append(('scaler', StandardScaler()))
    elif numerical_strategy == 'minmax':
        numerical_steps.append(('scaler', MinMaxScaler()))
    elif numerical_strategy == 'robust':
        numerical_steps.append(('scaler', RobustScaler()))
    
    numerical_pipeline = Pipeline(numerical_steps)
    
    # Categorical preprocessing pipeline
    categorical_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ]
    
    if categorical_strategy == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        categorical_steps.append(('encoder', OneHotEncoder(drop='first', sparse=False)))
    elif categorical_strategy == 'label':
        from sklearn.preprocessing import LabelEncoder
        categorical_steps.append(('encoder', LabelEncoder()))
    
    categorical_pipeline = Pipeline(categorical_steps)
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])
    
    logger.info("Preprocessing pipeline created")
    return preprocessor


def apply_feature_selection(X: pd.DataFrame, y: pd.Series,
                          method: str = 'mutual_info',
                          k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply feature selection to reduce dimensionality
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Selection method ('mutual_info', 'chi2', 'f_classif')
        k: Number of features to select
        
    Returns:
        Tuple of (selected features DataFrame, selected feature names)
    """
    logger.info(f"Applying feature selection using {method} method")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'chi2':
        selector = SelectKBest(score_func=chi2, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    return X_selected_df, selected_features


if __name__ == "__main__":
    """
    Example usage of preprocessing utilities
    """
    print("Preprocessing Utilities Module")
    print("This module provides advanced preprocessing functions")
    
    # Example data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),  # Skewed data
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'feature4': np.random.randint(0, 100, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add missing values
    sample_data.loc[sample_data.sample(50).index, 'feature1'] = np.nan
    sample_data.loc[sample_data.sample(30).index, 'feature3'] = np.nan
    
    # Add outliers
    sample_data.loc[sample_data.sample(20).index, 'feature2'] = np.random.normal(50, 10, 20)
    
    print(f"\nExample dataset shape: {sample_data.shape}")
    print(f"Missing values per column:")
    print(sample_data.isnull().sum())
    
    # Handle missing values
    processed_data, missing_report = handle_missing_values(sample_data)
    print(f"\nAfter handling missing values: {processed_data.isnull().sum().sum()} remaining")
    
    # Handle outliers
    numerical_cols = ['feature1', 'feature2', 'feature4']
    processed_data = detect_and_handle_outliers(processed_data, numerical_cols)
    print("Outliers handled using IQR method")
