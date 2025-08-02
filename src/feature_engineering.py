"""
Feature Engineering Module for Customer Churn Prediction
Handles feature transformation, encoding, scaling, and creation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import joblib

from .config import FEATURE_CONFIG, IMBALANCE_CONFIG, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive Feature Engineering for Customer Churn Prediction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FeatureEngineer with configuration
        
        Args:
            config: Configuration dictionary, defaults to FEATURE_CONFIG
        """
        self.config = config or FEATURE_CONFIG
        self.preprocessor = None
        self.feature_names = None
        self.scaler = None
        self.encoder = None
        
        logger.info("FeatureEngineer initialized")
    
    def prepare_features_and_target(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from raw data with proper preprocessing
        
        Args:
            data: Raw DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (processed features DataFrame, target Series)
        """
        logger.info(f"Preparing features and target from data with shape: {data.shape}")
        
        # Separate features and target
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Create features DataFrame (all columns except target)
        X = data.drop(columns=[target_column]).copy()
        
        # Create target Series
        y = data[target_column].copy()
        
        logger.info(f"Raw features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target value counts: {y.value_counts().to_dict()}")
        
        # Identify categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Categorical columns: {categorical_columns}")
        logger.info(f"Numerical columns: {numerical_columns}")
        
        # Handle missing values
        if X.isnull().any().any():
            logger.info("Handling missing values...")
            # Fill numerical missing values with median
            for col in numerical_columns:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if X[col].isnull().any():
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        # Encode categorical variables using one-hot encoding
        if categorical_columns:
            logger.info(f"Encoding categorical columns: {categorical_columns}")
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            logger.info(f"Features shape after encoding: {X_encoded.shape}")
        else:
            X_encoded = X
        
        logger.info(f"Final features shape: {X_encoded.shape}")
        logger.info(f"Feature columns: {list(X_encoded.columns)}")
        
        return X_encoded, y
    
    def create_new_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with new features added
        """
        logger.info("Creating new features")
        
        data = data.copy()
        
        # Tenure-based features
        if 'tenure' in data.columns:
            data['tenure_years'] = data['tenure'] / 12
            data['tenure_group'] = pd.cut(data['tenure'], 
                                        bins=[0, 12, 24, 48, 72], 
                                        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
        
        # Charges-based features
        if 'MonthlyCharges' in data.columns and 'TotalCharges' in data.columns:
            data['charges_per_month'] = data['TotalCharges'] / (data['tenure'] + 1)
            data['total_charges_log'] = np.log1p(data['TotalCharges'])
            data['monthly_charges_log'] = np.log1p(data['MonthlyCharges'])
            
            # Price sensitivity indicator
            data['high_monthly_charges'] = (data['MonthlyCharges'] > data['MonthlyCharges'].quantile(0.75)).astype(int)
            data['high_total_charges'] = (data['TotalCharges'] > data['TotalCharges'].quantile(0.75)).astype(int)
        
        # Service-based features
        service_columns = [col for col in data.columns if any(service in col.lower() 
                          for service in ['service', 'security', 'backup', 'protection', 'support'])]
        
        if service_columns:
            # Count of additional services
            data['num_services'] = 0
            for col in service_columns:
                if col in data.columns:
                    data['num_services'] += (data[col] == 'Yes').astype(int)
        
        # Contract and payment features
        if 'Contract' in data.columns:
            data['is_monthly_contract'] = (data['Contract'] == 'Month-to-month').astype(int)
            data['is_long_term_contract'] = (data['Contract'].isin(['One year', 'Two year'])).astype(int)
        
        if 'PaymentMethod' in data.columns:
            data['is_electronic_payment'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
            data['is_automatic_payment'] = data['PaymentMethod'].str.contains('automatic', na=False).astype(int)
        
        # Demographic features
        if 'SeniorCitizen' in data.columns and 'Partner' in data.columns:
            data['senior_no_partner'] = ((data['SeniorCitizen'] == 'Yes') & 
                                        (data['Partner'] == 'No')).astype(int)
        
        if 'Partner' in data.columns and 'Dependents' in data.columns:
            data['family_size'] = (data['Partner'] == 'Yes').astype(int) + (data['Dependents'] == 'Yes').astype(int)
        
        # Internet and phone service combinations
        if 'InternetService' in data.columns and 'PhoneService' in data.columns:
            data['has_internet'] = (data['InternetService'] != 'No').astype(int)
            data['has_phone'] = (data['PhoneService'] == 'Yes').astype(int)
            data['internet_and_phone'] = data['has_internet'] * data['has_phone']
        
        # Streaming services
        streaming_cols = [col for col in data.columns if 'streaming' in col.lower()]
        if streaming_cols:
            data['num_streaming_services'] = 0
            for col in streaming_cols:
                data['num_streaming_services'] += (data[col] == 'Yes').astype(int)
            data['is_streaming_user'] = (data['num_streaming_services'] > 0).astype(int)
        
        logger.info(f"Created {len(data.columns) - len(data.columns)} new features")
        
        return data
    
    def handle_categorical_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle categorical variable encoding
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of encoded training and test features
        """
        logger.info("Encoding categorical variables")
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy() if X_test is not None else None
        
        # Get categorical columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            logger.info("No categorical columns found")
            return X_train_encoded, X_test_encoded
        
        # Handle ordinal encoding for specified columns
        if 'ordinal_features' in self.config:
            for col, order in self.config['ordinal_features'].items():
                if col in categorical_cols:
                    # Create mapping
                    mapping = {val: i for i, val in enumerate(order)}
                    
                    # Apply to training data
                    X_train_encoded[col] = X_train[col].map(mapping)
                    
                    # Apply to test data if provided
                    if X_test_encoded is not None:
                        X_test_encoded[col] = X_test[col].map(mapping)
                    
                    # Remove from categorical list
                    categorical_cols.remove(col)
                    logger.info(f"Applied ordinal encoding to {col}")
        
        # Apply one-hot encoding to remaining categorical variables
        if categorical_cols:
            if self.config['encoding_method'] == 'OneHotEncoder':
                # Use get_dummies for simplicity and consistency
                X_train_encoded = pd.get_dummies(X_train_encoded, columns=categorical_cols, drop_first=True)
                
                if X_test_encoded is not None:
                    X_test_encoded = pd.get_dummies(X_test_encoded, columns=categorical_cols, drop_first=True)
                    
                    # Ensure same columns in both sets
                    train_cols = set(X_train_encoded.columns)
                    test_cols = set(X_test_encoded.columns)
                    
                    # Add missing columns to test set
                    for col in train_cols - test_cols:
                        X_test_encoded[col] = 0
                    
                    # Remove extra columns from test set
                    for col in test_cols - train_cols:
                        X_test_encoded = X_test_encoded.drop(col, axis=1)
                    
                    # Reorder columns to match training set
                    X_test_encoded = X_test_encoded[X_train_encoded.columns]
                
                logger.info(f"Applied one-hot encoding to {len(categorical_cols)} categorical columns")
        
        return X_train_encoded, X_test_encoded
    
    def handle_numerical_scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle numerical variable scaling
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled training and test features
        """
        logger.info("Scaling numerical variables")
        
        # Get numerical columns
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            logger.info("No numerical columns found")
            return X_train, X_test
        
        # Initialize scaler
        if self.config['scaling_method'] == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.config['scaling_method'] == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.config['scaling_method']}")
            return X_train, X_test
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        logger.info(f"Scaled {len(numerical_cols)} numerical columns using {self.config['scaling_method']}")
        
        return X_train_scaled, X_test_scaled
    
    def create_preprocessing_pipeline(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """
        Create a comprehensive preprocessing pipeline
        
        Args:
            X_train: Training features
            
        Returns:
            Fitted ColumnTransformer pipeline
        """
        logger.info("Creating preprocessing pipeline")
        
        # Identify column types
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Create transformers
        transformers = []
        
        # Numerical transformer
        if numerical_cols:
            if self.config['scaling_method'] == 'StandardScaler':
                num_transformer = StandardScaler()
            else:
                num_transformer = MinMaxScaler()
            
            transformers.append(('num', num_transformer, numerical_cols))
        
        # Categorical transformer
        if categorical_cols:
            cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
            transformers.append(('cat', cat_transformer, categorical_cols))
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(X_train)
        
        logger.info("Preprocessing pipeline created and fitted")
        
        return self.preprocessor
    
    def apply_preprocessing_pipeline(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Apply the preprocessing pipeline to data
        
        Args:
            X: Input features
            fit: Whether to fit the pipeline (True for training data)
            
        Returns:
            Transformed features as numpy array
        """
        if self.preprocessor is None:
            if fit:
                self.create_preprocessing_pipeline(X)
            else:
                raise ValueError("Preprocessor not fitted. Call with fit=True first.")
        
        if fit:
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            X_transformed = self.preprocessor.transform(X)
        
        return X_transformed
    
    def handle_class_imbalance(self, X: Union[pd.DataFrame, np.ndarray], y: pd.Series, 
                             strategy: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using various sampling techniques
        
        Args:
            X: Features
            y: Target variable
            strategy: Sampling strategy to use
            
        Returns:
            Tuple of resampled features and target
        """
        strategy = strategy or self.config.get('strategy', 'SMOTE')
        
        logger.info(f"Handling class imbalance using {strategy}")
        
        # Original class distribution
        original_dist = pd.Series(y).value_counts().sort_index()
        logger.info(f"Original class distribution: {original_dist.to_dict()}")
        
        # Apply sampling strategy
        if strategy == 'SMOTE':
            sampler = SMOTE(
                sampling_strategy=IMBALANCE_CONFIG['sampling_strategy'],
                k_neighbors=IMBALANCE_CONFIG['k_neighbors'],
                random_state=IMBALANCE_CONFIG['random_state']
            )
        elif strategy == 'ADASYN':
            sampler = ADASYN(
                sampling_strategy=IMBALANCE_CONFIG['sampling_strategy'],
                random_state=IMBALANCE_CONFIG['random_state']
            )
        elif strategy == 'RandomUnderSampler':
            sampler = RandomUnderSampler(
                sampling_strategy=IMBALANCE_CONFIG['sampling_strategy'],
                random_state=IMBALANCE_CONFIG['random_state']
            )
        elif strategy == 'SMOTEENN':
            sampler = SMOTEENN(
                random_state=IMBALANCE_CONFIG['random_state']
            )
        else:
            logger.warning(f"Unknown sampling strategy: {strategy}")
            return X, y
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # New class distribution
        new_dist = pd.Series(y_resampled).value_counts().sort_index()
        logger.info(f"New class distribution: {new_dist.to_dict()}")
        
        return X_resampled, y_resampled
    
    def get_feature_importance_preprocessing(self, X: pd.DataFrame) -> List[str]:
        """
        Get feature names after preprocessing
        
        Args:
            X: Original features DataFrame
            
        Returns:
            List of feature names after preprocessing
        """
        if self.preprocessor is None:
            return list(X.columns)
        
        feature_names = []
        
        # Get feature names from transformers
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older sklearn versions
                    feature_names.extend([f"{col}_{cat}" for col in columns 
                                        for cat in transformer.categories_[columns.get_loc(col)][1:]])
        
        self.feature_names = feature_names
        return feature_names
    
    def save_preprocessor(self, filename: str = "preprocessor.pkl") -> None:
        """
        Save the fitted preprocessor
        
        Args:
            filename: Name of the file to save
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor to save")
            return
        
        filepath = MODELS_DIR / filename
        joblib.dump(self.preprocessor, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filename: str = "preprocessor.pkl") -> None:
        """
        Load a saved preprocessor
        
        Args:
            filename: Name of the file to load
        """
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            logger.error(f"Preprocessor file not found: {filepath}")
            return
        
        self.preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def complete_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series, apply_sampling: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Complete feature engineering pipeline
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            apply_sampling: Whether to apply class imbalance handling
            
        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train_processed, feature_names)
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # Step 1: Create new features
        X_train_engineered = self.create_new_features(X_train)
        X_test_engineered = self.create_new_features(X_test)
        
        # Step 2: Handle encoding and scaling manually or use pipeline
        X_train_encoded, X_test_encoded = self.handle_categorical_encoding(
            X_train_engineered, X_test_engineered
        )
        
        X_train_scaled, X_test_scaled = self.handle_numerical_scaling(
            X_train_encoded, X_test_encoded
        )
        
        # Step 3: Apply class imbalance handling if requested
        y_train_processed = y_train.copy()
        if apply_sampling:
            X_train_final, y_train_processed = self.handle_class_imbalance(
                X_train_scaled.values, y_train
            )
        else:
            X_train_final = X_train_scaled.values
        
        X_test_final = X_test_scaled.values
        
        # Step 4: Get feature names
        feature_names = list(X_train_scaled.columns)
        
        logger.info("Feature engineering pipeline completed")
        logger.info(f"Final training shape: {X_train_final.shape}")
        logger.info(f"Final test shape: {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train_processed, feature_names

# Example usage and testing
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load and prepare data
    loader = DataLoader()
    try:
        data = loader.load_raw_data()
    except FileNotFoundError:
        data = loader.generate_sample_data()
    
    cleaned_data = loader.clean_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Apply complete feature engineering
    X_train_processed, X_test_processed, y_train_processed, feature_names = engineer.complete_feature_engineering(
        X_train, X_test, y_train
    )
    
    print(f"\nOriginal training shape: {X_train.shape}")
    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Original class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Processed class distribution: {pd.Series(y_train_processed).value_counts().to_dict()}")
    
    # Save preprocessor
    engineer.save_preprocessor()
    
    print("\nFeature engineering completed successfully!")