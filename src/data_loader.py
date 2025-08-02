"""
Data Loading and Preprocessing Module for Customer Churn Prediction
Handles data ingestion, cleaning, and initial preprocessing
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split

from config import DATA_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing customer churn data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: Configuration dictionary, defaults to DATA_CONFIG
        """
        self.config = config or DATA_CONFIG
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw customer churn data from CSV file
        
        Args:
            file_path: Path to the data file, if None uses default
            
        Returns:
            DataFrame with raw data
        """
        if file_path is None:
            # Look for common churn dataset names
            possible_files = [
                RAW_DATA_DIR / "customer_churn.csv",
                RAW_DATA_DIR / "telco_customer_churn.csv",
                RAW_DATA_DIR / "churn_data.csv",
                RAW_DATA_DIR / "data.csv"
            ]
            
            file_path = None
            for path in possible_files:
                if path.exists():
                    file_path = path
                    break
                    
            if file_path is None:
                raise FileNotFoundError(
                    f"No data file found in {RAW_DATA_DIR}. "
                    f"Please place your dataset in one of these locations: {possible_files}"
                )
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.raw_data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate sample customer churn data for testing purposes
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic churn data
        """
        logger.info(f"Generating {n_samples} sample records")
        
        np.random.seed(42)
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
        
        # Generate demographic features
        genders = np.random.choice(['Male', 'Female'], n_samples)
        senior_citizens = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        partners = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
        dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        
        # Generate service features
        phone_services = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
        multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.4, 0.2])
        internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.4, 0.3])
        
        # Generate additional services
        online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        
        # Generate contract and billing features
        contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
        paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
        payment_methods = np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.3, 0.2, 0.25, 0.25])
        
        # Generate numerical features
        tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
        tenure = np.clip(tenure, 0, 72)
        
        monthly_charges = np.random.normal(65, 30, n_samples)
        monthly_charges = np.clip(monthly_charges, 18, 120)
        
        total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
        total_charges = np.maximum(total_charges, monthly_charges)
        
        # Generate target variable (churn) with realistic probabilities
        churn_prob = 0.1 + 0.3 * (contracts == 'Month-to-month') + 0.2 * (tenure < 12) + 0.1 * (monthly_charges > 80)
        churn = np.random.binomial(1, churn_prob, n_samples)
        churn = ['Yes' if c == 1 else 'No' for c in churn]
        
        # Create DataFrame
        data = {
            'customerID': customer_ids,
            'gender': genders,
            'SeniorCitizen': senior_citizens,
            'Partner': partners,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_services,
            'MultipleLines': multiple_lines,
            'InternetService': internet_services,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contracts,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_methods,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn': churn
        }
        
        self.raw_data = pd.DataFrame(data)
        
        # Save sample data
        sample_path = RAW_DATA_DIR / "sample_customer_churn.csv"
        self.raw_data.to_csv(sample_path, index=False)
        logger.info(f"Sample data saved to {sample_path}")
        
        return self.raw_data
    
    def clean_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and data types
        
        Args:
            data: DataFrame to clean, if None uses self.raw_data
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            data = self.raw_data.copy()
        else:
            data = data.copy()
        
        logger.info("Starting data cleaning process")
        
        # Handle TotalCharges column (often stored as string in Telco dataset)
        if 'TotalCharges' in data.columns:
            # Convert empty strings to NaN
            data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
            # Convert to numeric
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            
            # Fill NaN values with MonthlyCharges (for new customers)
            data['TotalCharges'].fillna(data['MonthlyCharges'], inplace=True)
        
        # Convert SeniorCitizen to categorical
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Handle missing values in categorical columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != self.config['customer_id_column']:
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"Filling {missing_count} missing values in {col} with mode")
                    data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Handle missing values in numerical columns
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Filling {missing_count} missing values in {col} with median")
                data[col].fillna(data[col].median(), inplace=True)
        
        # Remove duplicates
        initial_shape = data.shape[0]
        data.drop_duplicates(inplace=True)
        final_shape = data.shape[0]
        
        if initial_shape != final_shape:
            logger.info(f"Removed {initial_shape - final_shape} duplicate rows")
        
        logger.info(f"Data cleaning completed. Final shape: {data.shape}")
        self.processed_data = data
        
        return data
    
    def split_data(self, data: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            data: DataFrame to split, if None uses self.processed_data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if data is None:
            data = self.processed_data
        
        logger.info("Splitting data into train and test sets")
        
        # Separate features and target
        X = data.drop([self.config['target_column'], self.config['customer_id_column']], axis=1)
        y = data[self.config['target_column']]
        
        # Convert target to binary
        y = (y == 'Yes').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y if self.config['stratify'] else None
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training set churn rate: {y_train.mean():.3f}")
        logger.info(f"Test set churn rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, data: pd.DataFrame, filename: str = "processed_data.csv"):
        """
        Save processed data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save
        """
        filepath = PROCESSED_DATA_DIR / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def get_data_info(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset
        
        Args:
            data: DataFrame to analyze, if None uses self.processed_data
            
        Returns:
            Dictionary with dataset information
        """
        if data is None:
            data = self.processed_data
        
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numerical_columns': list(data.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        }
        
        if self.config['target_column'] in data.columns:
            target_counts = data[self.config['target_column']].value_counts()
            info['target_distribution'] = target_counts.to_dict()
            info['churn_rate'] = (data[self.config['target_column']] == 'Yes').mean()
        
        return info

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Try to load existing data, if not found generate sample data
    try:
        data = loader.load_raw_data()
    except FileNotFoundError:
        logger.info("No existing data found, generating sample data")
        data = loader.generate_sample_data()
    
    # Clean the data
    cleaned_data = loader.clean_data()
    
    # Get data information
    data_info = loader.get_data_info()
    print("\nDataset Information:")
    for key, value in data_info.items():
        print(f"{key}: {value}")
    
    # Split the data
    X_train, X_test, y_train, y_test = loader.split_data()
    
    # Save processed data
    loader.save_processed_data(cleaned_data)
    
    print("\nData loading and preprocessing completed successfully!")