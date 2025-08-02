"""
Configuration file for Customer Churn Prediction ML Pipeline
Contains all hyperparameters, paths, and model configurations
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Configuration
DATA_CONFIG = {
    'target_column': 'Churn',
    'customer_id_column': 'customerID',
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ],
    'categorical_features': [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ],
    'ordinal_features': {
        'Contract': ['Month-to-month', 'One year', 'Two year']
    },
    'scaling_method': 'StandardScaler',
    'encoding_method': 'OneHotEncoder'
}

# Class Imbalance Configuration
IMBALANCE_CONFIG = {
    'strategy': 'SMOTE',
    'sampling_strategy': 'auto',
    'k_neighbors': 5,
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'models': {
        'logistic_regression': {
            'class': 'LogisticRegression',
            'params': {
                'random_state': 42,
                'max_iter': 1000
            },
            'hyperparameters': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'random_forest': {
            'class': 'RandomForestClassifier',
            'params': {
                'random_state': 42,
                'n_jobs': -1
            },
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'class': 'GradientBoostingClassifier',
            'params': {
                'random_state': 42
            },
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'xgboost': {
            'class': 'XGBClassifier',
            'params': {
                'random_state': 42,
                'eval_metric': 'logloss'
            },
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    }
}

# Cross-Validation Configuration
CV_CONFIG = {
    'cv_folds': 5,
    'scoring': 'roc_auc',
    'n_iter': 50,  # for RandomizedSearchCV
    'random_state': 42,
    'n_jobs': -1
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
    'confusion_matrix', 'classification_report'
]

# Visualization Configuration
VIZ_CONFIG = {
    'figsize': (12, 8),
    'style': 'whitegrid',
    'palette': 'viridis',
    'dpi': 300,
    'save_format': 'png'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'logs' / 'pipeline.log'
}