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
JSON_DIR = PROJECT_ROOT / "json"  # New directory for JSON files

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, JSON_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Configuration
DATA_CONFIG = {
    'target_column': 'quit',  # Changed from 'Churn' to 'quit' for employee data
    'customer_id_column': None,  # No customer ID in employee data
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'satisfaction_level', 'last_evaluation', 'number_project', 
        'average_montly_hours', 'time_spend_company', 'Work_accident', 
        'promotion_last_5years'
    ],
    'categorical_features': [
        'department', 'salary'
    ],
    'ordinal_features': {
        'salary': ['low', 'medium', 'high']
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

# Model Configuration - RESTRICTED TO RANDOM FOREST AND XGBOOST ONLY
MODEL_CONFIG = {
    'models': {
        'random_forest': {
            'class': 'RandomForestClassifier',
            'params': {
                'random_state': 42,
                'n_jobs': -1
            },
            'hyperparameters': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 20, 30, 40, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        },
        'xgboost': {
            'class': 'XGBClassifier',
            'params': {
                'random_state': 42,
                'eval_metric': 'logloss',
                'n_jobs': -1
            },
            'hyperparameters': {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
                'colsample_bytree': [0.8, 0.85, 0.9, 0.95, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0]
            }
        }
    }
}

# Cross-Validation Configuration - Optimized for RF and XGBoost
CV_CONFIG = {
    'cv_folds': 5,
    'scoring': 'roc_auc',
    'n_iter': 100,  # Increased for better hyperparameter search
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