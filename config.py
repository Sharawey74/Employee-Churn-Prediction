"""
Configuration file for the Employee Turnover Prediction project.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFY = True

# Visualization configuration
FIGURE_SIZE = (10, 6)
DPI = 100
STYLE = 'seaborn-v0_8'

# Model hyperparameters
DECISION_TREE_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': 'gini'
}

RANDOM_FOREST_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': 'gini'
}

# Target variable
TARGET_COLUMN = 'quit'

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = ['department', 'salary']

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'