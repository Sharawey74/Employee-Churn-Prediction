"""
Constants for the Employee Turnover Prediction project.
"""

# Data columns
FEATURE_COLUMNS = [
    'satisfaction_level',
    'last_evaluation', 
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years'
]

CATEGORICAL_COLUMNS = ['department', 'salary']
TARGET_COLUMN = 'quit'

# Department categories
DEPARTMENTS = [
    'IT', 'RandD', 'accounting', 'hr', 'management', 
    'marketing', 'product_mng', 'sales', 'support', 'technical'
]

# Salary categories  
SALARY_LEVELS = ['low', 'medium', 'high']

# Model parameter ranges for interactive controls
DECISION_TREE_PARAMS_RANGE = {
    'max_depth': (1, 20),
    'min_samples_split': (2, 50),
    'min_samples_leaf': (1, 50),
    'criterion': ['gini', 'entropy']
}

RANDOM_FOREST_PARAMS_RANGE = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 50),
    'min_samples_leaf': (1, 50),
    'criterion': ['gini', 'entropy']
}

# Visualization settings
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd'
}

# File paths
DATA_FILE = 'data/raw/employee_data.csv'
PROCESSED_DATA_FILE = 'data/processed/employee_data_processed.csv'