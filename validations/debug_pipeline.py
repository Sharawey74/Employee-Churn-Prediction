#!/usr/bin/env python3
"""
Debug script to check data at each step of the pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from validation/ to project root
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split

try:
    print("🔍 Debugging pipeline data...")
    
    # Step 1: Load data
    data_loader = DataLoader()
    raw_data = data_loader.load_raw_data("data/raw/employee_data.csv")
    print(f"✅ Raw data loaded: {raw_data.shape}")
    print(f"Raw data dtypes:\n{raw_data.dtypes}")
    print(f"Raw data sample:\n{raw_data.head()}")
    
    # Step 2: Feature engineering
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_features_and_target(
        raw_data, 
        target_column=DATA_CONFIG['target_column']
    )
    
    print(f"\n✅ Features prepared: {X.shape}")
    print(f"Features dtypes:\n{X.dtypes}")
    print(f"Features sample:\n{X.head()}")
    
    # Check for any remaining categorical data
    categorical_remaining = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_remaining) > 0:
        print(f"\n❌ PROBLEM: Still have categorical columns: {categorical_remaining}")
        for col in categorical_remaining:
            print(f"  - {col}: {X[col].unique()}")
    else:
        print(f"\n✅ All data properly encoded!")
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state'],
        stratify=y if DATA_CONFIG['stratify'] else None
    )
    
    print(f"\n✅ Data split: Train({X_train.shape[0]}), Test({X_test.shape[0]})")
    print(f"X_train dtypes:\n{X_train.dtypes}")
    
    # Step 4: Convert to values (as done in main.py)
    X_train_values = X_train.values
    y_train_values = y_train.values
    
    print(f"\n✅ Converted to numpy arrays:")
    print(f"X_train_values shape: {X_train_values.shape}")
    print(f"X_train_values dtype: {X_train_values.dtype}")
    print(f"y_train_values shape: {y_train_values.shape}")
    print(f"y_train_values dtype: {y_train_values.dtype}")
    
    # Check for any non-numeric values
    if X_train_values.dtype == 'object':
        print(f"\n❌ PROBLEM: X_train_values still contains objects!")
        print(f"Sample values: {X_train_values[0]}")
    else:
        print(f"\n✅ X_train_values is properly numeric!")
    
    # Test a simple model
    from sklearn.ensemble import RandomForestClassifier
    print(f"\n🧪 Testing RandomForest...")
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_train_values, y_train_values)
    print(f"✅ RandomForest trained successfully!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
