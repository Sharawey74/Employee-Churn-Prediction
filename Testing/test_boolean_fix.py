#!/usr/bin/env python3
"""
Test the boolean-to-integer conversion fix
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from Testing/ to project root
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split

try:
    print("🔧 Testing boolean-to-integer conversion fix...")
    
    # Load and prepare data
    data_loader = DataLoader()
    raw_data = data_loader.load_raw_data("data/raw/employee_data.csv")
    
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_features_and_target(raw_data, DATA_CONFIG['target_column'])
    
    print(f"✅ Original features shape: {X.shape}")
    print(f"Original dtypes:\n{X.dtypes}")
    
    # Convert boolean columns to integers (as done in updated main.py)
    bool_columns = X.select_dtypes(include=['bool']).columns
    if len(bool_columns) > 0:
        print(f"\n🔧 Converting boolean columns to integers: {list(bool_columns)}")
        X[bool_columns] = X[bool_columns].astype(int)
    
    print(f"\n✅ After conversion dtypes:\n{X.dtypes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to numpy arrays
    X_train_values = X_train.values
    print(f"\n✅ X_train_values dtype: {X_train_values.dtype}")
    print(f"✅ Sample values: {X_train_values[0]}")
    
    # Test with models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    print(f"\n🧪 Testing models...")
    
    # Test RandomForest
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_train_values, y_train.values)
    rf_score = rf.score(X_test.values, y_test.values)
    print(f"✅ RandomForest accuracy: {rf_score:.3f}")
    
    # Test LogisticRegression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_values, y_train.values)
    lr_score = lr.score(X_test.values, y_test.values)
    print(f"✅ LogisticRegression accuracy: {lr_score:.3f}")
    
    print(f"\n🎉 Fix successful! Models train without errors.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
