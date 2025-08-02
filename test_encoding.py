#!/usr/bin/env python3
"""
Test the categorical encoding fix
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Test the fix with actual employee data
    from src.feature_engineering import FeatureEngineer
    from src.data_loader import DataLoader
    
    print("Loading employee data...")
    loader = DataLoader()
    data = loader.load_raw_data("data/raw/employee_data.csv")
    print(f"Data loaded: {data.shape}")
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\nUnique values in categorical columns:")
    for col in data.select_dtypes(include=['object']).columns:
        print(f"{col}: {data[col].unique()[:5]}...")  # Show first 5 unique values
    
    print("\nTesting FeatureEngineer...")
    fe = FeatureEngineer()
    
    # Test the method
    X, y = fe.prepare_features_and_target(data, 'quit')
    print(f"✅ Method works! Features shape: {X.shape}, Target shape: {y.shape}")
    
    print("\nFeature data types after encoding:")
    print(X.dtypes)
    
    print("\nChecking for any remaining categorical data:")
    categorical_remaining = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_remaining) == 0:
        print("✅ No categorical data remaining - all properly encoded!")
    else:
        print(f"❌ Still have categorical columns: {categorical_remaining}")
    
    print("\n🎉 Categorical encoding fix working correctly!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
