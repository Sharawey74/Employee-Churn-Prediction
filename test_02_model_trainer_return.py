#!/usr/bin/env python3
"""
Test 2: Check what ModelTrainer.train_model_with_cv actually returns
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.model_trainer import ModelTrainer
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader
from src.config import DATA_CONFIG

def test_model_trainer_return():
    """Test what train_model_with_cv returns"""
    print("🧪 Testing ModelTrainer return values...")
    
    # Load small dataset for quick test
    loader = DataLoader()
    data = loader.load_raw_data("data/raw/employee_data.csv")
    
    # Get features
    fe = FeatureEngineer()
    X, y = fe.prepare_features_and_target(data, DATA_CONFIG['target_column'])
    bool_columns = X.select_dtypes(include=['bool']).columns
    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)
    
    # Use small subset for speed
    X_small = X.iloc[:500]
    y_small = y.iloc[:500]
    
    # Initialize trainer with minimal config
    trainer = ModelTrainer()
    trainer.cv_config['cv_folds'] = 2
    trainer.cv_config['n_iter'] = 2
    
    print("🎯 Training model and checking return value...")
    
    # Train model and inspect result
    result = trainer.train_model_with_cv(
        'random_forest',
        X_small.values,
        y_small.values,
        'random_search'
    )
    
    print(f"\n📊 Result type: {type(result)}")
    print(f"📋 Result keys: {list(result.keys()) if result else 'None'}")
    
    if result:
        for key, value in result.items():
            print(f"   {key}: {value} (type: {type(value)})")
    
    print("\n✅ ModelTrainer return test completed!")
    return result

if __name__ == "__main__":
    test_model_trainer_return()
