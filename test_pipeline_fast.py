#!/usr/bin/env python3
"""
Fast test of the complete pipeline with minimal hyperparameter search
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split

try:
    print("🚀 Testing complete pipeline...")
    
    # Step 1: Load data
    print("\n📁 Loading data...")
    data_loader = DataLoader()
    raw_data = data_loader.load_raw_data("data/raw/employee_data.csv")
    print(f"✅ Data loaded: {raw_data.shape}")
    
    # Step 2: Feature engineering
    print("\n🔧 Feature engineering...")
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_features_and_target(raw_data, DATA_CONFIG['target_column'])
    
    # Convert boolean columns to integers
    bool_columns = X.select_dtypes(include=['bool']).columns
    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)
    
    print(f"✅ Features prepared: {X.shape}")
    
    # Step 3: Split data
    print("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split: Train({X_train.shape[0]}), Test({X_test.shape[0]})")
    
    # Step 4: Train one model with minimal hyperparameter search
    print("\n🎯 Training model (quick test)...")
    trainer = ModelTrainer()
    
    # Override CV config for faster testing
    trainer.cv_config['cv_folds'] = 2
    trainer.cv_config['n_iter'] = 3
    
    # Train just one model using train_model_with_cv
    try:
        model_results = trainer.train_model_with_cv(
            'random_forest', 
            X_train.values, 
            y_train.values, 
            optimization_method='random_search'
        )
        
        if model_results and 'error' not in model_results:
            print(f"✅ Model trained successfully!")
            
            # Handle cv_score (this is what the method actually returns)
            cv_score = model_results.get('cv_score', 'N/A')
            if isinstance(cv_score, (int, float)):
                print(f"   CV Score: {cv_score:.3f}")
            else:
                print(f"   CV Score: {cv_score}")
            
            # Show optimization method used
            opt_method = model_results.get('optimization_method', 'unknown')
            print(f"   Optimization: {opt_method}")
            
            # Check if model is stored
            print(f"   Model stored: {'random_forest' in trainer.trained_models}")
        else:
            print(f"❌ Model training failed: {model_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Model training failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Pipeline test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
