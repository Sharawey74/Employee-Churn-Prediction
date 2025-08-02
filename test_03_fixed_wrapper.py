#!/usr/bin/env python3
"""
Test 3: Fix the wrapper function to handle correct return values
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

def fixed_train_single_model_wrapper(trainer, model_name, X_train, y_train, optimization_method='random_search'):
    """
    Fixed wrapper function with proper error handling for return values
    """
    try:
        # Use the existing train_model_with_cv method
        results = trainer.train_model_with_cv(
            model_name=model_name,
            X=X_train,
            y=y_train,
            optimization_method=optimization_method
        )
        
        if results and 'error' not in results:
            print(f"✅ {model_name} trained successfully!")
            
            # Handle best_score - it's actually cv_score
            best_score = results.get('cv_score', 'N/A')
            if isinstance(best_score, (int, float)):
                print(f"   Best score: {best_score:.3f}")
            else:
                print(f"   Best score: {best_score}")
            
            # Handle training_time - not available in current return
            print(f"   Training time: Not tracked")
            
            # Handle CV mean score
            cv_score = results.get('cv_score', 'N/A')
            if isinstance(cv_score, (int, float)):
                print(f"   CV mean score: {cv_score:.3f}")
            else:
                print(f"   CV mean score: {cv_score}")
                
            return results
        else:
            print(f"❌ {model_name} training failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Exception during {model_name} training: {str(e)}")
        return None

def test_fixed_wrapper():
    """Test the fixed wrapper function"""
    print("🧪 Testing fixed wrapper function...")
    
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
    X_small = X.iloc[:200]
    y_small = y.iloc[:200]
    
    # Initialize trainer with minimal config
    trainer = ModelTrainer()
    trainer.cv_config['cv_folds'] = 2
    trainer.cv_config['n_iter'] = 2
    
    print("🎯 Testing fixed wrapper...")
    
    # Test the fixed wrapper
    result = fixed_train_single_model_wrapper(
        trainer, 'random_forest', X_small.values, y_small.values, 'random_search'
    )
    
    if result:
        print("✅ Fixed wrapper works correctly!")
    else:
        print("❌ Fixed wrapper failed!")
    
    print("\n✅ Fixed wrapper test completed!")

if __name__ == "__main__":
    test_fixed_wrapper()
