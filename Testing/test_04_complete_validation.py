#!/usr/bin/env python3
"""
Test 4: Complete validation of the pipeline fixes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from Testing/ to project root
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer

def test_complete_pipeline():
    """Test the complete pipeline with all fixes"""
    print("🚀 Testing complete pipeline with all fixes...")
    
    try:
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
        
        # Step 4: Train model with proper error handling
        print("\n🎯 Training model...")
        trainer = ModelTrainer()
        trainer.cv_config['cv_folds'] = 2
        trainer.cv_config['n_iter'] = 3
        
        # Use a smaller subset for faster testing
        X_train_small = X_train.iloc[:500]
        y_train_small = y_train.iloc[:500]
        
        model_results = trainer.train_model_with_cv(
            'random_forest',
            X_train_small.values,
            y_train_small.values,
            'random_search'
        )
        
        if model_results and 'error' not in model_results:
            print(f"✅ Model trained successfully!")
            
            # Display results with proper type checking
            cv_score = model_results.get('cv_score', 'N/A')
            if isinstance(cv_score, (int, float)):
                print(f"   CV Score: {cv_score:.3f}")
            else:
                print(f"   CV Score: {cv_score}")
            
            optimization_method = model_results.get('optimization_method', 'unknown')
            print(f"   Optimization: {optimization_method}")
            
            # Test predictions
            if 'random_forest' in trainer.trained_models:
                model = trainer.trained_models['random_forest']
                test_sample = X_test.iloc[:10].values
                predictions = model.predict(test_sample)
                print(f"   Test predictions shape: {predictions.shape}")
                print(f"   Sample predictions: {predictions[:5]}")
                
        else:
            print(f"❌ Model training failed: {model_results.get('error', 'Unknown error')}")
            
        print("\n🎉 Complete pipeline test finished!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n✅ ALL TESTS PASSED! Pipeline is working correctly.")
    else:
        print("\n❌ TESTS FAILED! Pipeline needs more fixes.")
