#!/usr/bin/env python3
"""
Simple test to debug ModelTrainer issue
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def debug_trainer():
    """Debug ModelTrainer functionality"""
    
    try:
        print("Testing imports...")
        from src.config import MODEL_CONFIG
        print("✅ Config imported successfully")
        print(f"Available models: {list(MODEL_CONFIG['models'].keys())}")
        
        from src.model_trainer import ModelTrainer
        print("✅ ModelTrainer imported successfully")
        
        trainer = ModelTrainer()
        print("✅ ModelTrainer instantiated successfully")
        
        # Test supported models
        print("\nTesting supported models...")
        try:
            rf_model = trainer.get_model_instance('random_forest')
            print(f"✅ Random Forest: {rf_model.__class__.__name__}")
        except Exception as e:
            print(f"❌ Random Forest failed: {e}")
        
        try:
            xgb_model = trainer.get_model_instance('xgboost')
            print(f"✅ XGBoost: {xgb_model.__class__.__name__}")
        except Exception as e:
            print(f"❌ XGBoost failed: {e}")
        
        # Test unsupported model
        print("\nTesting unsupported model...")
        try:
            lr_model = trainer.get_model_instance('logistic_regression')
            print(f"❌ Should have failed: {lr_model.__class__.__name__}")
        except ValueError as e:
            print(f"✅ Correctly rejected: {e}")
        except Exception as e:
            print(f"❌ Wrong error type: {e}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Import or setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_trainer()
