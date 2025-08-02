#!/usr/bin/env python3
"""
Quick test script to validate the feature engineering fix
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Test import
    from src.feature_engineering import FeatureEngineer
    print("✅ FeatureEngineer import successful")
    
    # Test initialization
    fe = FeatureEngineer()
    print("✅ FeatureEngineer initialization successful")
    
    # Test method existence
    if hasattr(fe, 'prepare_features_and_target'):
        print("✅ prepare_features_and_target method exists")
    else:
        print("❌ prepare_features_and_target method missing")
    
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Test the method
    X, y = fe.prepare_features_and_target(sample_data, 'target')
    print(f"✅ Method works! Features shape: {X.shape}, Target shape: {y.shape}")
    
    print("\n🎉 All tests passed! The fix is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
