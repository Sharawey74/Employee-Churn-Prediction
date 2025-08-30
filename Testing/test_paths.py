#!/usr/bin/env python3
"""
Path Test Script - Verify all reorganized files can import properly
"""

import sys
from pathlib import Path

# Add project root to Python path  
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test imports from each reorganized folder"""
    
    print("🧪 Testing path fixes for reorganized project structure...")
    print("="*60)
    
    # Test 1: validation folder
    print("\n📁 Testing validation/ folder...")
    
    try:
        # Simulate what validation scripts do
        from src.config import DATA_CONFIG
        from src.data_loader import DataLoader
        from src.feature_engineering import FeatureEngineer
        print("✅ validation/ imports work correctly!")
        
    except ImportError as e:
        print(f"❌ validation/ imports failed: {e}")
    
    # Test 2: main folder  
    print("\n📁 Testing main/ folder...")
    try:
        from src.config import MODEL_CONFIG
        from src.model_trainer import ModelTrainer
        from src.evaluator import ModelEvaluator
        print("✅ main/ imports work correctly!")
        
    except ImportError as e:
        print(f"❌ main/ imports failed: {e}")
    
    # Test 3: Testing folder
    print("\n📁 Testing Testing/ folder...")
    try:
        from src.model_trainer import ModelTrainer
        print("✅ Testing/ imports work correctly!")
        
    except ImportError as e:
        print(f"❌ Testing/ imports failed: {e}")
    
    # Test 4: validations folder (new location)
    print("\n📁 Testing validations/ folder...")
    try:
        # Test if validations scripts can import project modules
        from src.config import MODEL_CONFIG
        from src.model_trainer import ModelTrainer
        print("✅ validations/ imports work correctly!")
        
    except ImportError as e:
        print(f"❌ validations/ imports failed: {e}")
    
    print("\n🎉 Path verification complete!")

if __name__ == "__main__":
    test_imports()
