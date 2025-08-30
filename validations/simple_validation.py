#!/usr/bin/env python3
"""
Simple Test Script for RF & XGBoost Project Restructuring
Validates that all changes work correctly
"""

import sys
import os
from pathlib import Path
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_config_changes():
    """Test that config only contains RF and XGBoost"""
    print("Testing configuration changes...")
    
    try:
        from src.config import MODEL_CONFIG, JSON_DIR
        
        # Check only RF and XGBoost in config
        model_names = list(MODEL_CONFIG['models'].keys())
        expected_models = ['random_forest', 'xgboost']
        
        assert set(model_names) == set(expected_models), f"Expected {expected_models}, got {model_names}"
        print("   PASS: Configuration restricted to RF and XGBoost")
        
        # Check JSON_DIR exists
        assert JSON_DIR.exists(), "JSON directory not created"
        print("   PASS: JSON directory configured and exists")
        
        return True
        
    except Exception as e:
        print(f"   FAIL: Config test failed: {str(e)}")
        return False

def test_model_trainer():
    """Test that ModelTrainer only supports RF and XGBoost"""
    print("Testing ModelTrainer restrictions...")
    
    try:
        from src.model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Test unsupported model raises error
        try:
            trainer.get_model_instance('logistic_regression')
            print("   FAIL: Should have rejected logistic_regression")
            return False
        except ValueError as e:
            if "Unsupported model" in str(e):
                print("   PASS: Correctly rejects unsupported models")
            else:
                print(f"   FAIL: Wrong error message: {str(e)}")
                return False
        
        # Test supported models work
        rf_model = trainer.get_model_instance('random_forest')
        xgb_model = trainer.get_model_instance('xgboost')
        
        assert rf_model.__class__.__name__ == 'RandomForestClassifier'
        assert xgb_model.__class__.__name__ == 'XGBClassifier'
        print("   PASS: Successfully creates RF and XGBoost models")
        
        # Test new methods exist
        assert hasattr(trainer, 'save_results_to_json'), "save_results_to_json method missing"
        assert hasattr(trainer, 'identify_best_model'), "identify_best_model method missing"
        print("   PASS: New methods available")
        
        return True
        
    except Exception as e:
        print(f"   FAIL: ModelTrainer test failed: {str(e)}")
        return False

def test_project_structure():
    """Test overall project structure"""
    print("Testing project structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "models",
        "results",
        "json",
        "src",
        "main"
    ]
    
    success = True
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"   PASS: {dir_path}/ exists")
        else:
            print(f"   FAIL: {dir_path}/ missing")
            success = False
    
    # Check key files exist
    key_files = [
        "src/config.py",
        "src/model_trainer.py",
        "main/main.py",
        "main/rf_xgb_trainer.py",
        "README_RF_XGB.md"
    ]
    
    for file_path in key_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"   PASS: {file_path} exists")
        else:
            print(f"   FAIL: {file_path} missing")
            success = False
    
    return success

def main():
    """Run all validation tests"""
    print("="*60)
    print("RF & XGBOOST PROJECT RESTRUCTURING VALIDATION")
    print("="*60)
    
    tests = [
        ("Configuration", test_config_changes),
        ("ModelTrainer", test_model_trainer),
        ("Project Structure", test_project_structure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED! Project restructuring successful.")
        return True
    else:
        print(f"\n{total - passed} test(s) failed. Please review issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
