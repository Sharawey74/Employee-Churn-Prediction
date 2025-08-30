#!/usr/bin/env python3
"""
Test Script for RF & XGBoost Project Restructuring
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
    print("🔧 Testing configuration changes...")
    
    try:
        from src.config import MODEL_CONFIG, JSON_DIR
        
        # Check only RF and XGBoost in config
        model_names = list(MODEL_CONFIG['models'].keys())
        expected_models = ['random_forest', 'xgboost']
        
        assert set(model_names) == set(expected_models), f"Expected {expected_models}, got {model_names}"
        print("   ✅ Configuration restricted to RF and XGBoost")
        
        # Check JSON_DIR exists
        assert JSON_DIR.exists(), "JSON directory not created"
        print("   ✅ JSON directory configured and exists")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {str(e)}")
        return False

def test_model_trainer():
    """Test that ModelTrainer only supports RF and XGBoost"""
    print("🤖 Testing ModelTrainer restrictions...")
    
    try:
        from src.model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Test unsupported model raises error
        try:
            trainer.get_model_instance('logistic_regression')
            print("   ❌ Should have rejected logistic_regression")
            return False
        except ValueError as e:
            if "Unsupported model" in str(e):
                print("   ✅ Correctly rejects unsupported models")
            else:
                print(f"   ❌ Wrong error message: {str(e)}")
                return False
        
        # Test supported models work
        rf_model = trainer.get_model_instance('random_forest')
        xgb_model = trainer.get_model_instance('xgboost')
        
        assert rf_model.__class__.__name__ == 'RandomForestClassifier'
        assert xgb_model.__class__.__name__ == 'XGBClassifier'
        print("   ✅ Successfully creates RF and XGBoost models")
        
        # Test new methods exist
        assert hasattr(trainer, 'save_results_to_json'), "save_results_to_json method missing"
        assert hasattr(trainer, 'identify_best_model'), "identify_best_model method missing"
        print("   ✅ New methods available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ModelTrainer test failed: {str(e)}")
        return False

def test_file_cleanup():
    """Test that old model files were removed"""
    print("🗂️ Testing file cleanup...")
    
    models_dir = PROJECT_ROOT / "models"
    
    # Files that should be removed
    removed_files = [
        "logistic_regression_model.pkl",
        "gradient_boosting_model.pkl", 
        "support_vector_machine_model.pkl"
    ]
    
    # Directories that should be removed
    removed_dirs = [
        "logistic_regression",
        "gradient_boosting"
    ]
    
    success = True
    
    for file in removed_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"   ❌ {file} still exists")
            success = False
        else:
            print(f"   ✅ {file} removed")
    
    for dir_name in removed_dirs:
        dir_path = models_dir / dir_name
        if dir_path.exists():
            print(f"   ❌ {dir_name}/ still exists")
            success = False
        else:
            print(f"   ✅ {dir_name}/ removed")
    
    # Files that should remain
    remaining_files = [
        "random_forest_model.pkl",
        "xgboost_model.pkl"
    ]
    
    for file in remaining_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"   ✅ {file} retained")
        else:
            print(f"   ⚠️ {file} not found (will be created during training)")
    
    return success

def test_json_directory():
    """Test JSON directory structure"""
    print("📄 Testing JSON directory...")
    
    try:
        from src.config import JSON_DIR
        
        assert JSON_DIR.exists(), "JSON directory doesn't exist"
        assert JSON_DIR.is_dir(), "JSON_DIR is not a directory"
        print("   ✅ JSON directory exists and is accessible")
        
        return True
        
    except Exception as e:
        print(f"   ❌ JSON directory test failed: {str(e)}")
        return False

def test_quick_trainer_exists():
    """Test that quick trainer script exists"""
    print("🚀 Testing quick trainer script...")
    
    try:
        quick_trainer_path = PROJECT_ROOT / "main" / "rf_xgb_trainer.py"
        
        assert quick_trainer_path.exists(), "Quick trainer script not found"
        
        # Read and check content
        with open(quick_trainer_path, 'r') as f:
            content = f.read()
        
        assert "quick_train_rf_xgb" in content, "Main function missing"
        assert "random_forest" in content, "RF not mentioned"
        assert "xgboost" in content, "XGBoost not mentioned"
        
        print("   ✅ Quick trainer script exists and looks correct")
        return True
        
    except Exception as e:
        print(f"   ❌ Quick trainer test failed: {str(e)}")
        return False

def test_project_structure():
    """Test overall project structure"""
    print("🏗️ Testing project structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "models",
        "results",
        "json",
        "src",
        "main",
        "validations"  # Add new validations directory
    ]
    
    success = True
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}/ exists")
        else:
            print(f"   ❌ {dir_path}/ missing")
            success = False
    
    # Check key files exist
    key_files = [
        "src/config.py",
        "src/model_trainer.py",
        "main/main.py",
        "main/rf_xgb_trainer.py",
        "README_RF_XGB.md",
        "validations/__init__.py"  # Add validations package check
    ]
    
    for file_path in key_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")
            success = False
    
    return success

def test_validations_scripts():
    """Test that validation scripts work from new location"""
    print("🔍 Testing validations scripts...")
    
    validation_scripts = [
        "analyze_cv.py",
        "analyze_pkl_detailed.py", 
        "analyze_pkl_files.py",
        "check_regularization.py",
        "debug_paths.py",
        "debug_trainer.py",
        "path_demo.py",
        "setup.py",
        "simple_validation.py"
    ]
    
    success = True
    validations_dir = PROJECT_ROOT / "validations"
    
    for script in validation_scripts:
        script_path = validations_dir / script
        if script_path.exists():
            print(f"   ✅ validations/{script} exists")
        else:
            print(f"   ❌ validations/{script} missing")
            success = False
    
    return success

def run_all_tests():
    """Run all validation tests"""
    print("="*60)
    print("🧪 RF & XGBOOST PROJECT RESTRUCTURING VALIDATION")
    print("="*60)
    
    tests = [
        ("Configuration", test_config_changes),
        ("ModelTrainer", test_model_trainer),
        ("File Cleanup", test_file_cleanup),
        ("JSON Directory", test_json_directory),
        ("Quick Trainer", test_quick_trainer_exists),
        ("Project Structure", test_project_structure),
        ("Validations Scripts", test_validations_scripts)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project restructuring successful.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
