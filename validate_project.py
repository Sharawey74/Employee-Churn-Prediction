"""
Validation Script for Customer Churn Prediction Project
Tests all major components to ensure everything works correctly
"""

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test config
        from src.config import MODEL_CONFIG, DATA_CONFIG, FEATURE_CONFIG
        print("✅ Config module imported successfully")
        
        # Test data loader
        from src.data_loader import DataLoader
        print("✅ DataLoader imported successfully")
        
        # Test exploratory analysis
        from src.exploratory_analysis import ExploratoryAnalyzer
        print("✅ ExploratoryAnalyzer imported successfully")
        
        # Test feature engineering
        from src.feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
        
        # Test model trainer
        from src.model_trainer import ModelTrainer
        print("✅ ModelTrainer imported successfully")
        
        # Test evaluator
        from src.evaluator import ModelEvaluator
        print("✅ ModelEvaluator imported successfully")
        
        # Test visualizer
        from src.visualizer import ModelVisualizer
        print("✅ ModelVisualizer imported successfully")
        
        # Test utilities
        from utils.helpers import create_feature_summary_report
        from utils.preprocessing import handle_missing_values
        from utils.plotting import plot_correlation_heatmap
        print("✅ Utility modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False

def test_sample_data_processing():
    """Test data processing with sample data"""
    print("\n🧪 Testing data processing...")
    
    try:
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.exponential(2, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Add some missing values
        sample_data.loc[sample_data.sample(10).index, 'feature1'] = np.nan
        
        print(f"✅ Created sample dataset: {sample_data.shape}")
        
        # Test exploratory analysis
        from src.exploratory_analysis import ExploratoryAnalyzer
        analyzer = ExploratoryAnalyzer(sample_data, 'target')
        overview = analyzer.generate_data_overview()
        print("✅ Exploratory analysis completed")
        
        # Test feature summary
        from utils.helpers import create_feature_summary_report
        summary = create_feature_summary_report(sample_data, 'target')
        print("✅ Feature summary created")
        
        # Test missing value handling
        from utils.preprocessing import handle_missing_values
        processed_data, report = handle_missing_values(sample_data)
        print("✅ Missing value handling completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {str(e)}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test model training with sample data"""
    print("\n🧪 Testing model training...")
    
    try:
        # Create sample data for modeling
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, 
            random_state=42, n_informative=8
        )
        
        print(f"✅ Created classification dataset: {X.shape}")
        
        # Test model trainer
        from src.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # Train a single model (faster for testing)
        results = trainer.train_model_with_cv('random_forest', X, y, 'default')
        print("✅ Model training completed")
        
        # Test model predictions
        predictions = trainer.predict('random_forest', X[:10])
        print(f"✅ Model predictions: {len(predictions)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {str(e)}")
        traceback.print_exc()
        return False

def test_model_evaluation():
    """Test model evaluation"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        # Create sample data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Test evaluator
        from src.evaluator import ModelEvaluator, quick_evaluate
        
        # Quick evaluation
        results = quick_evaluate(model, X_test, y_test, 'test_model')
        print("✅ Quick model evaluation completed")
        
        # Full evaluator
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate_single_model(model, X_test, y_test, model_name='test_model')
        print("✅ Full model evaluation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation error: {str(e)}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🧪 Testing utility functions...")
    
    try:
        # Test helpers
        from utils.helpers import memory_usage_mb, detect_outliers_iqr
        
        sample_df = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.exponential(1, 100)
        })
        
        memory = memory_usage_mb(sample_df)
        outliers = detect_outliers_iqr(sample_df['col1'])
        print(f"✅ Helper functions work: memory={memory:.2f}MB, outliers={outliers.sum()}")
        
        # Test preprocessing
        from utils.preprocessing import detect_and_handle_outliers
        processed = detect_and_handle_outliers(sample_df, ['col1', 'col2'])
        print("✅ Preprocessing functions work")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions error: {str(e)}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration settings"""
    print("\n🧪 Testing configuration...")
    
    try:
        from src.config import MODEL_CONFIG, DATA_CONFIG, FEATURE_CONFIG, CV_CONFIG
        
        # Verify required configurations exist
        assert 'models' in MODEL_CONFIG, "Models not configured"
        assert 'target_column' in DATA_CONFIG, "Target column not configured"
        assert 'numerical_features' in FEATURE_CONFIG, "Feature config incomplete"
        assert 'cv_folds' in CV_CONFIG, "CV config incomplete"
        
        print(f"✅ Configuration valid: {len(MODEL_CONFIG['models'])} models configured")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        traceback.print_exc()
        return False

def run_validation():
    """Run all validation tests"""
    print("🔍 Customer Churn Prediction - Project Validation")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Data Processing Tests", test_sample_data_processing),
        ("Model Training Tests", test_model_training),
        ("Model Evaluation Tests", test_model_evaluation),
        ("Utility Tests", test_utilities)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python quick_start.py' for a demo")
        print("2. Run 'python main.py --help' to see all options")
        print("3. Check the notebooks/ directory for detailed examples")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
