"""
Basic tests for the Employee Turnover Prediction project.

This module contains unit tests to validate the core functionality.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader
from src.feature_encoder import FeatureEncoder
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from data.sample_data import generate_employee_data


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = generate_employee_data(100)
        self.sample_data.to_csv('test_data.csv', index=False)
        self.loader = DataLoader('test_data.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
    
    def test_load_data(self):
        """Test data loading functionality."""
        data = self.loader.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertTrue('quit' in data.columns)
    
    def test_get_data_info(self):
        """Test data information extraction."""
        self.loader.load_data()
        info = self.loader.get_data_info()
        self.assertIn('shape', info)
        self.assertIn('target_distribution', info)
        self.assertEqual(info['shape'], (100, 10))
    
    def test_feature_target_split(self):
        """Test feature-target splitting."""
        self.loader.load_data()
        X, y = self.loader.get_feature_target_split()
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertNotIn('quit', X.columns)


class TestFeatureEncoder(unittest.TestCase):
    """Test cases for FeatureEncoder class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'satisfaction_level': [0.5, 0.8, 0.3],
            'department': ['sales', 'IT', 'hr'],
            'salary': ['low', 'high', 'medium'],
            'quit': [1, 0, 1]
        })
        self.encoder = FeatureEncoder()
    
    def test_fit_transform(self):
        """Test encoding functionality."""
        X = self.sample_data.drop('quit', axis=1)
        X_encoded = self.encoder.fit_transform(X)
        
        # Should have more columns after encoding
        self.assertGreater(X_encoded.shape[1], X.shape[1])
        
        # Original categorical columns should be removed
        self.assertNotIn('department', X_encoded.columns)
        self.assertNotIn('salary', X_encoded.columns)
        
        # Dummy columns should be added
        dept_columns = [col for col in X_encoded.columns if col.startswith('department_')]
        salary_columns = [col for col in X_encoded.columns if col.startswith('salary_')]
        self.assertGreater(len(dept_columns), 0)
        self.assertGreater(len(salary_columns), 0)
    
    def test_get_feature_names(self):
        """Test feature name extraction."""
        X = self.sample_data.drop('quit', axis=1)
        X_encoded = self.encoder.fit_transform(X)
        feature_names = self.encoder.get_feature_names()
        
        self.assertEqual(len(feature_names), X_encoded.shape[1])
        self.assertIn('satisfaction_level', feature_names)


class TestDecisionTreeModel(unittest.TestCase):
    """Test cases for DecisionTreeModel class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.X = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, n_samples),
            'feature2': np.random.uniform(0, 1, n_samples),
            'feature3': np.random.randint(0, 5, n_samples)
        })
        self.y = pd.Series(np.random.binomial(1, 0.3, n_samples))
        
        # Split data
        split_idx = int(0.8 * n_samples)
        self.X_train = self.X[:split_idx]
        self.X_val = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_val = self.y[split_idx:]
        
        self.model = DecisionTreeModel()
    
    def test_create_model(self):
        """Test model creation."""
        dt = self.model.create_model(max_depth=5, min_samples_split=2)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.max_depth, 5)
        self.assertEqual(dt.min_samples_split, 2)
    
    def test_train_model(self):
        """Test model training."""
        results = self.model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_depth=3
        )
        
        self.assertIn('train_accuracy', results)
        self.assertIn('val_accuracy', results)
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.model)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        importance_df = self.model.get_feature_importance()
        self.assertEqual(len(importance_df), 3)  # 3 features
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)


class TestRandomForestModel(unittest.TestCase):
    """Test cases for RandomForestModel class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.X = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, n_samples),
            'feature2': np.random.uniform(0, 1, n_samples),
            'feature3': np.random.randint(0, 5, n_samples)
        })
        self.y = pd.Series(np.random.binomial(1, 0.3, n_samples))
        
        # Split data
        split_idx = int(0.8 * n_samples)
        self.X_train = self.X[:split_idx]
        self.X_val = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_val = self.y[split_idx:]
        
        self.model = RandomForestModel()
    
    def test_create_model(self):
        """Test model creation."""
        rf = self.model.create_model(n_estimators=50, max_depth=5)
        self.assertIsNotNone(rf)
        self.assertEqual(rf.n_estimators, 50)
        self.assertEqual(rf.max_depth, 5)
    
    def test_train_model(self):
        """Test model training."""
        results = self.model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_estimators=10, max_depth=3
        )
        
        self.assertIn('train_accuracy', results)
        self.assertIn('val_accuracy', results)
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.model)
    
    def test_get_tree_statistics(self):
        """Test tree statistics extraction."""
        self.model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_estimators=5
        )
        
        stats = self.model.get_tree_statistics()
        self.assertIn('n_estimators', stats)
        self.assertIn('tree_depths', stats)
        self.assertEqual(stats['n_estimators'], 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline integration."""
        # Generate sample data
        data = generate_employee_data(200)
        
        # Load data
        data_path = 'integration_test_data.csv'
        data.to_csv(data_path, index=False)
        
        try:
            loader = DataLoader(data_path)
            loaded_data = loader.load_data()
            
            # Split features and target
            X, y = loader.get_feature_target_split()
            
            # Encode features
            encoder = FeatureEncoder()
            X_encoded = encoder.fit_transform(X)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Decision Tree
            dt_model = DecisionTreeModel()
            dt_results = dt_model.train_model(
                X_train, y_train, X_val, y_val, max_depth=5
            )
            
            # Train Random Forest
            rf_model = RandomForestModel()
            rf_results = rf_model.train_model(
                X_train, y_train, X_val, y_val, n_estimators=10, max_depth=5
            )
            
            # Verify results
            self.assertGreater(dt_results['train_accuracy'], 0.5)
            self.assertGreater(dt_results['val_accuracy'], 0.5)
            self.assertGreater(rf_results['train_accuracy'], 0.5)
            self.assertGreater(rf_results['val_accuracy'], 0.5)
            
            print("✓ Complete pipeline integration test passed")
            
        finally:
            if os.path.exists(data_path):
                os.remove(data_path)


if __name__ == '__main__':
    # Run tests
    print("Running Employee Turnover Prediction Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataLoader,
        TestFeatureEncoder,
        TestDecisionTreeModel,
        TestRandomForestModel,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("=" * 50)