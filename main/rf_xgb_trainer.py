#!/usr/bin/env python3
"""
Quick Start Script for Random Forest & XGBoost Training
Simplified script to train and compare only RF and XGBoost models
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator


def quick_train_rf_xgb(data_path: str = None, optimization: str = 'random_search'):
    """
    Quick training function for Random Forest and XGBoost
    
    Args:
        data_path: Path to training data
        optimization: Optimization method ('random_search', 'grid_search', 'optuna')
    
    Returns:
        Dictionary with training results
    """
    
    print("="*60)
    print("🚀 RANDOM FOREST & XGBOOST QUICK TRAINER")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Step 1: Load data
        print("📁 Loading data...")
        data_loader = DataLoader()
        if data_path:
            raw_data = data_loader.load_raw_data(data_path)
        else:
            raw_data = data_loader.load_raw_data()
        
        print(f"✅ Data loaded: {raw_data.shape}")
        
        # Step 2: Feature engineering
        print("🔧 Engineering features...")
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.prepare_features_and_target(
            raw_data, target_column=DATA_CONFIG['target_column']
        )
        
        # Convert boolean columns to integers
        bool_columns = X.select_dtypes(include=['bool']).columns
        if len(bool_columns) > 0:
            X[bool_columns] = X[bool_columns].astype(int)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Features prepared: {X.shape}")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # Step 3: Train models
        print("🤖 Training Random Forest & XGBoost...")
        trainer = ModelTrainer()
        
        # Train both models
        results = {}
        for model_name in ['random_forest', 'xgboost']:
            print(f"   Training {model_name}...")
            result = trainer.train_model_with_cv(
                model_name, X_train.values, y_train.values, optimization
            )
            results[model_name] = result
        
        # Save models
        trainer.save_models(save_all=True)
        
        # Save JSON results
        trainer.save_results_to_json(X_test.values, y_test.values)
        
        # Step 4: Compare models
        print("📊 Comparing models...")
        comparison_df = trainer.compare_models()
        
        print("\n" + "="*40)
        print("🏆 MODEL COMPARISON RESULTS")
        print("="*40)
        for _, row in comparison_df.iterrows():
            print(f"{row['rank']}. {row['model']}: {row['cv_score']:.4f}")
        
        # Identify best model
        best_model_name, best_model, best_metrics = trainer.identify_best_model()
        
        print(f"\n🥇 WINNER: {best_model_name}")
        print(f"   CV Score: {best_metrics['cv_score']:.4f}")
        print(f"   Model Type: {best_metrics['model_type']}")
        
        # Step 5: Evaluate on test data
        print("\n🧪 Test Set Evaluation...")
        evaluator = ModelEvaluator()
        test_results = evaluator.evaluate_multiple_models(
            trainer.trained_models,
            X_test.values, y_test.values
        )
        
        print("\n📋 Test Results:")
        for model_name, metrics in test_results.items():
            if 'accuracy' in metrics:
                print(f"{model_name}:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1-Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        # Summary
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📁 Models saved to: {MODELS_DIR}")
        print(f"📄 JSON results saved to: {JSON_DIR}")
        print(f"📊 Results saved to: {RESULTS_DIR}")
        
        return {
            'trainer': trainer,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_metrics': best_metrics,
            'comparison': comparison_df,
            'test_results': test_results
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function for quick training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick RF & XGBoost Trainer')
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--optimization', type=str, default='random_search',
                       choices=['random_search', 'grid_search', 'optuna'],
                       help='Optimization method')
    
    args = parser.parse_args()
    
    results = quick_train_rf_xgb(args.data_path, args.optimization)
    
    if results:
        print("\n🎉 Training completed successfully!")
        return True
    else:
        print("\n💥 Training failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
