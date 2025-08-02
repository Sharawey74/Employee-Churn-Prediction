#!/usr/bin/env python3
"""
Main Execution Script for Customer Churn Prediction Project
This script orchestrates the complete ML pipeline from data loading to model evaluation
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import project modules
from src.config import *
from src.data_loader import DataLoader
from src.exploratory_analysis import ExploratoryAnalyzer
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from utils.helpers import setup_logging, create_feature_summary_report, print_feature_summary

def setup_project_directories():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR,
        RESULTS_DIR / "figures", RESULTS_DIR / "reports", RESULTS_DIR / "metrics"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logging.info("Project directories created/verified")


def load_and_explore_data(file_path: Optional[str] = None) -> tuple:
    """
    Load and perform initial data exploration
    
    Args:
        file_path: Optional path to data file
        
    Returns:
        Tuple of (raw_data, processed_data, eda_results)
    """
    logging.info("="*50)
    logging.info("STEP 1: DATA LOADING AND EXPLORATION")
    logging.info("="*50)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data
    if file_path:
        raw_data = data_loader.load_raw_data(file_path)
    else:
        # Try to find data in raw directory
        raw_data = data_loader.load_raw_data()
    
    logging.info(f"Loaded data with shape: {raw_data.shape}")
    
    # Initial data exploration
    analyzer = ExploratoryAnalyzer(raw_data, DATA_CONFIG['target_column'])
    
    # Generate comprehensive overview
    data_overview = analyzer.generate_data_overview()
    logging.info("Data overview generated")
    
    # Create and print feature summary
    feature_summary = create_feature_summary_report(raw_data, DATA_CONFIG['target_column'])
    print_feature_summary(feature_summary)
    
    return raw_data, data_overview, analyzer


def engineer_features(raw_data: pd.DataFrame) -> tuple:
    """
    Perform feature engineering
    
    Args:
        raw_data: Raw dataset
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_engineer)
    """
    logging.info("="*50)
    logging.info("STEP 2: FEATURE ENGINEERING")
    logging.info("="*50)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Prepare data for modeling
    X, y = feature_engineer.prepare_features_and_target(
        raw_data, 
        target_column=DATA_CONFIG['target_column']
    )
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state'],
        stratify=y if DATA_CONFIG['stratify'] else None
    )
    
    logging.info(f"Training set size: {X_train.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_engineer


def train_models(X_train: pd.DataFrame, y_train: pd.Series, 
                optimization_method: str = 'random_search') -> ModelTrainer:
    """
    Train multiple ML models
    
    Args:
        X_train: Training features
        y_train: Training target
        optimization_method: Hyperparameter optimization method
        
    Returns:
        Trained ModelTrainer instance
    """
    logging.info("="*50)
    logging.info("STEP 3: MODEL TRAINING")
    logging.info("="*50)
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train all models
    training_results = trainer.train_all_models(
        X_train.values, y_train.values, 
        optimization_method=optimization_method
    )
    
    # Save models
    trainer.save_models(save_all=True)
    
    return trainer


def evaluate_models(trainer: ModelTrainer, X_test: pd.DataFrame, y_test: pd.Series,
                   X_train: pd.DataFrame = None, y_train: pd.Series = None) -> ModelEvaluator:
    """
    Evaluate trained models
    
    Args:
        trainer: Trained ModelTrainer instance
        X_test: Test features
        y_test: Test target
        X_train: Training features (optional)
        y_train: Training target (optional)
        
    Returns:
        ModelEvaluator instance with results
    """
    logging.info("="*50)
    logging.info("STEP 4: MODEL EVALUATION")
    logging.info("="*50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all trained models
    evaluation_results = evaluator.evaluate_multiple_models(
        trainer.trained_models,
        X_test.values, y_test.values,
        X_train.values if X_train is not None else None,
        y_train.values if y_train is not None else None
    )
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(save_report=True)
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(report)
    
    # Create visualizations
    try:
        # Plot confusion matrices for each model
        for model_name in trainer.trained_models.keys():
            evaluator.plot_confusion_matrix(model_name, save_plot=True)
        
        # Plot ROC curves comparison
        evaluator.plot_roc_curves(trainer.trained_models, X_test.values, y_test.values, save_plot=True)
        
        # Plot Precision-Recall curves
        evaluator.plot_precision_recall_curves(trainer.trained_models, X_test.values, y_test.values, save_plot=True)
        
        # Plot model comparison
        evaluator.plot_model_comparison(save_plot=True)
        
        # Plot feature importance for tree-based models
        for model_name, model in trainer.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                evaluator.plot_feature_importance(
                    model, X_train.columns.tolist(), 
                    model_name=model_name, save_plot=True
                )
        
        logging.info("Evaluation visualizations created successfully")
        
    except Exception as e:
        logging.warning(f"Error creating visualizations: {str(e)}")
    
    # Save evaluation results
    evaluator.save_results()
    
    return evaluator


def main():
    """Main execution function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Customer Churn Prediction ML Pipeline')
    parser.add_argument('--data-path', type=str, help='Path to the dataset file')
    parser.add_argument('--optimization', type=str, default='random_search',
                       choices=['grid_search', 'random_search', 'optuna', 'default'],
                       help='Hyperparameter optimization method')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--models', nargs='+', 
                       choices=['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost'],
                       help='Specific models to train (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Setup project structure
        setup_project_directories()
        
        logging.info("🚀 Starting Customer Churn Prediction Pipeline")
        logging.info(f"Configuration: {args}")
        
        # Step 1: Load and explore data
        raw_data, data_overview, analyzer = load_and_explore_data(args.data_path)
        
        # Step 2: Feature engineering
        X_train, X_test, y_train, y_test, feature_engineer = engineer_features(raw_data)
        
        # Step 3: Model training
        trainer = train_models(X_train, y_train, args.optimization)
        
        # Filter models if specified
        if args.models:
            trainer.trained_models = {k: v for k, v in trainer.trained_models.items() 
                                    if k in args.models}
            logging.info(f"Filtered models to: {list(trainer.trained_models.keys())}")
        
        # Step 4: Model evaluation
        evaluator = evaluate_models(trainer, X_test, y_test, X_train, y_train)
        
        # Final summary
        logging.info("="*50)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        logging.info("="*50)
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        if evaluator.comparison_results and 'best_model' in evaluator.comparison_results:
            best_models = evaluator.comparison_results['best_model']
            print(f"🏆 Best Model by F1-Score: {best_models['by_f1']}")
            print(f"🎯 Best Model by Accuracy: {best_models['by_accuracy']}")
            if best_models['by_roc_auc']:
                print(f"📊 Best Model by ROC-AUC: {best_models['by_roc_auc']}")
        
        print(f"\n📁 Results saved to: {RESULTS_DIR}")
        print(f"📁 Models saved to: {MODELS_DIR}")
        print(f"📊 Visualizations saved to: {RESULTS_DIR / 'figures'}")
        print(f"📋 Reports saved to: {RESULTS_DIR / 'reports'}")
        
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
