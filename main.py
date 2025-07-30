"""
Main execution script for Employee Turnover Prediction project.

This script runs all 9 tasks outlined in the README:
1. Import Libraries
2. Exploratory Data Analysis
3. Encode Categorical Features
4. Visualize Class Imbalance
5. Create Training and Validation Sets
6-7. Build Decision Tree Classifier with Interactive Controls
8. Build Random Forest Classifier with Interactive Controls
9. Feature Importance Plots and Evaluation Metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Task 1: Import Libraries (handled by module imports)
from src.data_loader import DataLoader
from src.feature_encoder import FeatureEncoder
from src.visualizer import DataVisualizer
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from src.models.evaluator import ModelEvaluator
from utils.helpers import setup_logging, create_directory_structure
from utils.constants import TARGET_COLUMN
from config import RANDOM_STATE, TEST_SIZE

from sklearn.model_selection import train_test_split

def main():
    """
    Main function to execute all 9 tasks.
    """
    # Setup logging and directories
    logger = setup_logging()
    create_directory_structure()
    
    logger.info("Starting Employee Turnover Prediction Pipeline")
    logger.info("=" * 60)
    
    try:
        # Task 1: Import Libraries
        logger.info("Task 1: Import Libraries - COMPLETED")
        print("✓ Task 1: Import Libraries - All necessary modules imported")
        
        # Task 2: Exploratory Data Analysis
        logger.info("Task 2: Starting Exploratory Data Analysis")
        print("\n" + "="*50)
        print("Task 2: Exploratory Data Analysis")
        print("="*50)
        
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_data()
        
        print(f"Dataset loaded successfully: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Get data information
        data_info = data_loader.get_data_info()
        print(f"Target distribution: {data_info['target_distribution']}")
        
        # Perform EDA
        visualizer = DataVisualizer()
        eda_figures = visualizer.exploratory_data_analysis(data)
        print(f"✓ Generated {len(eda_figures)} EDA visualizations")
        
        # Task 3: Encode Categorical Features
        logger.info("Task 3: Encoding Categorical Features")
        print("\n" + "="*50)
        print("Task 3: Encode Categorical Features")
        print("="*50)
        
        # Split into features and target
        X, y = data_loader.get_feature_target_split()
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Encode categorical features
        encoder = FeatureEncoder()
        X_encoded = encoder.fit_transform(X)
        feature_names = encoder.get_feature_names()
        
        print(f"Original features: {X.shape[1]}")
        print(f"Encoded features: {X_encoded.shape[1]}")
        print(f"✓ Categorical encoding completed")
        
        # Show encoding info
        encoding_info = encoder.get_encoding_info()
        for col, mapping in encoding_info["encoding_mappings"].items():
            print(f"  {col}: {len(mapping['original_values'])} categories -> "
                  f"{len(mapping['dummy_columns'])} dummy variables")
        
        # Task 4: Visualize Class Imbalance
        logger.info("Task 4: Visualizing Class Imbalance")
        print("\n" + "="*50)
        print("Task 4: Visualize Class Imbalance")
        print("="*50)
        
        class_fig, balance_stats = visualizer.visualize_class_imbalance(y)
        print(f"Class balance ratio: {balance_stats['balance_ratio']:.3f}")
        print(f"Class distribution: {balance_stats['counts']}")
        print(f"✓ Class imbalance visualization completed")
        
        # Task 5: Create Training and Validation Sets
        logger.info("Task 5: Creating Training and Validation Sets")
        print("\n" + "="*50)
        print("Task 5: Create Training and Validation Sets")
        print("="*50)
        
        # Split data using stratified sampling
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape} (80%)")
        print(f"Validation set: {X_val.shape} (20%)")
        print(f"Test set: {X_test.shape} (20%)")
        print(f"✓ Data split completed with stratified sampling")
        
        # Verify class distribution
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        print(f"Validation class distribution: {y_val.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        # Tasks 6 & 7: Build Decision Tree Classifier
        logger.info("Tasks 6 & 7: Building Decision Tree Classifier")
        print("\n" + "="*50)
        print("Tasks 6 & 7: Build Decision Tree Classifier")
        print("="*50)
        
        # Create and train Decision Tree
        dt_model = DecisionTreeModel()
        dt_results = dt_model.train_model(
            X_train, y_train, X_val, y_val,
            max_depth=10, min_samples_split=5, min_samples_leaf=2, criterion='gini'
        )
        
        print(f"Decision Tree Results:")
        print(f"  Training Accuracy: {dt_results['train_accuracy']:.4f}")
        print(f"  Validation Accuracy: {dt_results['val_accuracy']:.4f}")
        print(f"  Overfitting: {dt_results['overfitting']:.4f}")
        
        # Display tree visualization
        tree_fig = dt_model.plot_tree_visualization(max_depth_display=3)
        print(f"✓ Decision Tree trained and visualized")
        
        # Get feature importance
        dt_importance = dt_model.get_feature_importance()
        print(f"Top 5 Important Features (Decision Tree):")
        print(dt_importance.head())
        
        # Task 8: Build Random Forest Classifier
        logger.info("Task 8: Building Random Forest Classifier")
        print("\n" + "="*50)
        print("Task 8: Build Random Forest Classifier")
        print("="*50)
        
        # Create and train Random Forest
        rf_model = RandomForestModel()
        rf_results = rf_model.train_model(
            X_train, y_train, X_val, y_val,
            n_estimators=100, max_depth=10, min_samples_split=5, 
            min_samples_leaf=2, criterion='gini'
        )
        
        print(f"Random Forest Results:")
        print(f"  Training Accuracy: {rf_results['train_accuracy']:.4f}")
        print(f"  Validation Accuracy: {rf_results['val_accuracy']:.4f}")
        print(f"  Overfitting: {rf_results['overfitting']:.4f}")
        
        # Display sample tree from forest
        sample_tree_fig = rf_model.plot_sample_tree(tree_index=0, max_depth_display=3)
        
        # Get tree statistics
        tree_stats = rf_model.get_tree_statistics()
        print(f"  Number of Trees: {tree_stats['n_estimators']}")
        print(f"  Average Tree Depth: {tree_stats['tree_depths']['mean']:.1f}")
        print(f"✓ Random Forest trained and visualized")
        
        # Get feature importance
        rf_importance = rf_model.get_feature_importance()
        print(f"Top 5 Important Features (Random Forest):")
        print(rf_importance.head())
        
        # Task 9: Feature Importance Plots and Evaluation Metrics
        logger.info("Task 9: Feature Importance Plots and Evaluation Metrics")
        print("\n" + "="*50)
        print("Task 9: Feature Importance Plots and Evaluation Metrics")
        print("="*50)
        
        # Create model evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate Decision Tree
        dt_evaluation = evaluator.create_comprehensive_report(
            dt_model.model, X_test, y_test, feature_names, "Decision Tree"
        )
        
        # Evaluate Random Forest
        rf_evaluation = evaluator.create_comprehensive_report(
            rf_model.model, X_test, y_test, feature_names, "Random Forest"
        )
        
        print(f"Decision Tree Test Results:")
        print(f"  Accuracy: {dt_evaluation['metrics']['accuracy']:.4f}")
        print(f"  Precision: {dt_evaluation['metrics']['precision']:.4f}")
        print(f"  Recall: {dt_evaluation['metrics']['recall']:.4f}")
        print(f"  F1-Score: {dt_evaluation['metrics']['f1_score']:.4f}")
        if 'auc_roc' in dt_evaluation['metrics']:
            print(f"  AUC-ROC: {dt_evaluation['metrics']['auc_roc']:.4f}")
        
        print(f"\nRandom Forest Test Results:")
        print(f"  Accuracy: {rf_evaluation['metrics']['accuracy']:.4f}")
        print(f"  Precision: {rf_evaluation['metrics']['precision']:.4f}")
        print(f"  Recall: {rf_evaluation['metrics']['recall']:.4f}")
        print(f"  F1-Score: {rf_evaluation['metrics']['f1_score']:.4f}")
        if 'auc_roc' in rf_evaluation['metrics']:
            print(f"  AUC-ROC: {rf_evaluation['metrics']['auc_roc']:.4f}")
        
        # Plot feature importance comparison
        models_dict = {
            'Decision Tree': dt_model.model,
            'Random Forest': rf_model.model
        }
        importance_comparison_fig = evaluator.plot_feature_importance_comparison(
            models_dict, feature_names
        )
        
        # Create model comparison
        model_results = {
            'Decision Tree': dt_evaluation,
            'Random Forest': rf_evaluation
        }
        comparison_fig = evaluator.compare_models(model_results)
        
        print(f"✓ Comprehensive evaluation completed")
        
        # Create summary dashboard
        logger.info("Creating Summary Dashboard")
        print("\n" + "="*50)
        print("Summary Dashboard")
        print("="*50)
        
        # Prepare model results for dashboard
        dashboard_results = {
            'Decision Tree': {
                'val_accuracy': dt_results['val_accuracy'],
                'test_accuracy': dt_evaluation['metrics']['accuracy']
            },
            'Random Forest': {
                'val_accuracy': rf_results['val_accuracy'],
                'test_accuracy': rf_evaluation['metrics']['accuracy']
            }
        }
        
        dashboard_fig = visualizer.create_summary_dashboard(data, dashboard_results)
        print(f"✓ Summary dashboard created")
        
        # Final Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Dataset: {data.shape[0]} samples, {data.shape[1]} original features")
        print(f"Encoded features: {X_encoded.shape[1]}")
        print(f"Class distribution: {balance_stats['percentages']}")
        
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<15} {'Val Acc':<10} {'Test Acc':<10} {'Overfitting':<12}")
        print("-" * 50)
        print(f"{'Decision Tree':<15} {dt_results['val_accuracy']:<10.4f} "
              f"{dt_evaluation['metrics']['accuracy']:<10.4f} "
              f"{dt_results['overfitting']:<12.4f}")
        print(f"{'Random Forest':<15} {rf_results['val_accuracy']:<10.4f} "
              f"{rf_evaluation['metrics']['accuracy']:<10.4f} "
              f"{rf_results['overfitting']:<12.4f}")
        
        print(f"\n✓ All 9 tasks completed successfully!")
        print(f"✓ Results and visualizations saved to outputs/ directory")
        
        logger.info("Employee Turnover Prediction Pipeline completed successfully")
        
        return {
            'data': data,
            'X_encoded': X_encoded,
            'feature_names': feature_names,
            'models': {
                'decision_tree': dt_model,
                'random_forest': rf_model
            },
            'results': {
                'decision_tree': dt_evaluation,
                'random_forest': rf_evaluation
            },
            'figures': {
                'eda': eda_figures,
                'class_imbalance': class_fig,
                'tree_visualization': tree_fig,
                'sample_tree': sample_tree_fig,
                'importance_comparison': importance_comparison_fig,
                'model_comparison': comparison_fig,
                'dashboard': dashboard_fig
            }
        }
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the main pipeline
    results = main()
    
    # Keep plots open for viewing
    try:
        plt.show()
    except:
        pass