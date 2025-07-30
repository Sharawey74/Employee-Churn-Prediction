"""
Interactive demo for Employee Turnover Prediction project.

This script provides an interactive version with UI controls for Tasks 6-8:
- Interactive Decision Tree building
- Interactive Random Forest building  
- Real-time parameter adjustment and visualization
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

from src.data_loader import DataLoader
from src.feature_encoder import FeatureEncoder
from src.visualizer import DataVisualizer
from src.models.decision_tree import (
    DecisionTreeModel, build_decision_tree_interactive,
    create_interactive_decision_tree_widget
)
from src.models.random_forest import (
    RandomForestModel, build_random_forest_interactive,
    create_interactive_random_forest_widget
)
from src.models.evaluator import ModelEvaluator
from utils.helpers import setup_logging, create_directory_structure
from utils.constants import TARGET_COLUMN
from config import RANDOM_STATE

from sklearn.model_selection import train_test_split

class InteractiveDemo:
    """
    Interactive demonstration class for Employee Turnover Prediction.
    
    Provides interactive widgets for real-time model training and evaluation.
    """
    
    def __init__(self):
        """Initialize the interactive demo."""
        self.logger = setup_logging()
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.is_data_loaded = False
        
        # Models
        self.dt_model = None
        self.rf_model = None
        
        # Results storage
        self.model_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for interactive modeling."""
        print("Loading and preparing data...")
        print("=" * 50)
        
        try:
            # Load data
            data_loader = DataLoader()
            self.data = data_loader.load_data()
            
            print(f"✓ Dataset loaded: {self.data.shape}")
            
            # Split features and target
            X, y = data_loader.get_feature_target_split()
            
            # Encode categorical features
            encoder = FeatureEncoder()
            X_encoded = encoder.fit_transform(X)
            self.feature_names = encoder.get_feature_names()
            
            print(f"✓ Features encoded: {X.shape[1]} -> {X_encoded.shape[1]}")
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_encoded, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
            )
            
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
            )
            
            self.X_train = X_train
            self.y_train = y_train
            
            print(f"✓ Data split - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
            
            self.is_data_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def interactive_decision_tree(self):
        """Launch interactive Decision Tree interface."""
        if not self.is_data_loaded:
            self.load_and_prepare_data()
        
        print("\n" + "=" * 60)
        print("INTERACTIVE DECISION TREE BUILDER")
        print("=" * 60)
        print("Use the sliders below to adjust Decision Tree parameters")
        print("and see real-time results!")
        print()
        
        try:
            # Try to create interactive widget
            create_interactive_decision_tree_widget(
                self.X_train, self.y_train, self.X_val, self.y_val
            )
        except ImportError:
            print("ipywidgets not available. Running static version...")
            # Fallback to static version
            self.static_decision_tree_demo()
    
    def interactive_random_forest(self):
        """Launch interactive Random Forest interface."""
        if not self.is_data_loaded:
            self.load_and_prepare_data()
        
        print("\n" + "=" * 60)
        print("INTERACTIVE RANDOM FOREST BUILDER")
        print("=" * 60)
        print("Use the sliders below to adjust Random Forest parameters")
        print("and see real-time results!")
        print()
        
        try:
            # Try to create interactive widget
            create_interactive_random_forest_widget(
                self.X_train, self.y_train, self.X_val, self.y_val
            )
        except ImportError:
            print("ipywidgets not available. Running static version...")
            # Fallback to static version
            self.static_random_forest_demo()
    
    def static_decision_tree_demo(self):
        """Static Decision Tree demonstration with multiple parameter sets."""
        print("Static Decision Tree Demo - Testing Multiple Configurations")
        print("-" * 60)
        
        # Test different parameter combinations
        param_combinations = [
            {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
            {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini'},
            {'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'criterion': 'entropy'},
            {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 10, 'criterion': 'gini'}
        ]
        
        best_model = None
        best_score = 0
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nConfiguration {i}: {params}")
            
            dt_model, results = build_decision_tree_interactive(
                self.X_train, self.y_train, self.X_val, self.y_val, **params
            )
            
            # Store results
            self.model_results[f'DecisionTree_Config_{i}'] = results
            
            if results['val_accuracy'] > best_score:
                best_score = results['val_accuracy']
                best_model = dt_model
        
        print(f"\n🏆 Best Decision Tree - Validation Accuracy: {best_score:.4f}")
        self.dt_model = best_model
    
    def static_random_forest_demo(self):
        """Static Random Forest demonstration with multiple parameter sets."""
        print("Static Random Forest Demo - Testing Multiple Configurations")
        print("-" * 60)
        
        # Test different parameter combinations
        param_combinations = [
            {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
            {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 10}
        ]
        
        best_model = None
        best_score = 0
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nConfiguration {i}: {params}")
            
            rf_model, results = build_random_forest_interactive(
                self.X_train, self.y_train, self.X_val, self.y_val, **params
            )
            
            # Store results
            self.model_results[f'RandomForest_Config_{i}'] = results
            
            if results['val_accuracy'] > best_score:
                best_score = results['val_accuracy']
                best_model = rf_model
        
        print(f"\n🏆 Best Random Forest - Validation Accuracy: {best_score:.4f}")
        self.rf_model = best_model
    
    def model_comparison_demo(self):
        """Demonstrate model comparison capabilities."""
        if not self.is_data_loaded:
            self.load_and_prepare_data()
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON DEMO")
        print("=" * 60)
        
        # Train best models
        print("Training optimized models...")
        
        # Decision Tree
        self.dt_model = DecisionTreeModel()
        dt_results = self.dt_model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_depth=10, min_samples_split=5, min_samples_leaf=2, criterion='gini'
        )
        
        # Random Forest
        self.rf_model = RandomForestModel()
        rf_results = self.rf_model.train_model(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_estimators=100, max_depth=10, min_samples_split=5, 
            min_samples_leaf=2, criterion='gini'
        )
        
        # Evaluate on test set
        evaluator = ModelEvaluator()
        
        dt_evaluation = evaluator.create_comprehensive_report(
            self.dt_model.model, self.X_test, self.y_test, 
            self.feature_names, "Decision Tree", save_plot=False
        )
        
        rf_evaluation = evaluator.create_comprehensive_report(
            self.rf_model.model, self.X_test, self.y_test, 
            self.feature_names, "Random Forest", save_plot=False
        )
        
        # Display comparison
        print("\nModel Performance Comparison:")
        print("-" * 40)
        print(f"{'Metric':<20} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 40)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            dt_val = dt_evaluation['metrics'][metric]
            rf_val = rf_evaluation['metrics'][metric]
            print(f"{metric:<20} {dt_val:<15.4f} {rf_val:<15.4f}")
        
        # Feature importance comparison
        models_dict = {
            'Decision Tree': self.dt_model.model,
            'Random Forest': self.rf_model.model
        }
        importance_fig = evaluator.plot_feature_importance_comparison(
            models_dict, self.feature_names, save_plot=False
        )
        
        # Model metrics comparison
        model_results = {
            'Decision Tree': dt_evaluation,
            'Random Forest': rf_evaluation
        }
        comparison_fig = evaluator.compare_models(model_results, save_plot=False)
        
        print(f"\n✓ Model comparison completed")
        return dt_evaluation, rf_evaluation
    
    def feature_importance_explorer(self):
        """Interactive feature importance exploration."""
        if not self.dt_model or not self.rf_model:
            print("Training models first...")
            self.model_comparison_demo()
        
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE EXPLORER")
        print("=" * 60)
        
        # Get feature importance from both models
        dt_importance = self.dt_model.get_feature_importance()
        rf_importance = self.rf_model.get_feature_importance()
        
        print("Top 10 Most Important Features:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Feature':<30} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 80)
        
        # Combine and rank features
        all_features = set(dt_importance['feature'].tolist() + rf_importance['feature'].tolist())
        
        combined_importance = []
        for feature in all_features:
            dt_imp = dt_importance[dt_importance['feature'] == feature]['importance'].iloc[0] \
                    if feature in dt_importance['feature'].values else 0
            rf_imp = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0] \
                    if feature in rf_importance['feature'].values else 0
            
            avg_imp = (dt_imp + rf_imp) / 2
            combined_importance.append((feature, dt_imp, rf_imp, avg_imp))
        
        # Sort by average importance
        combined_importance.sort(key=lambda x: x[3], reverse=True)
        
        # Display top 10
        for i, (feature, dt_imp, rf_imp, avg_imp) in enumerate(combined_importance[:10], 1):
            print(f"{i:<5} {feature:<30} {dt_imp:<15.4f} {rf_imp:<15.4f}")
        
        # Create visualizations
        evaluator = ModelEvaluator()
        
        # Individual model importance plots
        dt_fig = evaluator.plot_feature_importance(
            self.dt_model.model, self.feature_names, top_n=10, save_plot=False
        )
        
        rf_fig = evaluator.plot_feature_importance(
            self.rf_model.model, self.feature_names, top_n=10, save_plot=False
        )
        
        print(f"\n✓ Feature importance analysis completed")
    
    def run_full_demo(self):
        """Run the complete interactive demo."""
        print("🚀 EMPLOYEE TURNOVER PREDICTION - INTERACTIVE DEMO")
        print("=" * 65)
        
        # Data preparation
        self.load_and_prepare_data()
        
        # Run demonstrations
        print("\n📊 Running Decision Tree Demo...")
        if self.is_jupyter_environment():
            self.interactive_decision_tree()
        else:
            self.static_decision_tree_demo()
        
        print("\n🌲 Running Random Forest Demo...")
        if self.is_jupyter_environment():
            self.interactive_random_forest()
        else:
            self.static_random_forest_demo()
        
        print("\n⚖️  Running Model Comparison...")
        self.model_comparison_demo()
        
        print("\n🔍 Running Feature Importance Analysis...")
        self.feature_importance_explorer()
        
        print("\n" + "=" * 65)
        print("🎉 INTERACTIVE DEMO COMPLETED!")
        print("=" * 65)
        print("Summary:")
        print("- Data loaded and preprocessed")
        print("- Decision Tree models trained and evaluated")
        print("- Random Forest models trained and evaluated")
        print("- Model performance compared")
        print("- Feature importance analyzed")
        print("- Visualizations generated")
        
        try:
            plt.show()
        except:
            pass
    
    def is_jupyter_environment(self):
        """Check if running in Jupyter environment."""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False


def menu_driven_demo():
    """Menu-driven interface for the interactive demo."""
    demo = InteractiveDemo()
    
    while True:
        print("\n" + "=" * 50)
        print("EMPLOYEE TURNOVER PREDICTION - INTERACTIVE MENU")
        print("=" * 50)
        print("1. Load and Prepare Data")
        print("2. Interactive Decision Tree")
        print("3. Interactive Random Forest")
        print("4. Model Comparison")
        print("5. Feature Importance Explorer")
        print("6. Run Full Demo")
        print("7. Exit")
        print("=" * 50)
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                demo.load_and_prepare_data()
            elif choice == '2':
                demo.interactive_decision_tree()
            elif choice == '3':
                demo.interactive_random_forest()
            elif choice == '4':
                demo.model_comparison_demo()
            elif choice == '5':
                demo.feature_importance_explorer()
            elif choice == '6':
                demo.run_full_demo()
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎯 Starting Interactive Demo...")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            # Run full demo
            demo = InteractiveDemo()
            demo.run_full_demo()
        elif sys.argv[1] == "--menu":
            # Run menu-driven demo
            menu_driven_demo()
        else:
            print("Usage: python interactive_demo.py [--full|--menu]")
    else:
        # Default: run menu-driven demo
        menu_driven_demo()