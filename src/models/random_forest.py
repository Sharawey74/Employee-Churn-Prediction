"""
Random Forest model for Employee Turnover Prediction.

This module handles Task 8: Build a Random Forest Classifier with Interactive Controls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from typing import Dict, Tuple, Optional, Any, List
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import RANDOM_FOREST_PARAMS, RANDOM_STATE
from utils.constants import RANDOM_FOREST_PARAMS_RANGE
from utils.helpers import setup_logging, save_figure, print_model_summary

class RandomForestModel:
    """
    Random Forest model class for employee turnover prediction.
    
    Implements Task 8: Build a Random Forest Classifier with Interactive Controls.
    Uses the interact function to automatically create UI controls for function arguments.
    To overcome the variance problem associated with decision trees, builds and trains a 
    random forests classifier with scikit-learn.
    Calculates the training and validation accuracies.
    Displays a fitted tree graphically.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the Random Forest model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.logger = setup_logging()
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.class_names = ['Stay', 'Quit']
        self.training_history = {}
        
    def create_model(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                    min_samples_split: int = 2, min_samples_leaf: int = 1,
                    criterion: str = 'gini') -> RandomForestClassifier:
        """
        Create a Random Forest classifier with specified parameters.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            criterion: The function to measure the quality of a split
            
        Returns:
            RandomForestClassifier instance
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.logger.info(f"Created Random Forest with parameters: n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
                        f"min_samples_leaf={min_samples_leaf}, criterion={criterion}")
        
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   n_estimators: int = 100, max_depth: Optional[int] = None,
                   min_samples_split: int = 2, min_samples_leaf: int = 1,
                   criterion: str = 'gini') -> Dict[str, float]:
        """
        Train the Random Forest model and calculate accuracies.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            criterion: The function to measure the quality of a split
            
        Returns:
            Dictionary with training and validation accuracies
        """
        # Create model with specified parameters
        self.create_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train the model
        self.logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate accuracies
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        # Store training history
        self.training_history = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'overfitting': train_accuracy - val_accuracy,
            'parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            }
        }
        
        self.logger.info(f"Training complete. Train accuracy: {train_accuracy:.4f}, "
                        f"Validation accuracy: {val_accuracy:.4f}")
        
        return self.training_history
    
    def get_predictions(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Tuple of (predictions, prediction probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def plot_sample_tree(self, tree_index: int = 0, max_depth_display: int = 3,
                        figsize: Tuple[int, int] = (20, 10), save_plot: bool = True) -> plt.Figure:
        """
        Display a sample tree from the random forest graphically.
        
        Args:
            tree_index: Index of the tree to display
            max_depth_display: Maximum depth to display (for readability)
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        if tree_index >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_index} out of range. "
                           f"Model has {len(self.model.estimators_)} trees.")
        
        self.logger.info(f"Creating visualization for tree {tree_index}...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the specific tree
        tree = self.model.estimators_[tree_index]
        
        plot_tree(
            tree,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth_display,
            ax=ax
        )
        
        ax.set_title(f"Random Forest - Tree {tree_index} "
                    f"(Max Depth Display: {max_depth_display})")
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, f'random_forest_tree_{tree_index}.png')
        
        return fig
    
    def plot_feature_importance_comparison(self, n_trees: int = 5, 
                                         figsize: Tuple[int, int] = (15, 10),
                                         save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance comparison across multiple trees.
        
        Args:
            n_trees: Number of trees to compare
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        n_trees = min(n_trees, len(self.model.estimators_))
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot individual tree importances
        tree_importances = []
        for i in range(n_trees):
            tree = self.model.estimators_[i]
            importance = tree.feature_importances_
            tree_importances.append(importance)
            axes[0].plot(importance, alpha=0.7, label=f'Tree {i}')
        
        axes[0].set_title('Feature Importance - Individual Trees')
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Importance')
        axes[0].legend()
        axes[0].set_xticks(range(len(self.feature_names)))
        axes[0].set_xticklabels(self.feature_names, rotation=45)
        
        # Plot ensemble (average) importance
        ensemble_importance = self.model.feature_importances_
        bars = axes[1].bar(range(len(ensemble_importance)), ensemble_importance)
        axes[1].set_title('Feature Importance - Random Forest Ensemble')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Importance')
        axes[1].set_xticks(range(len(self.feature_names)))
        axes[1].set_xticklabels(self.feature_names, rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, ensemble_importance):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'random_forest_feature_importance_comparison.png')
        
        return fig
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained Random Forest.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trees in the forest.
        
        Returns:
            Dictionary with tree statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in self.model.estimators_]
        tree_leaves = [tree.tree_.n_leaves for tree in self.model.estimators_]
        
        stats = {
            'n_estimators': len(self.model.estimators_),
            'tree_depths': {
                'mean': np.mean(tree_depths),
                'std': np.std(tree_depths),
                'min': np.min(tree_depths),
                'max': np.max(tree_depths)
            },
            'tree_nodes': {
                'mean': np.mean(tree_nodes),
                'std': np.std(tree_nodes),
                'min': np.min(tree_nodes),
                'max': np.max(tree_nodes)
            },
            'tree_leaves': {
                'mean': np.mean(tree_leaves),
                'std': np.std(tree_leaves),
                'min': np.min(tree_leaves),
                'max': np.max(tree_leaves)
            }
        }
        
        return stats
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        evaluation = {
            'test_accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions, 
                                                          target_names=self.class_names),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return evaluation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        tree_stats = self.get_tree_statistics()
        
        info = {
            "status": "Fitted",
            "model_type": "RandomForestClassifier",
            "parameters": self.model.get_params(),
            "training_history": self.training_history,
            "tree_statistics": tree_stats,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "classes": self.class_names
        }
        
        return info


def build_random_forest_interactive(X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series,
                                   n_estimators: int = 100, max_depth: int = 10,
                                   min_samples_split: int = 2, min_samples_leaf: int = 1,
                                   criterion: str = 'gini') -> Tuple[RandomForestModel, Dict]:
    """
    Interactive function to build and train a Random Forest classifier.
    
    This function is designed to work with ipywidgets.interact for interactive controls.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split an internal node
        min_samples_leaf: Minimum samples required to be at a leaf node
        criterion: The function to measure the quality of a split
        
    Returns:
        Tuple of (trained model, results dictionary)
    """
    # Create and train model
    rf_model = RandomForestModel()
    results = rf_model.train_model(
        X_train, y_train, X_val, y_val,
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )
    
    # Get tree statistics
    tree_stats = rf_model.get_tree_statistics()
    
    # Print results
    print("Random Forest Results:")
    print("=" * 50)
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print(f"Overfitting: {results['overfitting']:.4f}")
    print(f"Number of Trees: {n_estimators}")
    print(f"Average Tree Depth: {tree_stats['tree_depths']['mean']:.1f} "
          f"(±{tree_stats['tree_depths']['std']:.1f})")
    print(f"Average Nodes per Tree: {tree_stats['tree_nodes']['mean']:.1f}")
    print(f"Average Leaves per Tree: {tree_stats['tree_leaves']['mean']:.1f}")
    
    # Display sample tree visualization
    try:
        rf_model.plot_sample_tree(tree_index=0, max_depth_display=3, save_plot=False)
        plt.show()
    except Exception as e:
        print(f"Could not display tree visualization: {e}")
    
    # Display feature importance
    try:
        importance_df = rf_model.get_feature_importance()
        print(f"\nTop 5 Most Important Features:")
        print(importance_df.head())
    except Exception as e:
        print(f"Could not display feature importance: {e}")
    
    return rf_model, results


def create_interactive_random_forest_widget(X_train: pd.DataFrame, y_train: pd.Series,
                                           X_val: pd.DataFrame, y_val: pd.Series):
    """
    Create interactive widget for Random Forest parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    """
    try:
        from ipywidgets import interact, IntSlider, Dropdown, fixed
        
        # Create interactive widget
        interact(
            build_random_forest_interactive,
            X_train=fixed(X_train),
            y_train=fixed(y_train),
            X_val=fixed(X_val),
            y_val=fixed(y_val),
            n_estimators=IntSlider(
                value=100,
                min=10,
                max=200,
                step=10,
                description='N Trees:'
            ),
            max_depth=IntSlider(
                value=10,
                min=1,
                max=20,
                step=1,
                description='Max Depth:'
            ),
            min_samples_split=IntSlider(
                value=2,
                min=2,
                max=50,
                step=1,
                description='Min Split:'
            ),
            min_samples_leaf=IntSlider(
                value=1,
                min=1,
                max=50,
                step=1,
                description='Min Leaf:'
            ),
            criterion=Dropdown(
                options=['gini', 'entropy'],
                value='gini',
                description='Criterion:'
            )
        )
        
    except ImportError:
        print("ipywidgets not available. Using static version.")
        model, results = build_random_forest_interactive(
            X_train, y_train, X_val, y_val
        )
        return model, results


if __name__ == "__main__":
    # Example usage
    print("Random Forest Model Example")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'satisfaction_level': np.random.uniform(0.1, 1.0, n_samples),
        'last_evaluation': np.random.uniform(0.3, 1.0, n_samples),
        'number_project': np.random.randint(2, 8, n_samples),
        'average_montly_hours': np.random.randint(120, 320, n_samples)
    })
    
    # Create target based on realistic patterns
    y = ((X['satisfaction_level'] < 0.5) & (X['average_montly_hours'] > 250)).astype(int)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Create and train model
    rf_model = RandomForestModel()
    results = rf_model.train_model(X_train, y_train, X_val, y_val, 
                                  n_estimators=50, max_depth=5)
    
    print(f"\nModel Results:")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    
    # Get feature importance
    importance_df = rf_model.get_feature_importance()
    print(f"\nFeature Importance:")
    print(importance_df)
    
    # Get tree statistics
    tree_stats = rf_model.get_tree_statistics()
    print(f"\nTree Statistics:")
    print(f"Average depth: {tree_stats['tree_depths']['mean']:.1f}")
    print(f"Average nodes: {tree_stats['tree_nodes']['mean']:.1f}")