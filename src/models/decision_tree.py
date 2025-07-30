"""
Decision Tree model for Employee Turnover Prediction.

This module handles Tasks 6 & 7: Build a Decision Tree Classifier with Interactive Controls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from typing import Dict, Tuple, Optional, Any
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DECISION_TREE_PARAMS, RANDOM_STATE, TEST_SIZE
from utils.constants import DECISION_TREE_PARAMS_RANGE
from utils.helpers import setup_logging, save_figure, print_model_summary

class DecisionTreeModel:
    """
    Decision Tree model class for employee turnover prediction.
    
    Implements Tasks 6 & 7: Build a Decision Tree Classifier with Interactive Controls.
    Uses the interact function to automatically create UI controls for function arguments.
    Builds and trains a decision tree classifier with scikit-learn.
    Calculates the training and validation accuracies.
    Displays the fitted decision tree graphically.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the Decision Tree model.
        
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
        
    def create_model(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                    min_samples_leaf: int = 1, criterion: str = 'gini') -> DecisionTreeClassifier:
        """
        Create a Decision Tree classifier with specified parameters.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            criterion: The function to measure the quality of a split
            
        Returns:
            DecisionTreeClassifier instance
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=self.random_state
        )
        
        self.logger.info(f"Created Decision Tree with parameters: max_depth={max_depth}, "
                        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                        f"criterion={criterion}")
        
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   max_depth: Optional[int] = None, min_samples_split: int = 2,
                   min_samples_leaf: int = 1, criterion: str = 'gini') -> Dict[str, float]:
        """
        Train the Decision Tree model and calculate accuracies.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            criterion: The function to measure the quality of a split
            
        Returns:
            Dictionary with training and validation accuracies
        """
        # Create model with specified parameters
        self.create_model(max_depth, min_samples_split, min_samples_leaf, criterion)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train the model
        self.logger.info("Training Decision Tree model...")
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
    
    def plot_tree_visualization(self, max_depth_display: int = 3, figsize: Tuple[int, int] = (20, 10),
                               save_plot: bool = True) -> plt.Figure:
        """
        Display the fitted decision tree graphically.
        
        Args:
            max_depth_display: Maximum depth to display (for readability)
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        self.logger.info("Creating decision tree visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth_display,
            ax=ax
        )
        
        ax.set_title(f"Decision Tree Visualization (Max Depth Display: {max_depth_display})")
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'decision_tree_visualization.png')
        
        return fig
    
    def get_tree_text_representation(self, max_depth: Optional[int] = None) -> str:
        """
        Get text representation of the decision tree.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            String representation of the tree
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        return export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=max_depth
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
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
        
        info = {
            "status": "Fitted",
            "model_type": "DecisionTreeClassifier",
            "parameters": self.model.get_params(),
            "training_history": self.training_history,
            "tree_depth": self.model.tree_.max_depth,
            "n_nodes": self.model.tree_.node_count,
            "n_leaves": self.model.tree_.n_leaves,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "classes": self.class_names
        }
        
        return info


def build_decision_tree_interactive(X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  max_depth: int = 5, min_samples_split: int = 2,
                                  min_samples_leaf: int = 1, criterion: str = 'gini') -> Tuple[DecisionTreeModel, Dict]:
    """
    Interactive function to build and train a Decision Tree classifier.
    
    This function is designed to work with ipywidgets.interact for interactive controls.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split an internal node
        min_samples_leaf: Minimum samples required to be at a leaf node
        criterion: The function to measure the quality of a split
        
    Returns:
        Tuple of (trained model, results dictionary)
    """
    # Create and train model
    dt_model = DecisionTreeModel()
    results = dt_model.train_model(
        X_train, y_train, X_val, y_val,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )
    
    # Print results
    print("Decision Tree Results:")
    print("=" * 50)
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print(f"Overfitting: {results['overfitting']:.4f}")
    print(f"Tree Depth: {dt_model.model.tree_.max_depth}")
    print(f"Number of Nodes: {dt_model.model.tree_.node_count}")
    print(f"Number of Leaves: {dt_model.model.tree_.n_leaves}")
    
    # Display tree visualization
    try:
        dt_model.plot_tree_visualization(max_depth_display=3, save_plot=False)
        plt.show()
    except Exception as e:
        print(f"Could not display tree visualization: {e}")
    
    return dt_model, results


def create_interactive_decision_tree_widget(X_train: pd.DataFrame, y_train: pd.Series,
                                          X_val: pd.DataFrame, y_val: pd.Series):
    """
    Create interactive widget for Decision Tree parameters.
    
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
            build_decision_tree_interactive,
            X_train=fixed(X_train),
            y_train=fixed(y_train),
            X_val=fixed(X_val),
            y_val=fixed(y_val),
            max_depth=IntSlider(
                value=5,
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
        model, results = build_decision_tree_interactive(
            X_train, y_train, X_val, y_val
        )
        return model, results


if __name__ == "__main__":
    # Example usage
    print("Decision Tree Model Example")
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
    dt_model = DecisionTreeModel()
    results = dt_model.train_model(X_train, y_train, X_val, y_val, max_depth=5)
    
    print(f"\nModel Results:")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    
    # Get feature importance
    importance_df = dt_model.get_feature_importance()
    print(f"\nFeature Importance:")
    print(importance_df)
    
    # Display tree (first few levels)
    print(f"\nTree Structure (first 3 levels):")
    tree_text = dt_model.get_tree_text_representation(max_depth=3)
    print(tree_text)