"""
Model evaluation module for Employee Turnover Prediction.

This module handles Task 9: Feature Importance Plots and Evaluation Metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import validation_curve, learning_curve
import logging
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.helpers import setup_logging, save_figure, get_feature_importance_df
from utils.constants import COLORS

class ModelEvaluator:
    """
    Model evaluation class for comprehensive model assessment.
    
    Implements Task 9: Feature Importance Plots and Evaluation Metrics.
    Many model forms describe the underlying impact of features relative to each other.
    Decision Tree models and Random Forest in scikit-learn have feature_importances_ 
    attribute when fitted. Utilizes this attribute to rank and plot the features.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the model evaluator.
        
        Args:
            figsize: Default figure size for plots
        """
        self.logger = setup_logging()
        self.figsize = figsize
        self.class_names = ['Stay', 'Quit']
        
    def evaluate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_class_0': precision_score(y_true, y_pred, pos_label=0),
            'recall_class_0': recall_score(y_true, y_pred, pos_label=0),
            'f1_score_class_0': f1_score(y_true, y_pred, pos_label=0),
            'precision_class_1': precision_score(y_true, y_pred, pos_label=1),
            'recall_class_1': recall_score(y_true, y_pred, pos_label=1),
            'f1_score_class_1': f1_score(y_true, y_pred, pos_label=1)
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                # Multi-class probabilities, use positive class
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            metrics['auc_roc'] = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba_pos)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True, save_plot: bool = True) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'confusion_matrix.png')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_plot: bool = True) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if y_pred_proba.ndim == 2:
            y_proba_pos = y_pred_proba[:, 1]
        else:
            y_proba_pos = y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(fpr, tpr, color=COLORS['primary'], lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=COLORS['warning'], lw=2, linestyle='--',
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'roc_curve.png')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_plot: bool = True) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if y_pred_proba.ndim == 2:
            y_proba_pos = y_pred_proba[:, 1]
        else:
            y_proba_pos = y_pred_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        avg_precision = average_precision_score(y_true, y_proba_pos)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(recall, precision, color=COLORS['secondary'], lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color=COLORS['warning'], linestyle='--',
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'precision_recall_curve.png')
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              top_n: int = 10, save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance from model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        importance_df = get_feature_importance_df(model, feature_names)
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['importance'], color=COLORS['primary'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            width = bar.get_width()
            ax.text(width + max(top_features['importance'])*0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'feature_importance.png')
        
        return fig
    
    def plot_feature_importance_comparison(self, models: Dict[str, Any], 
                                         feature_names: List[str],
                                         top_n: int = 10, save_plot: bool = True) -> plt.Figure:
        """
        Compare feature importance across multiple models.
        
        Args:
            models: Dictionary of model name -> trained model
            feature_names: List of feature names
            top_n: Number of top features to display
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get feature importance for each model
        importance_data = {}
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = get_feature_importance_df(model, feature_names)
                importance_data[model_name] = importance_df.set_index('feature')['importance']
        
        if not importance_data:
            raise ValueError("No models with feature_importances_ attribute found")
        
        # Combine into single DataFrame
        comparison_df = pd.DataFrame(importance_data)
        
        # Get top features based on average importance
        comparison_df['avg_importance'] = comparison_df.mean(axis=1)
        top_features = comparison_df.nlargest(top_n, 'avg_importance')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        x = np.arange(len(top_features))
        width = 0.8 / len(models)
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                 COLORS['warning'], COLORS['info']]
        
        for i, model_name in enumerate(models.keys()):
            if model_name in top_features.columns:
                bars = ax.bar(x + i * width, top_features[model_name], width,
                            label=model_name, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Importance')
        ax.set_title(f'Feature Importance Comparison - Top {top_n} Features')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(top_features.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'feature_importance_comparison.png')
        
        return fig
    
    def plot_learning_curves(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5, train_sizes: Optional[np.ndarray] = None,
                           save_plot: bool = True) -> plt.Figure:
        """
        Plot learning curves to analyze bias/variance.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of cross-validation folds
            train_sizes: Training set sizes to use
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(train_sizes, train_mean, 'o-', color=COLORS['primary'],
               label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color=COLORS['primary'])
        
        ax.plot(train_sizes, val_mean, 'o-', color=COLORS['secondary'],
               label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color=COLORS['secondary'])
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'learning_curves.png')
        
        return fig
    
    def plot_validation_curves(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             param_name: str, param_range: np.ndarray,
                             cv: int = 5, save_plot: bool = True) -> plt.Figure:
        """
        Plot validation curves for hyperparameter tuning.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv: Number of cross-validation folds
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(param_range, train_mean, 'o-', color=COLORS['primary'],
               label='Training Score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color=COLORS['primary'])
        
        ax.plot(param_range, val_mean, 'o-', color=COLORS['secondary'],
               label='Validation Score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color=COLORS['secondary'])
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'Validation Curves - {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, f'validation_curves_{param_name}.png')
        
        return fig
    
    def create_comprehensive_report(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                  feature_names: List[str], model_name: str = "Model",
                                  save_plot: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            model_name: Name of the model
            save_plot: Whether to save plots
            
        Returns:
            Dictionary with evaluation results and figures
        """
        self.logger.info(f"Creating comprehensive evaluation report for {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self.evaluate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Create plots
        figures = {}
        
        # Confusion Matrix
        figures['confusion_matrix'] = self.plot_confusion_matrix(
            y_test, y_pred, save_plot=save_plot
        )
        
        # ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            figures['roc_curve'] = self.plot_roc_curve(
                y_test, y_pred_proba, save_plot=save_plot
            )
            figures['precision_recall_curve'] = self.plot_precision_recall_curve(
                y_test, y_pred_proba, save_plot=save_plot
            )
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            figures['feature_importance'] = self.plot_feature_importance(
                model, feature_names, save_plot=save_plot
            )
        
        # Classification Report
        class_report = classification_report(y_test, y_pred, target_names=self.class_names)
        
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'figures': figures,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.logger.info(f"Evaluation complete for {model_name}. "
                        f"Test accuracy: {metrics['accuracy']:.4f}")
        
        return report
    
    def compare_models(self, model_results: Dict[str, Dict], save_plot: bool = True) -> plt.Figure:
        """
        Create a comparison visualization of multiple models.
        
        Args:
            model_results: Dictionary with model names as keys and results as values
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract metrics for each model
        model_names = list(model_results.keys())
        metrics_data = {metric: [] for metric in metrics_to_plot}
        
        for model_name in model_names:
            model_metrics = model_results[model_name]['metrics']
            for metric in metrics_to_plot:
                metrics_data[metric].append(model_metrics.get(metric, 0))
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                 COLORS['warning']]
        
        for i, metric in enumerate(metrics_to_plot):
            bars = axes[i].bar(model_names, metrics_data[metric], 
                             color=colors[i % len(colors)], alpha=0.8)
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data[metric]):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'model_comparison_metrics.png')
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("Model Evaluator Example")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred = np.random.binomial(1, 0.35, n_samples)
    y_pred_proba = np.random.uniform(0, 1, n_samples)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.evaluate_classification_metrics(y_true, y_pred, y_pred_proba)
    
    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create plots
    conf_matrix_fig = evaluator.plot_confusion_matrix(y_true, y_pred, save_plot=False)
    roc_fig = evaluator.plot_roc_curve(y_true, y_pred_proba, save_plot=False)
    
    plt.show()