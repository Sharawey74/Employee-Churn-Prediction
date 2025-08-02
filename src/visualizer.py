"""
Advanced Visualization Module for Customer Churn Prediction
Provides comprehensive plotting capabilities for model results and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from config import VIZ_CONFIG, RESULTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

class ModelVisualizer:
    """
    Comprehensive visualization toolkit for ML model analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Initialize visualizer with configuration
        
        Args:
            figsize: Default figure size
            style: Seaborn style
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
        
        logger.info("ModelVisualizer initialized")
    
    def plot_confusion_matrices(self, y_true_list: List[np.ndarray], y_pred_list: List[np.ndarray], 
                               model_names: List[str], save_fig: bool = True) -> None:
        """
        Plot confusion matrices for multiple models
        
        Args:
            y_true_list: List of true labels
            y_pred_list: List of predicted labels
            model_names: List of model names
            save_fig: Whether to save the figure
        """
        n_models = len(model_names)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            axes[i].set_title(f'Confusion Matrix - {name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, y_true_list: List[np.ndarray], y_proba_list: List[np.ndarray],
                       model_names: List[str], save_fig: bool = True) -> None:
        """
        Plot ROC curves for multiple models
        
        Args:
            y_true_list: List of true labels
            y_proba_list: List of predicted probabilities
            model_names: List of model names
            save_fig: Whether to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        for y_true, y_proba, name in zip(y_true_list, y_proba_list, model_names):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true_list: List[np.ndarray], y_proba_list: List[np.ndarray],
                                    model_names: List[str], save_fig: bool = True) -> None:
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            y_true_list: List of true labels
            y_proba_list: List of predicted probabilities
            model_names: List of model names
            save_fig: Whether to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        for y_true, y_proba, name in zip(y_true_list, y_proba_list, model_names):
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_dict: Dict[str, np.ndarray], 
                               feature_names: List[str], top_n: int = 20, 
                               save_fig: bool = True) -> None:
        """
        Plot feature importance for multiple models
        
        Args:
            importance_dict: Dictionary of model_name -> importance_values
            feature_names: List of feature names
            top_n: Number of top features to show
            save_fig: Whether to save the figure
        """
        n_models = len(importance_dict)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, (model_name, importance) in enumerate(importance_dict.items()):
            # Create DataFrame for easier handling
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            sns.barplot(data=df_importance, y='feature', x='importance', ax=axes[i])
            axes[i].set_title(f'Top {top_n} Features - {model_name}')
            axes[i].set_xlabel('Importance')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_fig: bool = True) -> None:
        """
        Plot comprehensive model comparison
        
        Args:
            results_df: DataFrame with model results
            save_fig: Whether to save the figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, metric in enumerate(available_metrics):
            sns.barplot(data=results_df.reset_index(), x='index', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(results_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curves(self, train_scores: Dict[str, List[float]], 
                           val_scores: Dict[str, List[float]], 
                           train_sizes: List[int], save_fig: bool = True) -> None:
        """
        Plot learning curves for models
        
        Args:
            train_scores: Dictionary of model_name -> training scores
            val_scores: Dictionary of model_name -> validation scores
            train_sizes: List of training sizes
            save_fig: Whether to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        for model_name in train_scores.keys():
            plt.plot(train_sizes, train_scores[model_name], 'o-', 
                    label=f'{model_name} (Training)', alpha=0.7)
            plt.plot(train_sizes, val_scores[model_name], 's-', 
                    label=f'{model_name} (Validation)', alpha=0.7)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'learning_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_calibration_curves(self, y_true_list: List[np.ndarray], y_proba_list: List[np.ndarray],
                               model_names: List[str], save_fig: bool = True) -> None:
        """
        Plot calibration curves for models
        
        Args:
            y_true_list: List of true labels
            y_proba_list: List of predicted probabilities
            model_names: List of model names
            save_fig: Whether to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        for y_true, y_proba, name in zip(y_true_list, y_proba_list, model_names):
            prob_true, prob_pred = calibration_curve(
                y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, n_bins=10
            )
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_model_dashboard(self, results_df: pd.DataFrame, 
                                         y_true: np.ndarray, y_proba_dict: Dict[str, np.ndarray]) -> None:
        """
        Create interactive dashboard for model comparison
        
        Args:
            results_df: DataFrame with model results
            y_true: True labels
            y_proba_dict: Dictionary of model predictions
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Model Performance Comparison', 'ROC Curves', 
                          'Feature Importance', 'Prediction Distribution'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Model performance comparison
        models = results_df.index.tolist()
        fig.add_trace(
            go.Bar(x=models, y=results_df['accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        # ROC curves
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, y_proba) in enumerate(y_proba_dict.items()):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name}',
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=2
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      line=dict(dash='dash', color='black'), name='Random'),
            row=1, col=2
        )
        
        # Sample feature importance (you'd replace this with actual data)
        sample_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']
        sample_importance = [0.3, 0.25, 0.2, 0.15, 0.1]
        fig.add_trace(
            go.Bar(x=sample_importance, y=sample_features, orientation='h'),
            row=2, col=1
        )
        
        # Prediction distribution
        if list(y_proba_dict.values()):
            best_model_proba = list(y_proba_dict.values())[0]
            proba_values = best_model_proba[:, 1] if best_model_proba.ndim > 1 else best_model_proba
            fig.add_trace(
                go.Histogram(x=proba_values, nbinsx=30, name='Prediction Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Customer Churn Model Analysis Dashboard",
            showlegend=True
        )
        
        # Show and save
        fig.show()
        fig.write_html(RESULTS_DIR / "model_dashboard.html")
        logger.info("Interactive dashboard saved to model_dashboard.html")
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_proba: np.ndarray, model_name: str, 
                                save_fig: bool = True) -> None:
        """
        Plot comprehensive prediction analysis for a single model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_fig: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve - {model_name}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction Distribution
        proba_values = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        axes[1, 0].hist(proba_values[y_true == 0], bins=30, alpha=0.7, label='No Churn', color='blue')
        axes[1, 0].hist(proba_values[y_true == 1], bins=30, alpha=0.7, label='Churn', color='red')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution by True Class')
        axes[1, 0].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, proba_values)
        pr_auc = auc(recall, precision)
        axes[1, 1].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title(f'Precision-Recall Curve - {model_name}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / f'prediction_analysis_{model_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_all_plots(self, results_dict: Dict[str, Any]) -> None:
        """
        Save all generated plots
        
        Args:
            results_dict: Dictionary containing all results
        """
        logger.info("Saving all visualization plots")
        
        # Ensure results directory exists
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Save summary
        summary_text = f"""
        Customer Churn Prediction - Visualization Summary
        ================================================
        
        Generated plots:
        - confusion_matrices.png: Confusion matrices for all models
        - roc_curves.png: ROC curves comparison
        - precision_recall_curves.png: PR curves comparison
        - feature_importance.png: Feature importance plots
        - model_comparison.png: Overall model performance comparison
        - calibration_curves.png: Model calibration analysis
        - model_dashboard.html: Interactive dashboard
        
        Analysis completed on: {pd.Timestamp.now()}
        """
        
        with open(RESULTS_DIR / 'visualization_summary.txt', 'w') as f:
            f.write(summary_text)
        
        logger.info("All plots saved successfully")

# Example usage
if __name__ == "__main__":
    # This would be used with actual model results
    logger.info("ModelVisualizer example usage")
    
    # Create sample data for demonstration
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Sample predictions for multiple models
    y_pred_lr = np.random.binomial(1, 0.25, n_samples)
    y_pred_rf = np.random.binomial(1, 0.28, n_samples)
    
    y_proba_lr = np.column_stack([1-np.random.beta(2, 5, n_samples), np.random.beta(2, 5, n_samples)])
    y_proba_rf = np.column_stack([1-np.random.beta(1.8, 4.5, n_samples), np.random.beta(1.8, 4.5, n_samples)])
    
    # Initialize visualizer
    viz = ModelVisualizer()
    
    # Create sample results DataFrame
    results_df = pd.DataFrame({
        'accuracy': [0.85, 0.87],
        'precision': [0.75, 0.78],
        'recall': [0.70, 0.72],
        'f1': [0.725, 0.75],
        'roc_auc': [0.80, 0.82]
    }, index=['Logistic Regression', 'Random Forest'])
    
    # Demo plots
    viz.plot_model_comparison(results_df)
    viz.plot_roc_curves([y_true, y_true], [y_proba_lr, y_proba_rf], 
                       ['Logistic Regression', 'Random Forest'])
    
    print("Visualizer demo completed!")