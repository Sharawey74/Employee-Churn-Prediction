"""
Model Evaluation Module for Customer Churn Prediction
Provides comprehensive model evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, log_loss, matthews_corrcoef
)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
from pathlib import Path
import json

from .config import VIZ_CONFIG, RESULTS_DIR, EVALUATION_METRICS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison tool
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ModelEvaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or VIZ_CONFIG
        self.evaluation_results = {}
        self.comparison_results = {}
        
        # Ensure results directory exists
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "figures").mkdir(exist_ok=True)
        (RESULTS_DIR / "metrics").mkdir(exist_ok=True)
        (RESULTS_DIR / "reports").mkdir(exist_ok=True)
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_single_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                             X_train: np.ndarray = None, y_train: np.ndarray = None,
                             model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional, for overfitting analysis)
            y_train: Training targets (optional, for overfitting analysis)
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            logger.warning(f"Model {model_name} does not support probability prediction")
            y_pred_proba = None
        
        # Calculate basic metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'average_precision': average_precision_score(y_test, y_pred_proba),
                'log_loss': log_loss(y_test, y_pred_proba)
            })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Training vs Test comparison (if training data provided)
        if X_train is not None and y_train is not None:
            train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            metrics['training_accuracy'] = train_accuracy
            metrics['overfitting_score'] = train_accuracy - metrics['accuracy']
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate_multiple_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                               X_train: np.ndarray = None, y_train: np.ndarray = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models and compare their performance
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional)
            y_train: Training targets (optional)
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        logger.info(f"Evaluating {len(models)} models")
        
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_single_model(
                    model, X_test, y_test, X_train, y_train, model_name
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        # Generate comparison
        self.comparison_results = self._generate_model_comparison(all_results)
        
        return all_results
    
    def _generate_model_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive model comparison
        
        Args:
            results: Dictionary of model evaluation results
            
        Returns:
            Comparison results
        """
        # Filter out failed evaluations
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful model evaluations to compare'}
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in successful_results.items():
            row = {
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'matthews_corrcoef': metrics.get('matthews_corrcoef', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by different metrics
        ranking = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'matthews_corrcoef']:
            if metric in comparison_df.columns:
                ranking[metric] = comparison_df.nlargest(len(comparison_df), metric)['model'].tolist()
        
        # Overall ranking (using F1 score as primary metric)
        overall_ranking = comparison_df.nlargest(len(comparison_df), 'f1_score')
        
        comparison_results = {
            'comparison_table': comparison_df.to_dict('records'),
            'rankings': ranking,
            'best_model': {
                'by_f1': overall_ranking.iloc[0]['model'],
                'by_accuracy': comparison_df.nlargest(1, 'accuracy').iloc[0]['model'],
                'by_roc_auc': comparison_df.nlargest(1, 'roc_auc').iloc[0]['model'] if 'roc_auc' in comparison_df.columns else None
            },
            'summary_stats': {
                'mean_accuracy': comparison_df['accuracy'].mean(),
                'std_accuracy': comparison_df['accuracy'].std(),
                'mean_f1': comparison_df['f1_score'].mean(),
                'std_f1': comparison_df['f1_score'].std()
            }
        }
        
        return comparison_results
    
    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6),
                            save_plot: bool = True) -> plt.Figure:
        """
        Plot confusion matrix for a specific model
        
        Args:
            model_name: Name of the model
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        cm_data = self.evaluation_results[model_name]['confusion_matrix']
        cm = np.array(cm_data['matrix'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Add performance metrics as text
        metrics = self.evaluation_results[model_name]
        textstr = f"Accuracy: {metrics['accuracy']:.3f}\nPrecision: {metrics['precision']:.3f}\nRecall: {metrics['recall']:.3f}\nF1-Score: {metrics['f1_score']:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"confusion_matrix_{model_name}.png"
            filepath = RESULTS_DIR / "figures" / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix plot to {filepath}")
        
        return fig
    
    def plot_roc_curves(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                       figsize: Tuple[int, int] = (10, 8), save_plot: bool = True) -> plt.Figure:
        """
        Plot ROC curves for multiple models
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test targets
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for (model_name, model), color in zip(models.items(), colors):
            try:
                # Get prediction probabilities
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{model_name} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                logger.warning(f"Could not plot ROC curve for {model_name}: {str(e)}")
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = "roc_curves_comparison.png"
            filepath = RESULTS_DIR / "figures" / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves plot to {filepath}")
        
        return fig
    
    def plot_precision_recall_curves(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                                   figsize: Tuple[int, int] = (10, 8), save_plot: bool = True) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test targets
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for (model_name, model), color in zip(models.items(), colors):
            try:
                # Get prediction probabilities
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                avg_precision = average_precision_score(y_test, y_pred_proba)
                
                # Plot Precision-Recall curve
                ax.plot(recall, precision, color=color, lw=2,
                       label=f'{model_name} (AP = {avg_precision:.3f})')
                
            except Exception as e:
                logger.warning(f"Could not plot PR curve for {model_name}: {str(e)}")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = "precision_recall_curves_comparison.png"
            filepath = RESULTS_DIR / "figures" / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Precision-Recall curves plot to {filepath}")
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str = "model",
                              top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                              save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Names of features
            model_name: Name of the model
            top_n: Number of top features to show
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not support feature importance")
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(data=feature_importance_df, x='importance', y='feature', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"feature_importance_{model_name}.png"
            filepath = RESULTS_DIR / "figures" / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {filepath}")
        
        return fig
    
    def plot_model_comparison(self, metrics: List[str] = None, figsize: Tuple[int, int] = (12, 8),
                            save_plot: bool = True) -> plt.Figure:
        """
        Plot comparison of multiple models across different metrics
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.comparison_results or 'comparison_table' not in self.comparison_results:
            raise ValueError("No comparison results available. Run evaluate_multiple_models first.")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_df = pd.DataFrame(self.comparison_results['comparison_table'])
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            raise ValueError("None of the specified metrics are available")
        
        # Create subplot
        fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Sort models by metric value
            sorted_df = comparison_df.sort_values(metric, ascending=True)
            
            # Create horizontal bar plot
            bars = ax.barh(sorted_df['model'], sorted_df[metric])
            
            # Customize plot
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_xlim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, sorted_df[metric]):
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            filename = "model_comparison.png"
            filepath = RESULTS_DIR / "figures" / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {filepath}")
        
        return fig
    
    def generate_evaluation_report(self, model_name: str = None, save_report: bool = True) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_name: Specific model name (if None, generates for all models)
            save_report: Whether to save the report to file
            
        Returns:
            Report as string
        """
        if not self.evaluation_results:
            return "No evaluation results available"
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append()
        
        if model_name:
            # Single model report
            if model_name not in self.evaluation_results:
                return f"Model {model_name} not found in evaluation results"
            
            self._add_single_model_report(report_lines, model_name)
        else:
            # All models report
            for model_name in self.evaluation_results.keys():
                self._add_single_model_report(report_lines, model_name)
                report_lines.append("-" * 80)
                report_lines.append()
            
            # Add comparison section
            if self.comparison_results:
                self._add_comparison_report(report_lines)
        
        report = "\n".join(report_lines)
        
        if save_report:
            filename = f"evaluation_report_{model_name if model_name else 'all_models'}.txt"
            filepath = RESULTS_DIR / "reports" / filename
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved evaluation report to {filepath}")
        
        return report
    
    def _add_single_model_report(self, report_lines: List[str], model_name: str) -> None:
        """Add single model evaluation to report"""
        metrics = self.evaluation_results[model_name]
        
        report_lines.append(f"MODEL: {model_name.upper()}")
        report_lines.append("-" * 40)
        report_lines.append()
        
        # Basic metrics
        report_lines.append("CLASSIFICATION METRICS:")
        report_lines.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
        report_lines.append(f"  Precision:    {metrics['precision']:.4f}")
        report_lines.append(f"  Recall:       {metrics['recall']:.4f}")
        report_lines.append(f"  F1-Score:     {metrics['f1_score']:.4f}")
        report_lines.append(f"  Matthews CC:  {metrics['matthews_corrcoef']:.4f}")
        
        if 'roc_auc' in metrics:
            report_lines.append(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
            report_lines.append(f"  Avg Precision: {metrics['average_precision']:.4f}")
            report_lines.append(f"  Log Loss:     {metrics['log_loss']:.4f}")
        
        report_lines.append()
        
        # Confusion matrix
        cm_data = metrics['confusion_matrix']
        report_lines.append("CONFUSION MATRIX:")
        report_lines.append(f"  True Negatives:  {cm_data['tn']}")
        report_lines.append(f"  False Positives: {cm_data['fp']}")
        report_lines.append(f"  False Negatives: {cm_data['fn']}")
        report_lines.append(f"  True Positives:  {cm_data['tp']}")
        report_lines.append()
        
        # Training vs Test (if available)
        if 'training_accuracy' in metrics:
            report_lines.append("OVERFITTING ANALYSIS:")
            report_lines.append(f"  Training Accuracy: {metrics['training_accuracy']:.4f}")
            report_lines.append(f"  Test Accuracy:     {metrics['accuracy']:.4f}")
            report_lines.append(f"  Overfitting Score: {metrics['overfitting_score']:.4f}")
            report_lines.append()
    
    def _add_comparison_report(self, report_lines: List[str]) -> None:
        """Add model comparison to report"""
        report_lines.append("MODEL COMPARISON")
        report_lines.append("="*40)
        report_lines.append()
        
        if 'best_model' in self.comparison_results:
            best_models = self.comparison_results['best_model']
            report_lines.append("BEST MODELS BY METRIC:")
            report_lines.append(f"  Best by F1-Score:  {best_models['by_f1']}")
            report_lines.append(f"  Best by Accuracy:  {best_models['by_accuracy']}")
            if best_models['by_roc_auc']:
                report_lines.append(f"  Best by ROC-AUC:   {best_models['by_roc_auc']}")
            report_lines.append()
        
        if 'summary_stats' in self.comparison_results:
            stats = self.comparison_results['summary_stats']
            report_lines.append("PERFORMANCE STATISTICS:")
            report_lines.append(f"  Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
            report_lines.append(f"  Mean F1-Score: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
            report_lines.append()
    
    def save_results(self, filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to JSON file
        
        Args:
            filename: Name of the file to save
        """
        results_to_save = {
            'evaluation_results': self.evaluation_results,
            'comparison_results': self.comparison_results
        }
        
        filepath = RESULTS_DIR / "metrics" / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {filepath}")
    
    def load_results(self, filename: str = "evaluation_results.json") -> None:
        """
        Load evaluation results from JSON file
        
        Args:
            filename: Name of the file to load
        """
        filepath = RESULTS_DIR / "metrics" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.evaluation_results = results.get('evaluation_results', {})
        self.comparison_results = results.get('comparison_results', {})
        
        logger.info(f"Loaded evaluation results from {filepath}")


# Utility functions
def quick_evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                  model_name: str = "model") -> Dict[str, Any]:
    """
    Quick evaluation of a single model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        
    Returns:
        Basic evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_single_model(model, X_test, y_test, model_name=model_name)


def compare_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Quick comparison of multiple models
    
    Args:
        models: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Comparison results
    """
    evaluator = ModelEvaluator()
    evaluator.evaluate_multiple_models(models, X_test, y_test)
    return evaluator.comparison_results


if __name__ == "__main__":
    """
    Example usage of ModelEvaluator
    """
    print("ModelEvaluator Module")
    print("This module provides comprehensive model evaluation capabilities")
    print("Import this module and use the ModelEvaluator class or utility functions")
    
    print("\nKey features:")
    print("- Single and multiple model evaluation")
    print("- ROC and Precision-Recall curves")
    print("- Feature importance visualization")
    print("- Comprehensive evaluation reports")
    print("- Model comparison and ranking")
