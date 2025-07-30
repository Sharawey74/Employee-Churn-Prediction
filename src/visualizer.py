"""
Visualization module for Employee Turnover Prediction.

This module handles:
- Task 2: Exploratory Data Analysis
- Task 4: Visualize Class Imbalance  
- Task 9: Feature Importance Plots and Evaluation Metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ClassBalance
from yellowbrick.model_selection import FeatureImportances
import logging
from typing import List, Dict, Tuple, Optional, Any
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import TARGET_COLUMN, COLORS
from utils.helpers import setup_logging, save_figure, calculate_class_balance, get_feature_importance_df

class DataVisualizer:
    """
    Data visualization class for employee turnover analysis.
    
    Implements Tasks 2, 4, and 9 visualization requirements.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.logger = setup_logging()
        self.figsize = figsize
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn')
            self.logger.warning(f"Style '{style}' not found, using 'seaborn'")
        
        # Set color palette
        sns.set_palette("husl")
        
    def exploratory_data_analysis(self, data: pd.DataFrame, target_col: str = TARGET_COLUMN, 
                                save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Perform exploratory data analysis (Task 2).
        
        Explores the data visually by graphing various features against the target.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary of matplotlib figures
        """
        self.logger.info("Starting exploratory data analysis...")
        
        figures = {}
        
        # 1. Target distribution
        fig, ax = plt.subplots(figsize=self.figsize)
        data[target_col].value_counts().plot(kind='bar', ax=ax, color=COLORS['primary'])
        ax.set_title('Target Variable Distribution (Quit)')
        ax.set_xlabel('Quit (0=No, 1=Yes)')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        figures['target_distribution'] = fig
        if save_plots:
            save_figure(fig, 'target_distribution.png')
        
        # 2. Numerical features distribution by target
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if numerical_cols:
            n_cols = 2
            n_rows = (len(numerical_cols) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    for target_val in data[target_col].unique():
                        subset = data[data[target_col] == target_val][col]
                        axes[i].hist(subset, alpha=0.7, label=f'Quit={target_val}', bins=20)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            figures['numerical_distributions'] = fig
            if save_plots:
                save_figure(fig, 'numerical_distributions.png')
        
        # 3. Categorical features vs target
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            fig, axes = plt.subplots(1, len(categorical_cols), figsize=(6*len(categorical_cols), 6))
            if len(categorical_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(categorical_cols):
                crosstab = pd.crosstab(data[col], data[target_col], normalize='index')
                crosstab.plot(kind='bar', ax=axes[i], color=[COLORS['primary'], COLORS['secondary']])
                axes[i].set_title(f'{col} vs Quit Rate')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Proportion')
                axes[i].legend(['Stay', 'Quit'])
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            figures['categorical_analysis'] = fig
            if save_plots:
                save_figure(fig, 'categorical_analysis.png')
        
        # 4. Correlation heatmap
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=self.figsize)
            corr_matrix = data[numerical_cols + [target_col]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            plt.tight_layout()
            figures['correlation_matrix'] = fig
            if save_plots:
                save_figure(fig, 'correlation_matrix.png')
        
        # 5. Box plots for numerical features
        if numerical_cols:
            fig, axes = plt.subplots(1, min(len(numerical_cols), 3), figsize=(15, 5))
            if len(numerical_cols) == 1:
                axes = [axes]
            elif len(numerical_cols) == 2:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            
            for i, col in enumerate(numerical_cols[:3]):  # Show only first 3 for space
                sns.boxplot(data=data, x=target_col, y=col, ax=axes[i])
                axes[i].set_title(f'Box Plot: {col} by Quit Status')
                axes[i].set_xlabel('Quit (0=No, 1=Yes)')
            
            plt.tight_layout()
            figures['box_plots'] = fig
            if save_plots:
                save_figure(fig, 'box_plots.png')
        
        self.logger.info(f"EDA complete. Generated {len(figures)} visualizations.")
        return figures
    
    def visualize_class_imbalance(self, y: pd.Series, save_plot: bool = True) -> Tuple[plt.Figure, Dict]:
        """
        Visualize class imbalance (Task 4).
        
        Uses Yellowbrick's Class Balance visualizer and creates a frequency plot.
        
        Args:
            y: Target variable
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (matplotlib figure, class balance statistics)
        """
        self.logger.info("Visualizing class imbalance...")
        
        # Calculate class balance statistics
        balance_stats = calculate_class_balance(y)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Frequency plot
        counts = y.value_counts().sort_index()
        bars = ax1.bar(counts.index, counts.values, color=[COLORS['primary'], COLORS['secondary']])
        ax1.set_title('Class Frequency Distribution')
        ax1.set_xlabel('Class (0=Stay, 1=Quit)')
        ax1.set_ylabel('Frequency')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}\n({count/len(y)*100:.1f}%)',
                    ha='center', va='bottom')
        
        # 2. Yellowbrick Class Balance visualizer
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier()
        
        # Create class balance visualizer
        visualizer = ClassBalance(dummy_model, ax=ax2)
        visualizer.fit(np.zeros((len(y), 1)), y)  # Dummy X data
        visualizer.finalize()
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'class_imbalance.png')
        
        self.logger.info(f"Class balance - Ratio: {balance_stats['balance_ratio']:.3f}")
        return fig, balance_stats
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              title: str = "Feature Importance", save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance (Part of Task 9).
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting feature importance...")
        
        # Get feature importance DataFrame
        importance_df = get_feature_importance_df(model, feature_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['importance'], color=COLORS['primary'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()  # Features with highest importance on top
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            width = bar.get_width()
            ax.text(width + max(importance_df['importance'])*0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, f'{title.lower().replace(" ", "_")}.png')
        
        return fig
    
    def plot_yellowbrick_feature_importance(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                          save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance using Yellowbrick's FeatureImportances visualizer.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create Yellowbrick feature importance visualizer
        visualizer = FeatureImportances(model, ax=ax)
        visualizer.fit(X, y)
        visualizer.finalize()
        
        if save_plot:
            save_figure(fig, 'yellowbrick_feature_importance.png')
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_plot: bool = True) -> plt.Figure:
        """
        Plot comparison of model performance metrics.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting model comparison...")
        
        # Extract metrics
        models = list(results.keys())
        metrics = ['train_accuracy', 'val_accuracy']
        
        # Create data for plotting
        train_scores = [results[model].get('train_accuracy', 0) for model in models]
        val_scores = [results[model].get('val_accuracy', 0) for model in models]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Training Accuracy', 
                      color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, val_scores, width, label='Validation Accuracy', 
                      color=COLORS['secondary'], alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'model_comparison.png')
        
        return fig
    
    def create_summary_dashboard(self, data: pd.DataFrame, model_results: Optional[Dict] = None,
                               save_plot: bool = True) -> plt.Figure:
        """
        Create a summary dashboard with key visualizations.
        
        Args:
            data: Input DataFrame
            model_results: Optional model results for comparison
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        ax1 = plt.subplot(3, 3, 1)
        data[TARGET_COLUMN].value_counts().plot(kind='bar', ax=ax1, color=COLORS['primary'])
        ax1.set_title('Target Distribution')
        ax1.tick_params(axis='x', rotation=0)
        
        # 2. Correlation heatmap (top features)
        ax2 = plt.subplot(3, 3, 2)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2, cbar=False)
            ax2.set_title('Feature Correlations')
        
        # 3. Class imbalance
        ax3 = plt.subplot(3, 3, 3)
        balance_stats = calculate_class_balance(data[TARGET_COLUMN])
        counts = data[TARGET_COLUMN].value_counts()
        ax3.pie(counts.values, labels=['Stay', 'Quit'], autopct='%1.1f%%', 
               colors=[COLORS['primary'], COLORS['secondary']])
        ax3.set_title('Class Distribution')
        
        # 4-6. Top numerical features by target
        top_numerical = numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
        for i, col in enumerate(top_numerical):
            ax = plt.subplot(3, 3, 4 + i)
            for target_val in data[TARGET_COLUMN].unique():
                subset = data[data[TARGET_COLUMN] == target_val][col]
                ax.hist(subset, alpha=0.7, label=f'Quit={target_val}', bins=15)
            ax.set_title(f'{col}')
            ax.legend()
        
        # 7-8. Categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        for i, col in enumerate(categorical_cols[:2]):
            ax = plt.subplot(3, 3, 7 + i)
            crosstab = pd.crosstab(data[col], data[TARGET_COLUMN], normalize='index')
            crosstab.plot(kind='bar', ax=ax, color=[COLORS['primary'], COLORS['secondary']])
            ax.set_title(f'{col} vs Quit Rate')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(['Stay', 'Quit'])
        
        # 9. Model comparison (if provided)
        if model_results:
            ax9 = plt.subplot(3, 3, 9)
            models = list(model_results.keys())
            val_scores = [model_results[model].get('val_accuracy', 0) for model in models]
            bars = ax9.bar(models, val_scores, color=COLORS['info'])
            ax9.set_title('Model Performance')
            ax9.set_ylabel('Validation Accuracy')
            ax9.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, val_scores):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            save_figure(fig, 'summary_dashboard.png')
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("Data Visualizer Example")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'satisfaction_level': np.random.uniform(0.1, 1.0, 100),
        'last_evaluation': np.random.uniform(0.3, 1.0, 100),
        'number_project': np.random.randint(2, 8, 100),
        'department': np.random.choice(['sales', 'IT', 'hr'], 100),
        'salary': np.random.choice(['low', 'medium', 'high'], 100),
        'quit': np.random.binomial(1, 0.3, 100)
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Create visualizer
    visualizer = DataVisualizer()
    
    # EDA
    eda_figures = visualizer.exploratory_data_analysis(sample_data, save_plots=False)
    print(f"Generated {len(eda_figures)} EDA visualizations")
    
    # Class imbalance
    class_fig, balance_stats = visualizer.visualize_class_imbalance(sample_data['quit'], save_plot=False)
    print(f"Class balance ratio: {balance_stats['balance_ratio']:.3f}")
    
    plt.show()