"""
Advanced Plotting Utilities for Customer Churn Prediction
Specialized plotting functions for machine learning analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, validation_curve
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


def plot_distribution_comparison(df: pd.DataFrame, 
                               column: str, 
                               target_column: str,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot distribution comparison between target classes
    
    Args:
        df: Input DataFrame
        column: Column to plot distribution for
        target_column: Target column name
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    for target_value in df[target_column].unique():
        subset = df[df[target_column] == target_value][column]
        axes[0].hist(subset, alpha=0.7, label=f'{target_column}={target_value}', bins=30)
    
    axes[0].set_title(f'Distribution of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    df.boxplot(column=column, by=target_column, ax=axes[1])
    axes[1].set_title(f'Box Plot of {column} by {target_column}')
    axes[1].set_xlabel(target_column)
    axes[1].set_ylabel(column)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, 
                           figsize: Tuple[int, int] = (12, 10),
                           method: str = 'pearson',
                           annot: bool = True) -> plt.Figure:
    """
    Plot correlation heatmap
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        method: Correlation method
        annot: Whether to annotate cells
        
    Returns:
        Matplotlib figure
    """
    # Calculate correlation matrix
    correlation_matrix = df.select_dtypes(include=[np.number]).corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=annot, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title(f'Correlation Heatmap ({method.title()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_feature_importance_comparison(feature_importance_dict: Dict[str, pd.DataFrame],
                                     top_n: int = 15,
                                     figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Compare feature importance across multiple models
    
    Args:
        feature_importance_dict: Dictionary of {model_name: feature_importance_df}
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine all feature importance data
    all_features = set()
    for df in feature_importance_dict.values():
        all_features.update(df['feature'].head(top_n))
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, importance_df in feature_importance_dict.items():
        model_features = importance_df.set_index('feature')['importance'].head(top_n)
        for feature in all_features:
            importance = model_features.get(feature, 0)
            comparison_data.append({
                'model': model_name,
                'feature': feature,
                'importance': importance
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create pivot table for heatmap
    pivot_df = comparison_df.pivot(index='feature', columns='model', values='importance')
    pivot_df = pivot_df.fillna(0)
    
    # Sort by average importance
    pivot_df['avg_importance'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_importance', ascending=True).drop('avg_importance', axis=1)
    
    # Create heatmap
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Feature Importance'},
                ax=ax)
    
    ax.set_title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_learning_curves(estimator, X, y, 
                        cv: int = 5,
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot learning curves to analyze model performance vs training set size
    
    Args:
        estimator: ML model
        X: Features
        y: Target
        cv: Cross-validation folds
        train_sizes: Training set sizes to evaluate
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, 
        scoring='roc_auc', n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('ROC AUC Score')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_validation_curves(estimator, X, y, 
                         param_name: str, 
                         param_range: List,
                         cv: int = 5,
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot validation curves to analyze model performance vs hyperparameter values
    
    Args:
        estimator: ML model
        X: Features
        y: Target
        param_name: Parameter name to vary
        param_range: Range of parameter values
        cv: Cross-validation folds
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot validation curves
    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('ROC AUC Score')
    ax.set_title(f'Validation Curves for {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Handle log scale for certain parameters
    if 'C' in param_name.lower() or 'alpha' in param_name.lower():
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot residual analysis for regression models
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_class_distribution(df: pd.DataFrame, 
                          target_column: str,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot class distribution with detailed statistics
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    value_counts = df[target_column].value_counts()
    colors = ['skyblue', 'lightcoral'] if len(value_counts) == 2 else None
    
    axes[0].bar(value_counts.index.astype(str), value_counts.values, color=colors)
    axes[0].set_title(f'Class Distribution - {target_column}')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Add count labels on bars
    for i, count in enumerate(value_counts.values):
        axes[0].text(i, count + count * 0.01, str(count), ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[1].set_title(f'Class Proportion - {target_column}')
    
    # Add statistics text
    total = len(df)
    majority_class = value_counts.index[0]
    minority_class = value_counts.index[1] if len(value_counts) > 1 else None
    
    stats_text = f"Total Samples: {total}\n"
    stats_text += f"Majority Class ({majority_class}): {value_counts.iloc[0]} ({value_counts.iloc[0]/total*100:.1f}%)\n"
    if minority_class is not None:
        stats_text += f"Minority Class ({minority_class}): {value_counts.iloc[1]} ({value_counts.iloc[1]/total*100:.1f}%)\n"
        stats_text += f"Imbalance Ratio: {value_counts.iloc[0]/value_counts.iloc[1]:.2f}:1"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_feature_distribution_by_target(df: pd.DataFrame,
                                       features: List[str],
                                       target_column: str,
                                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot distribution of multiple features by target class
    
    Args:
        df: Input DataFrame
        features: List of feature names
        target_column: Target column name
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Determine if feature is numerical or categorical
        if df[feature].dtype in ['int64', 'float64']:
            # Numerical feature - use histogram
            for target_value in df[target_column].unique():
                subset = df[df[target_column] == target_value][feature]
                ax.hist(subset, alpha=0.7, label=f'{target_column}={target_value}', bins=20)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
        else:
            # Categorical feature - use count plot
            df_plot = df.groupby([feature, target_column]).size().unstack(fill_value=0)
            df_plot.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(f'{feature} by {target_column}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_missing_values_heatmap(df: pd.DataFrame, 
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot heatmap of missing values
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate missing values
    missing_data = df.isnull()
    
    if missing_data.sum().sum() == 0:
        ax.text(0.5, 0.5, 'No Missing Values Found', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('Missing Values Analysis')
        return fig
    
    # Create heatmap
    sns.heatmap(missing_data, 
                cbar=True, 
                cmap='viridis',
                yticklabels=False,
                ax=ax)
    
    ax.set_title('Missing Values Heatmap (Yellow = Missing)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features')
    
    # Add missing percentage information
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_info = missing_percentages[missing_percentages > 0].sort_values(ascending=False)
    
    if len(missing_info) > 0:
        info_text = "Missing Value Percentages:\n"
        for col, pct in missing_info.head(10).items():
            info_text += f"{col}: {pct:.1f}%\n"
        
        fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_outlier_detection(df: pd.DataFrame, 
                         column: str,
                         method: str = 'iqr',
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot outlier detection visualization
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        method: Outlier detection method ('iqr', 'zscore')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        outliers = data[z_scores > 3]
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Box plot
    axes[0].boxplot(data, vert=True)
    axes[0].set_title(f'Box Plot - {column}')
    axes[0].set_ylabel(column)
    axes[0].grid(True, alpha=0.3)
    
    # Histogram with outlier boundaries
    axes[1].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
    axes[1].axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
    
    # Highlight outliers
    if len(outliers) > 0:
        axes[1].hist(outliers, bins=20, alpha=0.8, color='red', label=f'Outliers ({len(outliers)})')
    
    axes[1].set_title(f'Distribution with Outliers - {column}')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_interactive_scatter_plot(df: pd.DataFrame,
                                  x_column: str,
                                  y_column: str,
                                  color_column: str = None,
                                  size_column: str = None,
                                  title: str = None) -> go.Figure:
    """
    Create interactive scatter plot using Plotly
    
    Args:
        df: Input DataFrame
        x_column: X-axis column name
        y_column: Y-axis column name
        color_column: Column for color coding
        size_column: Column for size coding
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(df, 
                     x=x_column, 
                     y=y_column,
                     color=color_column,
                     size=size_column,
                     title=title or f'{y_column} vs {x_column}',
                     hover_data=df.select_dtypes(include=[np.number]).columns.tolist())
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=True if color_column else False
    )
    
    return fig


def create_interactive_correlation_heatmap(df: pd.DataFrame, 
                                         title: str = "Correlation Heatmap") -> go.Figure:
    """
    Create interactive correlation heatmap using Plotly
    
    Args:
        df: Input DataFrame
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        width=800,
        height=800
    )
    
    return fig


if __name__ == "__main__":
    """
    Example usage of plotting utilities
    """
    print("Plotting Utilities Module")
    print("This module provides advanced plotting functions for ML analysis")
    
    # Example data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    print(f"\nExample dataset shape: {sample_data.shape}")
    
    # Example plots
    print("\nKey plotting functions available:")
    print("- plot_distribution_comparison")
    print("- plot_correlation_heatmap") 
    print("- plot_feature_importance_comparison")
    print("- plot_learning_curves")
    print("- plot_class_distribution")
    print("- plot_missing_values_heatmap")
    print("- plot_outlier_detection")
    print("- create_interactive_scatter_plot")
    print("- create_interactive_correlation_heatmap")
