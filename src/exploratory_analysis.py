"""
Exploratory Data Analysis Module for Customer Churn Prediction
Provides comprehensive statistical analysis and visualization capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from config import VIZ_CONFIG, RESULTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

class ExploratoryAnalyzer:
    """
    Comprehensive Exploratory Data Analysis for Customer Churn Data
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'Churn'):
        """
        Initialize the analyzer with data
        
        Args:
            data: DataFrame containing the dataset
            target_column: Name of the target column
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numerical_columns = list(data.select_dtypes(include=['int64', 'float64']).columns)
        self.categorical_columns = list(data.select_dtypes(include=['object']).columns)
        
        # Remove target from categorical if present
        if target_column in self.categorical_columns:
            self.categorical_columns.remove(target_column)
        
        logger.info(f"Initialized EDA for dataset with shape {data.shape}")
        logger.info(f"Numerical columns: {len(self.numerical_columns)}")
        logger.info(f"Categorical columns: {len(self.categorical_columns)}")
    
    def generate_data_overview(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset overview
        
        Returns:
            Dictionary containing dataset statistics
        """
        logger.info("Generating data overview")
        
        overview = {
            'basic_info': {
                'shape': self.data.shape,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
                'duplicates': self.data.duplicated().sum(),
                'missing_values_total': self.data.isnull().sum().sum()
            },
            'target_analysis': {},
            'numerical_summary': {},
            'categorical_summary': {}
        }
        
        # Target variable analysis
        if self.target_column in self.data.columns:
            target_counts = self.data[self.target_column].value_counts()
            overview['target_analysis'] = {
                'distribution': target_counts.to_dict(),
                'churn_rate': (self.data[self.target_column] == 'Yes').mean(),
                'class_balance_ratio': target_counts.min() / target_counts.max()
            }
        
        # Numerical variables summary
        if self.numerical_columns:
            num_desc = self.data[self.numerical_columns].describe()
            overview['numerical_summary'] = {
                'statistics': num_desc.to_dict(),
                'skewness': self.data[self.numerical_columns].skew().to_dict(),
                'kurtosis': self.data[self.numerical_columns].kurtosis().to_dict()
            }
        
        # Categorical variables summary
        if self.categorical_columns:
            cat_summary = {}
            for col in self.categorical_columns:
                cat_summary[col] = {
                    'unique_values': self.data[col].nunique(),
                    'most_frequent': self.data[col].mode()[0],
                    'missing_count': self.data[col].isnull().sum()
                }
            overview['categorical_summary'] = cat_summary
        
        return overview
    
    def plot_target_distribution(self, save_fig: bool = True) -> None:
        """
        Plot target variable distribution
        
        Args:
            save_fig: Whether to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts = self.data[self.target_column].value_counts()
        axes[0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                   colors=['skyblue', 'lightcoral'])
        axes[0].set_title('Target Distribution (Pie Chart)')
        
        # Bar plot
        sns.countplot(data=self.data, x=self.target_column, ax=axes[1])
        axes[1].set_title('Target Distribution (Count Plot)')
        
        # Add percentage labels
        total = len(self.data)
        for p in axes[1].patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            axes[1].annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_numerical_distributions(self, save_fig: bool = True) -> None:
        """
        Plot distributions of numerical variables
        
        Args:
            save_fig: Whether to save the figure
        """
        if not self.numerical_columns:
            logger.warning("No numerical columns found")
            return
        
        n_cols = 3
        n_rows = (len(self.numerical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(self.numerical_columns):
            sns.histplot(data=self.data, x=col, hue=self.target_column, 
                        kde=True, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(self.numerical_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_categorical_distributions(self, save_fig: bool = True) -> None:
        """
        Plot distributions of categorical variables
        
        Args:
            save_fig: Whether to save the figure
        """
        if not self.categorical_columns:
            logger.warning("No categorical columns found")
            return
        
        # Select top 12 categorical variables (for readability)
        cols_to_plot = self.categorical_columns[:12]
        
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(cols_to_plot):
            # Create a crosstab for better visualization
            ct = pd.crosstab(self.data[col], self.data[self.target_column], normalize='index')
            ct.plot(kind='bar', ax=axes[i], color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{col} vs {self.target_column}')
            axes[i].legend(title=self.target_column)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, save_fig: bool = True) -> None:
        """
        Plot correlation matrix for numerical variables
        
        Args:
            save_fig: Whether to save the figure
        """
        if len(self.numerical_columns) < 2:
            logger.warning("Need at least 2 numerical columns for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numerical_columns].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_churn_by_features(self, save_fig: bool = True) -> None:
        """
        Plot churn rates by different categorical features
        
        Args:
            save_fig: Whether to save the figure
        """
        # Calculate churn rates for each categorical feature
        churn_rates = {}
        for col in self.categorical_columns[:8]:  # Top 8 features
            churn_rate = self.data.groupby(col)[self.target_column].apply(
                lambda x: (x == 'Yes').mean()
            ).sort_values(ascending=False)
            churn_rates[col] = churn_rate
        
        n_cols = 2
        n_rows = (len(churn_rates) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, (col, rates) in enumerate(churn_rates.items()):
            rates.plot(kind='bar', ax=axes[i], color='lightcoral')
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for j, v in enumerate(rates.values):
                axes[i].text(j, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(churn_rates), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(RESULTS_DIR / 'churn_by_features.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self) -> None:
        """
        Create an interactive dashboard using Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Target Distribution', 'Monthly Charges vs Tenure', 
                          'Churn by Contract Type', 'Service Usage Analysis'],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Target distribution pie chart
        target_counts = self.data[self.target_column].value_counts()
        fig.add_trace(
            go.Pie(labels=target_counts.index, values=target_counts.values,
                   name="Target Distribution"),
            row=1, col=1
        )
        
        # Scatter plot for numerical variables
        if 'MonthlyCharges' in self.data.columns and 'tenure' in self.data.columns:
            colors = ['red' if x == 'Yes' else 'blue' for x in self.data[self.target_column]]
            fig.add_trace(
                go.Scatter(x=self.data['tenure'], y=self.data['MonthlyCharges'],
                          mode='markers', marker=dict(color=colors, opacity=0.6),
                          name="Monthly Charges vs Tenure"),
                row=1, col=2
            )
        
        # Churn by contract type
        if 'Contract' in self.data.columns:
            contract_churn = self.data.groupby('Contract')[self.target_column].apply(
                lambda x: (x == 'Yes').mean()
            )
            fig.add_trace(
                go.Bar(x=contract_churn.index, y=contract_churn.values,
                       name="Churn by Contract"),
                row=2, col=1
            )
        
        # Service usage analysis
        if 'InternetService' in self.data.columns:
            internet_churn = self.data.groupby('InternetService')[self.target_column].apply(
                lambda x: (x == 'Yes').mean()
            )
            fig.add_trace(
                go.Bar(x=internet_churn.index, y=internet_churn.values,
                       name="Churn by Internet Service"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Customer Churn Analysis Dashboard")
        fig.show()
        
        # Save as HTML
        fig.write_html(RESULTS_DIR / "interactive_dashboard.html")
        logger.info("Interactive dashboard saved to interactive_dashboard.html")
    
    def statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical tests to identify significant relationships
        
        Returns:
            Dictionary containing test results
        """
        logger.info("Performing statistical tests")
        
        results = {
            'chi_square_tests': {},
            't_tests': {},
            'anova_tests': {}
        }
        
        # Encode target variable for tests
        target_encoded = (self.data[self.target_column] == 'Yes').astype(int)
        
        # Chi-square tests for categorical variables
        for col in self.categorical_columns:
            if col != self.target_column:
                contingency_table = pd.crosstab(self.data[col], self.data[self.target_column])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                results['chi_square_tests'][col] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # T-tests for numerical variables
        for col in self.numerical_columns:
            churn_yes = self.data[self.data[self.target_column] == 'Yes'][col]
            churn_no = self.data[self.data[self.target_column] == 'No'][col]
            
            t_stat, p_value = stats.ttest_ind(churn_yes, churn_no)
            results['t_tests'][col] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_churn_yes': churn_yes.mean(),
                'mean_churn_no': churn_no.mean()
            }
        
        return results
    
    def generate_insights(self) -> List[str]:
        """
        Generate key insights from the exploratory analysis
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # Target distribution insights
        churn_rate = (self.data[self.target_column] == 'Yes').mean()
        insights.append(f"Overall churn rate: {churn_rate:.1%}")
        
        if churn_rate < 0.3:
            insights.append("Dataset shows class imbalance - consider using SMOTE or other balancing techniques")
        
        # Numerical variables insights
        for col in self.numerical_columns:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                churn_mean = self.data[self.data[self.target_column] == 'Yes'][col].mean()
                no_churn_mean = self.data[self.data[self.target_column] == 'No'][col].mean()
                
                if churn_mean < no_churn_mean:
                    insights.append(f"Customers who churn have lower average {col}")
                else:
                    insights.append(f"Customers who churn have higher average {col}")
        
        # Categorical variables insights
        for col in ['Contract', 'PaymentMethod', 'InternetService']:
            if col in self.data.columns:
                churn_by_cat = self.data.groupby(col)[self.target_column].apply(
                    lambda x: (x == 'Yes').mean()
                )
                highest_churn = churn_by_cat.idxmax()
                highest_rate = churn_by_cat.max()
                insights.append(f"Highest churn rate in {col}: {highest_churn} ({highest_rate:.1%})")
        
        return insights
    
    def run_complete_eda(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete exploratory data analysis
        
        Args:
            save_plots: Whether to save generated plots
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete EDA process")
        
        # Generate overview
        overview = self.generate_data_overview()
        
        # Create visualizations
        self.plot_target_distribution(save_plots)
        self.plot_numerical_distributions(save_plots)
        self.plot_categorical_distributions(save_plots)
        self.plot_correlation_matrix(save_plots)
        self.plot_churn_by_features(save_plots)
        
        # Create interactive dashboard
        self.create_interactive_dashboard()
        
        # Perform statistical tests
        stat_results = self.statistical_tests()
        
        # Generate insights
        insights = self.generate_insights()
        
        # Compile complete results
        complete_results = {
            'data_overview': overview,
            'statistical_tests': stat_results,
            'insights': insights
        }
        
        logger.info("EDA process completed successfully")
        
        return complete_results

# Example usage
if __name__ == "__main__":
    # This would be used with actual data
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    try:
        data = loader.load_raw_data()
    except FileNotFoundError:
        data = loader.generate_sample_data()
    
    cleaned_data = loader.clean_data()
    
    # Run EDA
    analyzer = ExploratoryAnalyzer(cleaned_data)
    results = analyzer.run_complete_eda()
    
    # Print insights
    print("\nKey Insights:")
    for insight in results['insights']:
        print(f"• {insight}")