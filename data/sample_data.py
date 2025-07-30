"""
Sample data generator for Employee Turnover Prediction project.

This module generates synthetic employee data for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import DEPARTMENTS, SALARY_LEVELS

def generate_employee_data(n_samples: int = 5000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic employee turnover data.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the generated data
        
    Returns:
        DataFrame with synthetic employee data
    """
    np.random.seed(42)
    
    print(f"Generating {n_samples} synthetic employee records...")
    
    # Generate base features
    data = {
        # Satisfaction level (0.0 to 1.0)
        'satisfaction_level': np.random.beta(2, 2, n_samples),
        
        # Last evaluation score (0.3 to 1.0 - realistic range)
        'last_evaluation': np.random.uniform(0.36, 1.0, n_samples),
        
        # Number of projects (2 to 7)
        'number_project': np.random.choice([2, 3, 4, 5, 6, 7], n_samples, 
                                         p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]),
        
        # Average monthly hours (120 to 310)
        'average_montly_hours': np.random.normal(200, 40, n_samples).astype(int),
        
        # Time spent at company (1 to 10 years)
        'time_spend_company': np.random.choice(range(1, 11), n_samples,
                                             p=[0.15, 0.15, 0.15, 0.15, 0.1, 
                                                0.1, 0.08, 0.06, 0.04, 0.02]),
        
        # Work accident (binary)
        'Work_accident': np.random.binomial(1, 0.144, n_samples),
        
        # Promotion in last 5 years (binary, rare)
        'promotion_last_5years': np.random.binomial(1, 0.021, n_samples),
        
        # Department
        'department': np.random.choice(DEPARTMENTS, n_samples),
        
        # Salary level
        'salary': np.random.choice(SALARY_LEVELS, n_samples, p=[0.48, 0.43, 0.09])
    }
    
    df = pd.DataFrame(data)
    
    # Clip hours to realistic range
    df['average_montly_hours'] = df['average_montly_hours'].clip(120, 310)
    
    # Generate realistic quit patterns
    quit_probability = np.zeros(n_samples)
    
    # Satisfaction-based factors (strongest predictor)
    quit_probability += 0.4 * (1 - df['satisfaction_level'])
    
    # Overwork factor
    quit_probability += 0.2 * (df['average_montly_hours'] > 250).astype(float)
    quit_probability += 0.1 * (df['average_montly_hours'] > 280).astype(float)
    
    # Underwork factor (very few hours might indicate dissatisfaction)
    quit_probability += 0.15 * (df['average_montly_hours'] < 150).astype(float)
    
    # High project load
    quit_probability += 0.1 * (df['number_project'] >= 6).astype(float)
    quit_probability += 0.05 * (df['number_project'] >= 7).astype(float)
    
    # Low project load (might indicate being sidelined)
    quit_probability += 0.1 * (df['number_project'] <= 2).astype(float)
    
    # Low evaluation scores
    quit_probability += 0.15 * (df['last_evaluation'] < 0.5).astype(float)
    
    # Time at company (longer tenure might lead to leaving)
    quit_probability += 0.05 * (df['time_spend_company'] >= 6).astype(float)
    quit_probability += 0.1 * (df['time_spend_company'] >= 8).astype(float)
    
    # Salary impact
    salary_quit_rates = {'low': 0.15, 'medium': 0.05, 'high': -0.05}
    for salary_level in SALARY_LEVELS:
        mask = df['salary'] == salary_level
        quit_probability[mask] += salary_quit_rates[salary_level]
    
    # Work accident (might increase quit probability slightly)
    quit_probability += 0.05 * df['Work_accident']
    
    # Promotion (reduces quit probability)
    quit_probability -= 0.2 * df['promotion_last_5years']
    
    # Department-specific effects
    dept_quit_rates = {
        'sales': 0.05, 'technical': -0.02, 'support': 0.03, 'IT': -0.01,
        'hr': 0.02, 'accounting': 0.01, 'marketing': 0.02, 'product_mng': -0.01,
        'RandD': -0.03, 'management': -0.05
    }
    for dept in DEPARTMENTS:
        mask = df['department'] == dept
        quit_probability[mask] += dept_quit_rates.get(dept, 0)
    
    # Add some random noise
    quit_probability += np.random.normal(0, 0.1, n_samples)
    
    # Clip probabilities to [0, 1] range
    quit_probability = np.clip(quit_probability, 0, 1)
    
    # Generate binary quit variable
    df['quit'] = np.random.binomial(1, quit_probability)
    
    # Add some realistic correlations
    # High performers with low satisfaction are more likely to quit
    high_perf_low_sat = (df['last_evaluation'] > 0.8) & (df['satisfaction_level'] < 0.3)
    df.loc[high_perf_low_sat, 'quit'] = np.random.binomial(1, 0.8, high_perf_low_sat.sum())
    
    # Low performers with high satisfaction are less likely to quit
    low_perf_high_sat = (df['last_evaluation'] < 0.5) & (df['satisfaction_level'] > 0.7)
    df.loc[low_perf_high_sat, 'quit'] = np.random.binomial(1, 0.2, low_perf_high_sat.sum())
    
    print(f"Generated data with {df['quit'].sum()} quits out of {n_samples} employees")
    print(f"Quit rate: {df['quit'].mean():.1%}")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return df

def generate_sample_files():
    """Generate sample data files for the project."""
    # Ensure data directory exists
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main dataset
    main_data = generate_employee_data(
        n_samples=10000, 
        save_path="data/raw/employee_data.csv"
    )
    
    # Generate smaller test dataset
    test_data = generate_employee_data(
        n_samples=1000,
        save_path="data/raw/employee_data_small.csv"
    )
    
    print("\nDataset Statistics:")
    print("=" * 40)
    print("Main Dataset:")
    print(f"  Shape: {main_data.shape}")
    print(f"  Quit rate: {main_data['quit'].mean():.1%}")
    print(f"  Departments: {main_data['department'].nunique()}")
    print(f"  Salary levels: {main_data['salary'].nunique()}")
    
    print("\nSample data generation completed!")
    
    return main_data, test_data

if __name__ == "__main__":
    # Generate sample data
    main_data, test_data = generate_sample_files()
    
    # Show sample records
    print("\nSample Records:")
    print("=" * 60)
    print(main_data.head(10))
    
    print(f"\nData Types:")
    print(main_data.dtypes)