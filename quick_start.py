"""
Quick Start Example - Customer Churn Prediction
This script demonstrates basic usage of the churn prediction pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.model_trainer import ModelTrainer, train_and_compare_models
from src.evaluator import ModelEvaluator, quick_evaluate
from src.data_loader import DataLoader
from utils.helpers import create_feature_summary_report

def create_sample_data(n_samples=1000):
    """Create sample customer churn data for demonstration"""
    np.random.seed(42)
    
    # Generate synthetic customer data
    data = {
        'tenure': np.random.randint(1, 73, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    churn_prob = (
        0.1 + 
        0.3 * (df['Contract'] == 'Month-to-month') +
        0.2 * (df['tenure'] < 12) +
        0.2 * (df['MonthlyCharges'] > 80) +
        0.1 * df['SeniorCitizen']
    )
    
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    return df

def quick_start_example():
    """Run a quick example of the churn prediction pipeline"""
    print("🚀 Customer Churn Prediction - Quick Start Example")
    print("="*60)
    
    # Step 1: Create or load sample data
    print("\n📊 Step 1: Creating sample data...")
    df = create_sample_data(1000)
    print(f"Created dataset with {df.shape[0]} customers and {df.shape[1]} features")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    
    # Step 2: Feature summary
    print("\n🔍 Step 2: Data analysis...")
    summary = create_feature_summary_report(df, 'Churn')
    print(f"Dataset shape: {summary['dataset_info']['shape']}")
    print(f"Missing values: {summary['dataset_info']['total_missing_values']}")
    print(f"Numerical features: {summary['feature_types']['numerical_count']}")
    print(f"Categorical features: {summary['feature_types']['categorical_count']}")
    
    # Step 3: Prepare data for modeling
    print("\n🔧 Step 3: Preparing data for modeling...")
    
    # Simple preprocessing
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Encode categorical variables
    df_processed = df.copy()
    le = LabelEncoder()
    
    categorical_cols = ['Contract', 'PaymentMethod', 'gender', 'InternetService']
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Separate features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train models
    print("\n🤖 Step 4: Training models...")
    
    # Use the utility function for quick training
    trainer, results = train_and_compare_models(
        X_train.values, y_train.values,
        model_names=['logistic_regression', 'random_forest'],
        optimization_method='default'  # Use default parameters for quick demo
    )
    
    print("Models trained successfully!")
    
    # Step 5: Evaluate models
    print("\n📈 Step 5: Evaluating models...")
    
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_multiple_models(
        trainer.trained_models,
        X_test.values, y_test.values
    )
    
    # Print results
    print("\nModel Performance Summary:")
    print("-" * 40)
    
    for model_name, metrics in evaluation_results.items():
        if 'error' not in metrics:
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    # Step 6: Best model recommendation
    print("\n🏆 Step 6: Model recommendation...")
    
    if evaluator.comparison_results and 'best_model' in evaluator.comparison_results:
        best_models = evaluator.comparison_results['best_model']
        print(f"\nRecommended model: {best_models['by_f1']}")
        print("(Based on F1-Score, which balances precision and recall)")
    
    print("\n✅ Quick start example completed!")
    print("\nNext steps:")
    print("1. Use your own dataset by replacing the sample data")
    print("2. Tune hyperparameters using 'random_search' or 'optuna'")
    print("3. Try additional models like 'gradient_boosting' or 'xgboost'")
    print("4. Explore the full pipeline using main.py")
    
    return trainer, evaluator

def demonstrate_advanced_features():
    """Demonstrate advanced features of the pipeline"""
    print("\n🔬 Advanced Features Demonstration")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(500)
    
    # Demonstrate feature engineering
    print("\n1. Feature Engineering:")
    from src.feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    
    # Show some advanced preprocessing capabilities
    print("   - Automated feature type detection")
    print("   - Missing value handling")
    print("   - Categorical encoding")
    print("   - Feature scaling")
    print("   - Class imbalance handling")
    
    # Demonstrate model training with hyperparameter optimization
    print("\n2. Advanced Model Training:")
    print("   - Grid search hyperparameter optimization")
    print("   - Random search optimization")
    print("   - Optuna-based optimization")
    print("   - Cross-validation")
    
    # Demonstrate evaluation features
    print("\n3. Comprehensive Evaluation:")
    print("   - Multiple evaluation metrics")
    print("   - ROC and Precision-Recall curves")
    print("   - Feature importance analysis")
    print("   - Model comparison visualizations")
    print("   - Automated report generation")
    
    print("\n4. Visualization Capabilities:")
    print("   - Interactive plots with Plotly")
    print("   - Statistical visualizations")
    print("   - Model performance plots")
    print("   - Feature analysis plots")

if __name__ == "__main__":
    # Run the quick start example
    trainer, evaluator = quick_start_example()
    
    # Show advanced features
    demonstrate_advanced_features()
    
    print(f"\n📁 For more examples, check the notebooks directory")
    print(f"📁 For full pipeline, run: python main.py")
