# Customer Churn Prediction - Complete ML Pipeline

A comprehensive machine learning project to predict customer churn using multiple algorithms and advanced analysis techniques.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for predicting customer churn, focusing on:

- **Business Value**: Identify at-risk customers before they leave to reduce acquisition costs
- **Technical Excellence**: Comprehensive ML pipeline with proper evaluation and comparison
- **Scalability**: Modular design for easy extension and maintenance

## 🚀 Key Features

### 🤖 Machine Learning Models
- **Logistic Regression**: Fast baseline model
- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: Advanced boosting algorithm
- **XGBoost**: State-of-the-art gradient boosting

### 🔧 Advanced Preprocessing
- Automated feature type detection
- Missing value imputation (multiple strategies)
- Outlier detection and treatment
- Feature scaling and normalization
- Class imbalance handling (SMOTE, undersampling)

### 📊 Comprehensive Evaluation
- Multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Cross-validation with stratified sampling
- Learning curves and validation curves
- Feature importance analysis
- Model comparison and ranking

### 📈 Rich Visualizations
- Interactive plots with Plotly
- ROC and Precision-Recall curves
- Confusion matrices
- Feature importance plots
- Distribution analysis

### ⚙️ Hyperparameter Optimization
- Grid Search
- Random Search
- Optuna (Bayesian optimization)
- Automated parameter tuning

## 📁 Project Structure

```
AI-Project/
├── 📖 README.md                       # This file
├── 📦 requirements.txt                # Dependencies
├── ⚙️ setup.py                       # Package configuration
├── 🚀 main.py                        # Main pipeline script
├── ⚡ quick_start.py                  # Quick demo script
│
├── data/                              # Data storage
│   ├── raw/                          # Original datasets
│   ├── processed/                    # Cleaned data
│   └── README.md                     # Data documentation
│
├── src/                              # Core source code
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py               # Data loading utilities
│   ├── exploratory_analysis.py      # EDA functions
│   ├── feature_engineering.py       # Feature processing
│   ├── model_trainer.py            # Model training
│   ├── evaluator.py               # Model evaluation
│   └── visualizer.py              # Visualization tools
│
├── utils/                           # Utility functions
│   ├── helpers.py                  # General utilities
│   ├── preprocessing.py           # Advanced preprocessing
│   └── plotting.py               # Plotting utilities
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_class_imbalance_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/                         # Saved models
│   ├── decision_tree/
│   ├── random_forest/
│   └── README.md
│
└── results/                       # Output results
    ├── figures/                   # Generated plots
    ├── reports/                   # Analysis reports
    └── metrics/                   # Performance metrics
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Project
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python quick_start.py
```

## 🚀 Quick Start

### Option 1: Quick Demo (5 minutes)
```bash
python quick_start.py
```
This runs a complete example with synthetic data.

### Option 2: Full Pipeline
```bash
python main.py --data-path your_data.csv
```

### Option 3: Custom Configuration
```bash
python main.py \
    --data-path data/raw/customer_data.csv \
    --optimization optuna \
    --models random_forest xgboost \
    --log-level INFO
```

## 📊 Usage Examples

### Basic Model Training
```python
from src.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train all models
results = trainer.train_all_models(X_train, y_train)

# Save models
trainer.save_models()
```

### Model Evaluation
```python
from src.evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate models
results = evaluator.evaluate_multiple_models(
    trainer.trained_models, X_test, y_test
)

# Generate report
report = evaluator.generate_evaluation_report()
print(report)
```

### Feature Engineering
```python
from src.feature_engineering import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer()

# Process features
X_processed = engineer.fit_transform(X_raw)
```

## 📈 Expected Results

The pipeline typically achieves:
- **Accuracy**: 85-92%
- **ROC-AUC**: 0.88-0.95
- **F1-Score**: 0.82-0.90

Performance varies based on:
- Data quality and size
- Feature engineering
- Model selection
- Hyperparameter tuning

## 🔬 Advanced Features

### Hyperparameter Optimization
```bash
# Grid search
python main.py --optimization grid_search

# Random search (recommended)
python main.py --optimization random_search

# Optuna (Bayesian optimization)
python main.py --optimization optuna
```

### Custom Model Configuration
Edit `src/config.py` to:
- Add new models
- Modify hyperparameter ranges
- Change evaluation metrics
- Adjust preprocessing options

### Interactive Analysis
Use Jupyter notebooks for detailed analysis:
```bash
jupyter lab notebooks/
```

## 📋 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --data-path PATH           Path to dataset file
  --optimization METHOD      Optimization method [grid_search|random_search|optuna|default]
  --models MODEL [MODEL...]  Specific models to train
  --log-level LEVEL         Logging level [DEBUG|INFO|WARNING|ERROR]
  --skip-eda                Skip exploratory data analysis
  --help                    Show help message
```

## 🎯 Task Breakdown (9 Core Tasks)

✅ **Task 1**: Import Libraries - Complete environment setup
✅ **Task 2**: Exploratory Data Analysis - Comprehensive data exploration
✅ **Task 3**: Encoding Categorical Variables - Advanced feature engineering
✅ **Task 4**: Class Imbalance Analysis - Multiple balancing techniques
✅ **Task 5**: Train/Validation Split - Stratified data splitting
✅ **Task 6-7**: Decision Tree Implementation - With hyperparameter tuning
✅ **Task 8**: Random Forest Implementation - Ensemble learning
✅ **Task 9**: Model Evaluation - Comprehensive performance assessment

## 📊 Output Files

After running the pipeline, you'll find:

- **Models**: `models/` - Trained model files (.pkl)
- **Figures**: `results/figures/` - All visualizations (.png)
- **Reports**: `results/reports/` - Evaluation reports (.txt)
- **Metrics**: `results/metrics/` - Performance metrics (.json)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Loading Issues**: Check data file path and format
   ```python
   # Verify data format
   import pandas as pd
   df = pd.read_csv('your_data.csv')
   print(df.info())
   ```

3. **Memory Issues**: For large datasets, use data sampling
   ```bash
   python main.py --data-path large_dataset.csv --sample-size 10000
   ```

### Getting Help

- Check the documentation in `docs/`
- Review example notebooks
- Run `python quick_start.py` for basic example
- Open an issue on GitHub

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
- Optuna for hyperparameter optimization
- The open-source community for inspiration

---

**Ready to predict customer churn? Start with `python quick_start.py`! 🚀**