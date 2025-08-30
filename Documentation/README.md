# AI Project - Employee Turnover Prediction

## 🎯 Project Overview

This project is a comprehensive machine learning pipeline for predicting employee turnover using **Random Forest and XGBoost** algorithms. The project has been optimized to focus exclusively on these two high-performing tree-based models, providing automated model comparison, hyperparameter optimization, and comprehensive evaluation.

### Key Features
- **Focused Algorithm Set**: Random Forest and XGBoost only for optimal performance
- **Automated Model Selection**: Best model identification based on cross-validation scores
- **Comprehensive Evaluation**: Multiple metrics with detailed performance analysis
- **Organized Structure**: Clean project organization with dedicated directories
- **JSON Outputs**: Structured results in JSON format for easy integration
- **Validation Tools**: Comprehensive validation and debugging utilities

## 📁 Directory Structure

```
AI-Project/
├── 📊 data/                          # Dataset storage
│   ├── raw/                          # Original datasets
│   │   ├── employee_data.csv         # Primary training data (11,413 rows, 28 features)
│   │   └── sample_customer_churn.csv # Alternative dataset
│   └── processed/                    # Cleaned and engineered datasets
│       ├── balanced_data.csv         # Class-balanced dataset
│       ├── feature_engineered_data.csv # Final training dataset
│       ├── feature_info.json         # Feature engineering metadata
│       └── processed_data.csv        # Intermediate processed data
├── 🤖 models/                        # Model artifacts (RF & XGBoost only)
│   ├── random_forest_model.pkl      # Trained Random Forest
│   ├── xgboost_model.pkl           # Trained XGBoost
│   ├── best_parameters.pkl          # Optimal hyperparameters
│   ├── evaluation_data.pkl          # Train/test splits
│   ├── model_registry_rf_xgb.json   # Model metadata
│   ├── random_forest/              # RF-specific files
│   │   ├── feature_importance.json
│   │   ├── hyperparameters.json
│   │   ├── metrics.json
│   │   └── model_info.md
│   └── xgboost/                   # XGBoost-specific files
│       ├── feature_importance.json
│       ├── hyperparameters.json
│       ├── metrics.json
│       └── model_info.md
├── 📄 json/                        # All JSON outputs
│   ├── model_comparison.json       # RF vs XGBoost comparison
│   ├── best_parameters.json       # Hyperparameters in JSON format
│   ├── cv_results.json            # Cross-validation scores
│   ├── model_summary.json         # Training summary
│   ├── test_results.json          # Test evaluation metrics
│   ├── random_forest_feature_importance.json
│   └── xgboost_feature_importance.json
├── 📈 results/                     # Training results and visualizations
│   ├── figures/                   # Model performance plots
│   │   ├── model_comparison.png   # Performance comparison chart
│   │   ├── roc_curves.png         # ROC curves visualization
│   │   └── target_distribution.png # Target variable distribution
│   ├── reports/                   # Generated reports
│   └── metrics/                   # Performance metrics
├── 🔧 src/                        # Core modules (updated for RF & XGBoost)
│   ├── __init__.py
│   ├── config.py                 # Configuration (RF & XGBoost only)
│   ├── model_trainer.py          # Training pipeline (restricted)
│   ├── data_loader.py            # Data loading utilities
│   ├── feature_engineering.py    # Feature engineering
│   ├── evaluator.py              # Model evaluation
│   └── visualizer.py             # Visualization tools
├── 🚀 main/                       # Execution scripts
│   ├── main.py                   # Full pipeline (RF & XGBoost)
│   ├── rf_xgb_trainer.py         # Quick trainer script
│   └── quick_start.py            # Simplified execution
├── 🔍 validations/                # Validation and debugging scripts
│   ├── __init__.py               # Package initialization
│   ├── analyze_cv.py             # Cross-validation analysis
│   ├── analyze_pkl_detailed.py   # Detailed PKL file analysis
│   ├── analyze_pkl_files.py      # Basic PKL file analysis
│   ├── check_regularization.py   # Regularization parameter analysis
│   ├── debug_paths.py            # Path debugging utilities
│   ├── debug_trainer.py          # ModelTrainer debugging
│   ├── path_demo.py              # Path navigation demonstration
│   ├── setup.py                  # Project setup script
│   ├── simple_validation.py      # Simple validation tests
│   ├── validate_restructure.py   # Complete project validation
│   └── validation_output.txt     # Validation results log
├── 🧪 Testing/                    # Unit tests and integration tests
│   ├── test_paths.py             # Path testing utilities
│   ├── test_01_string_formatting.py
│   ├── test_02_model_trainer_return.py
│   ├── test_03_fixed_wrapper.py
│   ├── test_04_complete_validation.py
│   └── ...
├── 📓 notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_class_imbalance_analysis.ipynb
│   ├── 04_model_training.ipynb      # RF & XGBoost focus
│   ├── 05_model_evaluation.ipynb    # Comparison analysis
│   └── 06_ModelTrainer_Fix_and_Integration.ipynb
├── 🛠️ utils/                      # Utility functions
│   ├── __init__.py
│   ├── helpers.py
│   ├── plotting.py
│   └── preprocessing.py
├── ✅ validation/                  # Legacy validation scripts
└── 📋 Documentation/               # Project documentation
    ├── README.md
    └── requirements.txt
```

## 🚀 Usage Instructions

### Quick Start (Recommended)

#### Method 1: Quick Trainer Script
```bash
# Quick training with default settings
python main/rf_xgb_trainer.py

# With custom data and optimization
python main/rf_xgb_trainer.py --data-path data/raw/employee_data.csv --optimization random_search
```

#### Method 2: Full Pipeline
```bash
# Full pipeline with all steps
python main/main.py --optimization random_search

# Train specific models only
python main/main.py --models random_forest xgboost
```

#### Method 3: Python Import
```python
from main.rf_xgb_trainer import quick_train_rf_xgb

# Train and compare models
results = quick_train_rf_xgb(
    data_path='data/raw/employee_data.csv',
    optimization='random_search'
)

# Access best model
best_model = results['best_model']
best_name = results['best_model_name']
```

### Validation and Debugging

#### Run All Validations
```bash
# Complete project validation
python -m validations.validate_restructure

# Simple validation tests
python -m validations.simple_validation
```

#### Analyze Project Components
```bash
# Cross-validation analysis
python -m validations.analyze_cv

# PKL file analysis (detailed)
python -m validations.analyze_pkl_detailed

# Regularization analysis
python -m validations.check_regularization

# Debug path issues
python -m validations.debug_paths

# Test ModelTrainer functionality
python -m validations.debug_trainer
```

#### Path Navigation Demo
```bash
# Understand path navigation concepts
python -m validations.path_demo
```

### Check Evaluation Results

#### JSON Results
- **Model Comparison**: `json/model_comparison.json`
- **Cross-Validation**: `json/cv_results.json`
- **Test Results**: `json/test_results.json`
- **Feature Importance**: `json/{model}_feature_importance.json`

#### Visualizations
- **Performance Charts**: `results/figures/model_comparison.png`
- **ROC Curves**: `results/figures/roc_curves.png`
- **Target Distribution**: `results/figures/target_distribution.png`

#### Model Artifacts
- **Best Model**: Automatically identified and saved
- **Hyperparameters**: `models/best_parameters.pkl` and `json/best_parameters.json`
- **Model Registry**: `models/model_registry_rf_xgb.json`

## 📦 Dependencies

### Core Requirements
```txt
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
scikit-learn>=1.1.0
imbalanced-learn>=0.9.0
xgboost>=1.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
jupyterlab>=3.4.0
ipywidgets>=8.0.0
tqdm>=4.64.0
joblib>=1.1.0
optuna>=3.0.0
shap>=0.41.0
```

### Installation
```bash
# Install from requirements file
pip install -r requirements.txt

# Or install from setup script
python validations/setup.py install

# Or install in development mode
pip install -e .
```

## 🎯 Model Configuration

### Random Forest Hyperparameters
- **n_estimators**: [100, 200, 300, 400, 500]
- **max_depth**: [10, 20, 30, 40, None]
- **min_samples_split**: [2, 5, 10, 15]
- **min_samples_leaf**: [1, 2, 4, 8]
- **max_features**: ['sqrt', 'log2', None]
- **bootstrap**: [True, False]

### XGBoost Hyperparameters
- **n_estimators**: [100, 200, 300, 400, 500]
- **learning_rate**: [0.01, 0.05, 0.1, 0.15, 0.2]
- **max_depth**: [3, 4, 5, 6, 7, 8]
- **subsample**: [0.8, 0.85, 0.9, 0.95, 1.0]
- **colsample_bytree**: [0.8, 0.85, 0.9, 0.95, 1.0]
- **reg_alpha**: [0, 0.01, 0.1, 0.5, 1.0] (L1 regularization)
- **reg_lambda**: [0, 0.01, 0.1, 0.5, 1.0] (L2 regularization)

## 🔄 Cross-Validation & Regularization

### Cross-Validation Setup
- **Method**: 5-fold StratifiedKFold
- **Scoring**: ROC-AUC (primary), F1-score (secondary)
- **Iterations**: 100 (RandomizedSearchCV)
- **Parallel Processing**: Full CPU utilization
- **Random State**: Fixed for reproducibility

### Regularization Techniques
- **Random Forest**: Tree depth limits, sample splits, bootstrap sampling
- **XGBoost**: L1/L2 regularization, subsampling, column sampling
- **Early Stopping**: Prevented through proper validation

## 📊 Performance Metrics

Both models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate (weighted average)
- **Recall**: Sensitivity (weighted average)
- **F1-Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Area under ROC curve
- **Cross-Validation Score**: 5-fold CV performance

## 🏆 Model Selection Criteria

1. **Primary**: Cross-validation ROC-AUC score
2. **Secondary**: Test set F1-score
3. **Tertiary**: Test set accuracy
4. **Considerations**: Training time, interpretability, feature importance

## 📋 Output Storage

### Models Directory
- **Trained Models**: `{model_name}_model.pkl`
- **Hyperparameters**: `best_parameters.pkl`
- **Evaluation Data**: `evaluation_data.pkl`
- **Model-Specific**: `{model_name}/` subdirectories

### JSON Directory
- **All Results**: Structured JSON format
- **Feature Importance**: Per-model feature rankings
- **Comparison Data**: Model performance comparisons
- **Metadata**: Training configuration and timestamps

### Results Directory
- **Visualizations**: `figures/` subdirectory
- **Reports**: Generated analysis reports
- **Metrics**: Detailed performance metrics

## 🔧 Notes

### Validation Scripts
- All validation and debug scripts are now organized under `validations/`
- Use Python module syntax: `python -m validations.script_name`
- Scripts are self-contained with proper path handling
- Comprehensive validation available via `validate_restructure.py`

### Path Handling
- All scripts use relative imports and proper path resolution
- Project root is automatically detected using `Path(__file__).parent.parent`
- No hardcoded paths - fully portable across environments
- Import issues resolved through proper `sys.path` management

### Project Structure
- **Organized**: Clear separation of concerns across directories
- **Scalable**: Easy to add new models or validation scripts
- **Maintainable**: Consistent structure and naming conventions
- **Documented**: Comprehensive documentation and examples

## 🆘 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running scripts from project root
2. **Path Issues**: Use `python -m validations.debug_paths` to check paths
3. **Model Issues**: Use `python -m validations.debug_trainer` to test models
4. **Validation Errors**: Run `python -m validations.validate_restructure` for full check

### Getting Help
- Check validation outputs in `validations/validation_output.txt`
- Run individual validation scripts for specific issues
- Review JSON outputs for detailed results
- Check model artifacts in `models/` directory

---

**Last Updated**: Project restructured with dedicated `validations/` directory and comprehensive documentation.
