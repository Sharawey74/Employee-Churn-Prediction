# 🎯 Models Directory

This directory contains all trained machine learning models for the Employee Turnover Prediction project.

## 📊 Model Performance Summary

| Model | Accuracy | F1-Score | ROC-AUC | Status |
|-------|----------|----------|---------|---------|
| **XGBoost** 🏆 | 98.32% | 94.95% | 98.72% | Best Model |
| **Gradient Boosting** | 97.97% | 94.01% | 98.49% | Production Ready |
| **Random Forest** | 98.19% | 94.49% | 98.21% | Production Ready |
| **Logistic Regression** | 81.61% | 16.47% | 84.11% | Baseline |

## 📁 Directory Structure

### 🔍 Individual Model Directories

Each model has its own directory containing:

- `*.pkl` - Serialized trained model
- `hyperparameters.json` - Optimized hyperparameters
- `metrics.json` - Performance metrics
- `feature_importance.json` - Feature importance scores (tree-based models)
- `model_info.md` - Detailed model documentation

### 📦 Archives (`/archives/`)

Version control for models:

- Date-stamped model versions
- Historical performance tracking
- Rollback capabilities

## 🚀 Quick Start

### Load XGBoost Model (Best Performing)

```python
import joblib
from pathlib import Path

# Load the best performing model (XGBoost)
xgb_model = joblib.load('models/xgboost/xgboost_model.pkl')

# Make predictions
predictions = xgb_model.predict(X_test)
```

### Load Specific Model
```python
# Load Random Forest
rf_model = joblib.load('models/random_forest/random_forest_model.pkl')

# Load XGBoost
xgb_model = joblib.load('models/xgboost/xgboost_model.pkl')
```

## 📈 Model Details

### XGBoost (Best Model)
- **Type**: Gradient Boosting Ensemble
- **Trees**: 100 estimators
- **Depth**: 5 levels
- **Learning Rate**: 0.1
- **Use Case**: Production deployment

### Random Forest
- **Type**: Bagging Ensemble
- **Trees**: 300 estimators
- **Depth**: 30 levels
- **Use Case**: Interpretability + Performance

### Gradient Boosting
- **Type**: Sequential Ensemble
- **Trees**: 100 estimators
- **Depth**: 3 levels
- **Learning Rate**: 0.2
- **Use Case**: Balanced performance

### Logistic Regression
- **Type**: Linear Classifier
- **Regularization**: L1 (Lasso)
- **C**: 0.1
- **Use Case**: Baseline comparison

## 🔧 Model Management

### Training New Models
```bash
# Train all models
python main.py --optimization random_search

# Train specific model
python main.py --models random_forest xgboost
```

### Model Evaluation
```bash
# Evaluate all models
python -m src.evaluator

# Compare models
python scripts/model_comparison.py
```

## 📊 Feature Importance

All tree-based models provide feature importance scores:

1. **satisfaction_level** - Employee satisfaction rating
2. **last_evaluation** - Recent performance evaluation
3. **number_project** - Number of projects assigned
4. **average_montly_hours** - Monthly working hours
5. **time_spend_company** - Years at company
6. **department_*** - Department one-hot encoded features
7. **salary_*** - Salary level encoded features

## 🎯 Model Selection Criteria

### For Production (XGBoost chosen):
- ✅ Highest accuracy (98.32%)
- ✅ Best F1-score (94.95%)
- ✅ Fastest inference time
- ✅ Good generalization (low overfitting: 0.0028)

### For Interpretability (Random Forest):
- ✅ High accuracy (98.19%)
- ✅ Easy feature importance interpretation
- ✅ Robust to outliers
- ✅ Less prone to overfitting

## 🚨 Important Notes

1. **Class Imbalance**: Dataset has 17.3% positive rate (employee turnover)
2. **Feature Engineering**: 18 features from original 10 (one-hot encoding)
3. **Cross-Validation**: 5-fold stratified CV used for all models
4. **Hyperparameter Tuning**: RandomizedSearchCV with 50 iterations

## 📝 Version History

- **v1.0** (2025-08-02): Initial model training with 4 algorithms
- **v0.9** (2025-08-01): Feature engineering and data preprocessing
- **v0.5** (2025-07-30): Exploratory data analysis and baseline models

## 🔗 Related Files

- **Training**: `main.py`
- **Configuration**: `src/config.py`
- **Evaluation**: `src/evaluator.py`
- **Feature Engineering**: `src/feature_engineering.py`
- **Results**: `results/reports/`
- **Visualizations**: `results/figures/`

---
*Last updated: August 2, 2025*
*Best model: XGBoost (98.32% accuracy)*
