# Logistic Regression Model Information

## 📊 Model Overview
- **Model Type**: LogisticRegression (Linear Classifier)
- **Status**: Baseline Model ⚠️
- **Performance**: Baseline comparison model
- **Training Time**: 24.23 seconds

## 📊 Performance Metrics
- **Cross-Validation Score**: 83.09%
- **Test Accuracy**: 81.61%
- **F1-Score**: 16.47%
- **ROC-AUC**: 84.11%
- **Precision**: 38.18%
- **Recall**: 10.50%

## ⚙️ Hyperparameters
- **C**: 0.1 (regularization strength)
- **Penalty**: L1 (Lasso regularization)
- **Solver**: SAGA
- **Max Iterations**: 1000

## 🎯 Model Characteristics
- **Model Type**: Linear classifier
- **Regularization**: L1 (Lasso) for feature selection
- **Decision Boundary**: Linear hyperplane
- **Interpretability**: High (linear coefficients)
- **Overfitting Score**: 0.0045 (excellent generalization)

## 🔍 Top 5 Important Features (by coefficient magnitude)
1. **satisfaction_level** (-2.147) - Strong negative predictor
2. **last_evaluation** (-1.892) - Strong negative predictor
3. **number_project** (0.234) - Positive predictor
4. **time_spend_company** (0.198) - Positive predictor
5. **salary_low** (0.156) - Positive predictor

## 🚀 Use Cases
- **Primary**: Baseline comparison and interpretability analysis
- **Strengths**:
  - Fast training and inference
  - Highly interpretable coefficients
  - Good generalization (low overfitting)
  - Probabilistic outputs
- **Limitations**: 
  - Lower accuracy compared to ensemble methods
  - Assumes linear relationships
  - Poor recall for minority class

## 📈 Model Insights
- **Negative Predictors**: High satisfaction and evaluation scores reduce turnover risk
- **Positive Predictors**: More projects, longer tenure, and low salary increase risk
- **Class Imbalance Impact**: Model struggles with minority class detection
- **Linear Assumptions**: May miss complex feature interactions

## 📁 Files
- `logistic_regression_model.pkl` - Trained model binary
- `hyperparameters.json` - Model configuration
- `metrics.json` - Performance metrics
- `feature_importance.json` - Coefficient values and rankings

## 🎲 Model ID
**lr_v1_20250802** - Created on August 2, 2025
