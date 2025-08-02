# Gradient Boosting Model Information

## 🌳 Model Overview
- **Model Type**: GradientBoostingClassifier (Sequential Ensemble)
- **Status**: Production Ready ✅
- **Performance**: Second-best model with excellent balance
- **Training Time**: 128.28 seconds

## 📊 Performance Metrics
- **Cross-Validation Score**: 98.47%
- **Test Accuracy**: 97.97%
- **F1-Score**: 94.01%
- **ROC-AUC**: 98.49%
- **Precision**: 95.84%
- **Recall**: 92.25%

## ⚙️ Hyperparameters
- **n_estimators**: 100 trees
- **learning_rate**: 0.2
- **max_depth**: 3 levels
- **subsample**: 1.0 (100% of data)
- **Loss Function**: Deviance (logistic regression)

## 🎯 Model Characteristics
- **Ensemble Type**: Sequential Boosting
- **Tree Count**: 100 sequential trees
- **Learning Strategy**: Each tree corrects previous errors
- **Regularization**: Shallow trees (depth=3) prevent overfitting
- **Overfitting Score**: 0.0052 (good generalization)

## 🔍 Feature Importance Top 5
1. **satisfaction_level** (29.8%) - Employee satisfaction rating
2. **last_evaluation** (27.6%) - Recent performance evaluation
3. **number_project** (9.5%) - Number of projects assigned
4. **average_montly_hours** (9.2%) - Monthly working hours
5. **time_spend_company** (7.9%) - Years at company

## 🚀 Use Cases
- **Primary**: Balanced performance and interpretability
- **Strengths**:
  - Strong predictive performance
  - Good balance of accuracy and speed
  - Sequential learning captures complex patterns
  - Robust feature importance
- **Considerations**: Sensitive to outliers, sequential training

## 📈 Learning Progression
- **Best Iteration**: 95 trees
- **Convergence**: Achieved ✅
- **Training Loss**: 0.074 (final)
- **Validation Loss**: 0.084 (final)
- **Learning Stability**: Consistent improvement

## 📁 Files
- `gradient_boosting_model.pkl` - Trained model binary
- `hyperparameters.json` - Model configuration
- `metrics.json` - Performance metrics
- `feature_importance.json` - Feature importance scores
- `learning_curve.json` - Training progression details

## 🎲 Model ID
**gb_v1_20250802** - Created on August 2, 2025
