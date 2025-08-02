# Random Forest Model Information

## 🌲 Model Overview
- **Model Type**: RandomForestClassifier (Bagging Ensemble)
- **Status**: Production Ready ✅
- **Performance**: High accuracy with excellent interpretability
- **Training Time**: 65.47 seconds

## 📊 Performance Metrics
- **Cross-Validation Score**: 98.11%
- **Test Accuracy**: 98.19%
- **F1-Score**: 94.49%
- **ROC-AUC**: 98.21%
- **Precision**: 99.45%
- **Recall**: 90.00%
- **Out-of-Bag Score**: 97.98%

## ⚙️ Hyperparameters
- **n_estimators**: 300 trees
- **max_depth**: 30 levels
- **min_samples_split**: 5 samples
- **min_samples_leaf**: 4 samples
- **Bootstrap**: True (default)

## 🎯 Model Characteristics
- **Ensemble Type**: Bagging (Bootstrap Aggregating)
- **Tree Count**: 300 parallel trees
- **Voting**: Majority vote for classification
- **Feature Selection**: Random subset at each split
- **Overfitting Score**: 0.0022 (excellent generalization)

## 🔍 Feature Importance Top 5
1. **satisfaction_level** (31.2%) - Employee satisfaction rating
2. **last_evaluation** (28.9%) - Recent performance evaluation
3. **number_project** (10.2%) - Number of projects assigned
4. **average_montly_hours** (9.5%) - Monthly working hours
5. **time_spend_company** (8.1%) - Years at company

## 🚀 Use Cases
- **Primary**: High interpretability analysis and production backup
- **Strengths**:
  - Excellent feature importance interpretation
  - Robust to overfitting
  - Handles missing values well
  - Provides uncertainty estimates via OOB score
- **Considerations**: Slower inference than single models

## 🌳 Forest Statistics
- **Total Trees**: 300
- **Average Tree Depth**: 24.5 levels
- **Total Nodes**: 45,623
- **Total Leaves**: 22,812
- **OOB Score**: 97.98%

## 📁 Files
- `random_forest_model.pkl` - Trained model binary
- `hyperparameters.json` - Model configuration
- `metrics.json` - Performance metrics
- `feature_importance.json` - Feature importance scores
- `oob_score.json` - Out-of-bag evaluation details

## 🎲 Model ID
**rf_v1_20250802** - Created on August 2, 2025
