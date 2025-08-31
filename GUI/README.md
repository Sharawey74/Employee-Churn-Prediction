# Employee Churn Prediction - Streamlit GUI

## 🎯 Overview

This is an interactive web application built with Streamlit for predicting employee churn using Random Forest and XGBoost machine learning models. The application provides a comprehensive interface for making predictions, exploring data visualizations, and comparing model performance.

## 🚀 Features

### 🏠 Main Dashboard
- **Interactive Input Form**: Enter employee data through user-friendly sliders and dropdowns
- **Model Selection**: Choose between Random Forest and XGBoost models
- **Real-time Predictions**: Get instant churn predictions with confidence scores
- **Visual Results**: Probability gauges and detailed result displays

### 📊 Data Visualization
- **Dataset Overview**: Key statistics and metrics
- **Interactive Charts**: Feature distributions, correlation heatmaps
- **Churn Analysis**: Distribution of churn vs retention
- **Feature Importance**: Model-specific feature importance plots

### ⚖️ Model Comparison
- **Performance Metrics**: Side-by-side comparison of accuracy, precision, recall, F1-score, ROC-AUC
- **Radar Charts**: Visual performance comparison
- **Interactive Comparisons**: Detailed metric analysis
- **Model Rankings**: Based on cross-validation scores

### ℹ️ About Section
- **Project Information**: Comprehensive project overview
- **Dataset Details**: Information about preprocessing and features
- **Model Explanations**: Technical details about algorithms used
- **Performance Summary**: Current model performance statistics

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Trained Random Forest and XGBoost models (already included in project)

### Setup Instructions

1. **Navigate to GUI directory**:
   ```bash
   cd "C:\Users\DELL\Desktop\AI-Project\AI-Project\GUI"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

## 📁 File Structure

```
GUI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎮 Usage Guide

### Making Predictions

1. **Navigate to Main Dashboard**
2. **Select a Model**: Choose between Random Forest or XGBoost
3. **Enter Employee Data**:
   - Satisfaction Level (0-1)
   - Last Evaluation Score (0-1)
   - Number of Projects (2-7)
   - Average Monthly Hours (96-310)
   - Years at Company (2-10)
   - Work Accident (Yes/No)
   - Promotion in Last 5 Years (Yes/No)
   - Department (dropdown selection)
   - Salary Level (Low/Medium/High)
4. **Click "Predict Employee Churn"**
5. **View Results**: Prediction, confidence score, and probabilities

### Exploring Data

1. **Go to Visualization Tab**
2. **View Dataset Overview**: Total records, features, churn rate
3. **Explore Feature Distributions**: Select different features to analyze
4. **Check Correlations**: Interactive correlation heatmap
5. **Review Feature Importance**: Model-specific importance rankings

### Comparing Models

1. **Visit Model Comparison Tab**
2. **Review Performance Table**: All metrics side-by-side
3. **Analyze Radar Chart**: Visual performance comparison
4. **Select Specific Metrics**: Detailed metric comparisons
5. **Check Rankings**: Model performance rankings

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting model
- **Joblib**: Model loading

### Data Pipeline
1. **Data Loading**: Reads feature-engineered dataset
2. **Model Loading**: Loads trained RF and XGBoost models
3. **Input Processing**: Converts user input to model format
4. **Prediction**: Generates predictions with probabilities
5. **Visualization**: Creates interactive charts and displays

### Model Integration
- **Random Forest**: Loaded from `../models/random_forest_model.pkl`
- **XGBoost**: Loaded from `../models/xgboost_model.pkl`
- **Results**: Reads performance data from `../json/` directory
- **Feature Importance**: Displays model-specific importance scores

## 🚨 Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure model files exist in `../models/` directory
   - Check file paths in `app.py`

2. **Data not found**:
   - Verify dataset exists at specified path
   - Check `DATA_PATH` constant in `app.py`

3. **Import errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Streamlit not starting**:
   - Ensure Streamlit is installed: `pip install streamlit`
   - Try: `python -m streamlit run app.py`

### Performance Tips

- **Caching**: The app uses Streamlit caching for better performance
- **Data Loading**: Large datasets are cached after first load
- **Model Loading**: Models are loaded once and cached

## 🎨 Customization

### Modifying the Interface
- **Colors**: Update color schemes in Plotly charts
- **Layout**: Modify column layouts and spacing
- **Features**: Add new input fields in `create_input_form()`

### Adding New Models
1. Save new model to `../models/` directory
2. Update `load_models()` function
3. Add to model selection dropdown
4. Update prediction logic if needed

### Custom Visualizations
- Add new charts in `visualization_tab()`
- Modify existing Plotly configurations
- Add new metrics to comparison tab

## 📊 Performance Metrics

The application displays the following metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to actual churners
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## 🔒 Security Notes

- The application runs locally by default
- No sensitive data is transmitted externally
- Models and data remain on local system
- Consider authentication if deploying publicly

## 📝 Version History

- **v2.0.0**: Initial Streamlit GUI implementation
  - Four-tab interface design
  - Interactive prediction dashboard
  - Comprehensive data visualization
  - Model performance comparison
  - Professional about section

## 👨‍💻 Developer Information

**Author**: Sharawey74  
**Project**: AI Project - Employee Turnover Prediction  
**Framework**: Streamlit  
**Date**: August 2025  

For technical support or feature requests, please refer to the main project repository.

---

**Note**: This GUI application is designed to work with the existing AI Project structure and requires the trained models and processed data to be available in the parent directories.
