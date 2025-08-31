#!/usr/bin/env python3
"""
Employee Churn Prediction - Streamlit GUI Application
=====================================================

Interactive web application for predicting employee churn using 
Random Forest and XGBoost models with comprehensive visualization 
and model comparison capabilities.

Author: Sharawey74
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import joblib
import json
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Constants - using absolute paths to avoid Streamlit path issues
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv"
MODELS_PATH = PROJECT_ROOT / "models"
JSON_PATH = PROJECT_ROOT / "json"

# Debug: Check if paths exist and provide fallback
def get_corrected_paths():
    """Get corrected paths if the default ones don't work"""
    global DATA_PATH, MODELS_PATH, JSON_PATH
    
    if not DATA_PATH.exists():
        # Try alternative path resolution (in case Streamlit changes working directory)
        alt_project_root = Path.cwd()
        if (alt_project_root / "data").exists():
            # We're already in the project root
            DATA_PATH = alt_project_root / "data" / "processed" / "feature_engineered_data.csv"
            MODELS_PATH = alt_project_root / "models"
            JSON_PATH = alt_project_root / "json"
        elif (alt_project_root.parent / "data").exists():
            # We're in a subdirectory, go up one level
            DATA_PATH = alt_project_root.parent / "data" / "processed" / "feature_engineered_data.csv"
            MODELS_PATH = alt_project_root.parent / "models"
            JSON_PATH = alt_project_root.parent / "json"
    
    return DATA_PATH, MODELS_PATH, JSON_PATH

# Initialize corrected paths
DATA_PATH, MODELS_PATH, JSON_PATH = get_corrected_paths()

@st.cache_data
def load_data():
    """Load the feature-engineered dataset"""
    try:
        # Ensure we have the correct path
        current_data_path, _, _ = get_corrected_paths()
        
        if not current_data_path.exists():
            st.error(f"❌ Data file not found!")
            st.info(f"🔍 Looking for: {current_data_path}")
            st.info(f"📁 Current working directory: {Path.cwd()}")
            st.info(f"📂 Project root: {PROJECT_ROOT}")
            
            # List available files for debugging
            if current_data_path.parent.exists():
                files = list(current_data_path.parent.glob("*.csv"))
                st.info(f"📄 Available CSV files in {current_data_path.parent}: {[f.name for f in files]}")
            
            return None
            
        df = pd.read_csv(current_data_path)
        st.success(f"✅ Data loaded successfully from: {current_data_path}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load trained Random Forest and XGBoost models"""
    models = {}
    try:
        # Get corrected paths
        _, current_models_path, _ = get_corrected_paths()
        
        # Load Random Forest
        rf_path = current_models_path / "random_forest_model.pkl"
        if rf_path.exists():
            models['Random Forest'] = joblib.load(rf_path)
            st.success(f"✅ Random Forest model loaded")
        else:
            st.warning(f"⚠️ Random Forest model not found at: {rf_path}")
        
        # Load XGBoost
        xgb_path = current_models_path / "xgboost_model.pkl"
        if xgb_path.exists():
            models['XGBoost'] = joblib.load(xgb_path)
            st.success(f"✅ XGBoost model loaded")
        else:
            st.warning(f"⚠️ XGBoost model not found at: {xgb_path}")
            
        if not models:
            st.error(f"❌ No models found in: {current_models_path}")
            # List available files for debugging
            if current_models_path.exists():
                files = list(current_models_path.glob("*.pkl"))
                st.info(f"📄 Available model files: {[f.name for f in files]}")
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

@st.cache_data
def load_results():
    """Load model comparison and test results"""
    results = {}
    try:
        # Get corrected paths
        _, _, current_json_path = get_corrected_paths()
        
        # Load test results
        test_results_path = current_json_path / "test_results.json"
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                results['test_results'] = json.load(f)
        
        # Load model comparison
        comparison_path = current_json_path / "model_comparison.json"
        if comparison_path.exists():
            with open(comparison_path, 'r') as f:
                results['comparison'] = json.load(f)
                
        # Load feature importance
        rf_importance_path = JSON_PATH / "random_forest_feature_importance.json"
        if rf_importance_path.exists():
            with open(rf_importance_path, 'r') as f:
                results['rf_importance'] = json.load(f)
                
        xgb_importance_path = JSON_PATH / "xgboost_feature_importance.json"
        if xgb_importance_path.exists():
            with open(xgb_importance_path, 'r') as f:
                results['xgb_importance'] = json.load(f)
                
        return results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return {}

def get_model_feature_importance(model):
    """Extract feature importance directly from a trained model"""
    try:
        if hasattr(model, 'feature_importances_'):
            # For Random Forest and XGBoost models
            importances = model.feature_importances_
            
            # Define the 18 training features
            feature_names = [
                'satisfaction_level_scaled', 'last_evaluation_scaled', 'number_project_scaled',
                'average_montly_hours_scaled', 'time_spend_company_scaled', 'Work_accident_scaled',
                'promotion_last_5years_scaled', 'department_IT', 'department_RandD', 'department_accounting',
                'department_marketing', 'department_product_mng', 'department_sales', 'department_support',
                'department_technical', 'salary_high', 'salary_low', 'salary_medium'
            ]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],  # Ensure we don't exceed the available features
                'Importance': importances
            })
            
            return importance_df
        else:
            # Model doesn't have feature_importances_ attribute
            st.warning("Selected model doesn't provide feature importance information.")
            return None
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return None

def predict_churn(model, input_data, model_name):
    """Make prediction using selected model"""
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Get confidence (max probability)
        confidence = np.max(probability)
        
        return {
            'prediction': 'Will Leave' if prediction == 1 else 'Will Stay',
            'probability_leave': probability[1],
            'probability_stay': probability[0],
            'confidence': confidence,
            'model_used': model_name
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_input_form():
    """Create input form for employee data"""
    st.subheader("📋 Employee Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_level = st.slider(
            "Satisfaction Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Employee satisfaction level (0-1)"
        )
        
        last_evaluation = st.slider(
            "Last Evaluation Score", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            help="Last performance evaluation score (0-1)"
        )
        
        number_project = st.selectbox(
            "Number of Projects", 
            options=[2, 3, 4, 5, 6, 7],
            index=2,
            help="Number of projects assigned"
        )
        
        average_monthly_hours = st.number_input(
            "Average Monthly Hours", 
            min_value=96, 
            max_value=310, 
            value=200,
            help="Average monthly working hours"
        )
    
    with col2:
        time_spend_company = st.selectbox(
            "Years at Company", 
            options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=2,
            help="Number of years spent at the company"
        )
        
        work_accident = st.selectbox(
            "Work Accident", 
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has the employee had a work accident?"
        )
        
        promotion_last_5years = st.selectbox(
            "Promotion in Last 5 Years", 
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has the employee been promoted in the last 5 years?"
        )
        
        department = st.selectbox(
            "Department", 
            options=['sales', 'technical', 'support', 'IT', 'product_mng', 
                    'marketing', 'RandD', 'accounting', 'hr', 'management'],
            help="Employee department"
        )
        
        salary = st.selectbox(
            "Salary Level", 
            options=['low', 'medium', 'high'],
            index=1,
            help="Salary level category"
        )
    
    return {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_monthly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'department': department,
        'salary': salary
    }

def prepare_input_data(input_dict, data_sample):
    """Convert input dictionary to model-ready format"""
    try:
        # Define the exact 18 feature columns used in training (excludes department_hr and department_management)
        training_features = [
            'satisfaction_level_scaled', 'last_evaluation_scaled', 'number_project_scaled',
            'average_montly_hours_scaled', 'time_spend_company_scaled', 'Work_accident_scaled',
            'promotion_last_5years_scaled', 'department_IT', 'department_RandD', 'department_accounting',
            'department_marketing', 'department_product_mng', 'department_sales', 'department_support', 
            'department_technical', 'salary_high', 'salary_low', 'salary_medium'
        ]
        
        # Create a copy of the first row as template
        input_row = data_sample.iloc[0:1].copy()
        
        # Calculate scaling parameters from the dataset (excluding the target column)
        train_data = data_sample.drop('quit', axis=1) if 'quit' in data_sample.columns else data_sample
        
        # For numerical features, calculate scaled values using the dataset statistics
        # This ensures consistency with the training data scaling
        
        # Scale satisfaction level
        satisfaction_mean = train_data['satisfaction_level'].mean()
        satisfaction_std = train_data['satisfaction_level'].std()
        scaled_satisfaction = (input_dict['satisfaction_level'] - satisfaction_mean) / satisfaction_std
        input_row['satisfaction_level_scaled'] = scaled_satisfaction
        
        # Scale last evaluation
        evaluation_mean = train_data['last_evaluation'].mean()
        evaluation_std = train_data['last_evaluation'].std()
        scaled_evaluation = (input_dict['last_evaluation'] - evaluation_mean) / evaluation_std
        input_row['last_evaluation_scaled'] = scaled_evaluation
        
        # Scale number of projects
        project_mean = train_data['number_project'].mean()
        project_std = train_data['number_project'].std()
        scaled_projects = (input_dict['number_project'] - project_mean) / project_std
        input_row['number_project_scaled'] = scaled_projects
        
        # Scale average monthly hours
        hours_mean = train_data['average_montly_hours'].mean()
        hours_std = train_data['average_montly_hours'].std()
        scaled_hours = (input_dict['average_montly_hours'] - hours_mean) / hours_std
        input_row['average_montly_hours_scaled'] = scaled_hours
        
        # Scale time spend company
        time_mean = train_data['time_spend_company'].mean()
        time_std = train_data['time_spend_company'].std()
        scaled_time = (input_dict['time_spend_company'] - time_mean) / time_std
        input_row['time_spend_company_scaled'] = scaled_time
        
        # Scale work accident (binary, but scaled in training)
        accident_mean = train_data['Work_accident'].mean()
        accident_std = train_data['Work_accident'].std()
        scaled_accident = (input_dict['Work_accident'] - accident_mean) / accident_std if accident_std > 0 else input_dict['Work_accident']
        input_row['Work_accident_scaled'] = scaled_accident
        
        # Scale promotion (binary, but scaled in training)
        promo_mean = train_data['promotion_last_5years'].mean()
        promo_std = train_data['promotion_last_5years'].std()
        scaled_promo = (input_dict['promotion_last_5years'] - promo_mean) / promo_std if promo_std > 0 else input_dict['promotion_last_5years']
        input_row['promotion_last_5years_scaled'] = scaled_promo
        
        # Reset all department columns to False
        dept_cols = [col for col in input_row.columns if col.startswith('department_')]
        for col in dept_cols:
            input_row[col] = False
        
        # Handle department mapping (hr and management were excluded from training)
        dept_mapping = {
            'hr': 'support',  # Map HR to support (similar administrative function)
            'management': 'support',  # Map management to support (similar administrative function)
        }
        
        # Use mapped department if needed
        selected_dept = dept_mapping.get(input_dict['department'], input_dict['department'])
        
        # Set the selected department to True (only if it exists in training features)
        dept_col = f"department_{selected_dept}"
        if dept_col in input_row.columns and dept_col.replace('department_', '') in [
            'IT', 'RandD', 'accounting', 'marketing', 'product_mng', 
            'sales', 'support', 'technical'
        ]:
            input_row[dept_col] = True
        
        # Reset all salary columns to False
        salary_cols = [col for col in input_row.columns if col.startswith('salary_')]
        for col in salary_cols:
            input_row[col] = False
        
        # Set the selected salary to True
        salary_col = f"salary_{input_dict['salary']}"
        if salary_col in input_row.columns:
            input_row[salary_col] = True
        
        # Select only the training features
        model_input = input_row[training_features]
        
        return model_input
    except Exception as e:
        st.error(f"Error preparing input data: {str(e)}")
        return None

def main_dashboard():
    """Main dashboard tab for predictions"""
    st.title("👥 Employee Churn Prediction Dashboard")
    st.markdown("---")
    
    # Load data and models
    data = load_data()
    models = load_models()
    
    if data is None or not models:
        st.error("Unable to load data or models. Please check file paths.")
        return
    
    # Model selection
    st.subheader("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose Prediction Model:", 
        options=list(models.keys()),
        help="Select which model to use for prediction"
    )
    
    # Input form
    input_data = create_input_form()
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Employee Churn", type="primary", use_container_width=True):
            # Prepare input data
            processed_input = prepare_input_data(input_data, data)
            
            if processed_input is not None:
                # Make prediction
                result = predict_churn(models[selected_model], processed_input, selected_model)
                
                if result:
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    # Create result display
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        # Prediction result
                        color = "red" if result['prediction'] == 'Will Leave' else "green"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border-radius: 10px; 
                                    background-color: {'#ffebee' if color == 'red' else '#e8f5e8'}">
                            <h2 style="color: {color}; margin: 0;">
                                {result['prediction']}
                            </h2>
                            <p style="color: gray; margin: 5px 0;">
                                Model: {result['model_used']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = result['probability_leave'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.subheader("📈 Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Outcome': ['Will Stay', 'Will Leave'],
                        'Probability': [result['probability_stay'], result['probability_leave']],
                        'Percentage': [f"{result['probability_stay']*100:.1f}%", 
                                     f"{result['probability_leave']*100:.1f}%"]
                    })
                    
                    st.dataframe(prob_df, use_container_width=True)

def visualization_tab():
    """Visualization tab for data exploration"""
    st.title("📊 Data Visualization & Insights")
    st.markdown("---")
    
    # Load data, models, and results
    data = load_data()
    models = load_models()
    results = load_results()
    
    if data is None:
        st.error("Unable to load data.")
        return
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns) - 1)  # Excluding target
    with col3:
        churn_rate = data['quit'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Retention Rate", f"{100-churn_rate:.1f}%")
    
    # Churn distribution
    st.subheader("🎯 Churn Distribution")
    churn_counts = data['quit'].value_counts()
    
    fig = px.pie(
        values=churn_counts.values, 
        names=['Stayed', 'Left'],
        title="Employee Churn Distribution",
        color_discrete_map={'Stayed': '#2E8B57', 'Left': '#DC143C'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    
    # Select features for visualization
    numeric_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                       'average_montly_hours', 'time_spend_company']
    
    selected_feature = st.selectbox(
        "Select feature to visualize:", 
        numeric_features,
        help="Choose a feature to see its distribution by churn status"
    )
    
    # Create distribution plot
    fig = px.histogram(
        data, 
        x=selected_feature, 
        color='quit',
        nbins=30,
        title=f"{selected_feature.replace('_', ' ').title()} Distribution by Churn Status",
        labels={'quit': 'Churn Status'},
        color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
    )
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    
    # Select numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    if 'rf_importance' in results or 'xgb_importance' in results:
        st.subheader("⭐ Feature Importance")
        
        importance_model = st.selectbox(
            "Select model for feature importance:",
            options=['Random Forest', 'XGBoost']
        )
        
        if importance_model == 'Random Forest' and 'rf_importance' in results:
            importance_data = results['rf_importance']
        elif importance_model == 'XGBoost' and 'xgb_importance' in results:
            importance_data = results['xgb_importance']
        else:
            importance_data = None
        
        if importance_data:
            # Handle different data formats
            if isinstance(importance_data, dict):
                # If it's a dictionary, convert directly
                importance_df = pd.DataFrame(list(importance_data.items()), 
                                           columns=['Feature', 'Importance'])
            elif isinstance(importance_data, list):
                # If it's a list, check if it contains tuples or dictionaries
                if importance_data and isinstance(importance_data[0], (list, tuple)):
                    # List of [feature, importance] pairs
                    importance_df = pd.DataFrame(importance_data, columns=['Feature', 'Importance'])
                elif importance_data and isinstance(importance_data[0], dict):
                    # List of dictionaries with feature/importance keys
                    importance_df = pd.DataFrame(importance_data)
                else:
                    # Unknown format, create mock data from model if available
                    if models and importance_model in models:
                        st.warning("⚠️ Feature importance data format not recognized. Using model's current feature importance.")
                        importance_df = get_model_feature_importance(models[importance_model])
                    else:
                        st.warning("⚠️ Feature importance data not available and models not loaded.")
                        importance_df = None
            else:
                # Unknown format, create mock data from model if available
                if models and importance_model in models:
                    st.warning("⚠️ Feature importance data format not recognized. Using model's current feature importance.")
                    importance_df = get_model_feature_importance(models[importance_model])
                else:
                    st.warning("⚠️ Feature importance data not available and models not loaded.")
                    importance_df = None
        else:
            # No importance data available, try to get from model
            if models and importance_model in models:
                st.info("ℹ️ Loading feature importance directly from model...")
                importance_df = get_model_feature_importance(models[importance_model])
            else:
                st.warning("⚠️ Feature importance data not available.")
                importance_df = None
            
            if importance_df is not None and not importance_df.empty:
                importance_df = importance_df.sort_values('Importance', ascending=True).tail(15)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"{importance_model} - Top 15 Feature Importances",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def model_comparison_tab():
    """Model comparison tab"""
    st.title("⚖️ Model Performance Comparison")
    st.markdown("---")
    
    # Load results
    results = load_results()
    
    if 'test_results' not in results:
        st.error("Unable to load model results.")
        return
    
    test_results = results['test_results']
    
    # Performance metrics comparison
    st.subheader("📊 Performance Metrics Comparison")
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(test_results).T
    metrics_df.index.name = 'Model'
    metrics_df = metrics_df.reset_index()
    metrics_df['Model'] = metrics_df['Model'].str.replace('_', ' ').str.title()
    
    # Display metrics table
    st.dataframe(
        metrics_df.round(4),
        use_container_width=True,
        column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
        }
    )
    
    # Radar chart for model comparison
    st.subheader("🎯 Performance Radar Chart")
    
    fig = go.Figure()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    for model, data in test_results.items():
        model_name = model.replace('_', ' ').title()
        values = [data[metric] for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.9, 1.0]
            )),
        showlegend=True,
        title="Model Performance Comparison (Radar Chart)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metric comparison bars
    st.subheader("📈 Detailed Metric Comparison")
    
    selected_metric = st.selectbox(
        "Select metric to compare:",
        options=metrics,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    metric_data = {model.replace('_', ' ').title(): data[selected_metric] 
                   for model, data in test_results.items()}
    
    fig = px.bar(
        x=list(metric_data.keys()),
        y=list(metric_data.values()),
        title=f"{selected_metric.replace('_', ' ').title()} Comparison",
        color=list(metric_data.values()),
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=selected_metric.replace('_', ' ').title()
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model ranking and comprehensive comparison
    if 'comparison' in results and 'test_results' in results:
        st.subheader("🏆 Model Ranking & Detailed Comparison")
        
        # Create comprehensive comparison
        comparison_data = []
        test_results = results['test_results']
        
        for model_info in results['comparison']:
            model_name = model_info['model'].replace('_', ' ').title()
            model_key = model_info['model']
            
            # Get test results for this model
            if model_key in test_results:
                test_metrics = test_results[model_key]
                
                # Calculate average performance score (excluding ROC-AUC for balance)
                avg_score = (test_metrics['accuracy'] + test_metrics['precision'] + 
                           test_metrics['recall'] + test_metrics['f1_score']) / 4
                
                comparison_data.append({
                    'Model': model_name,
                    'CV Score': model_info['cv_score'],
                    'Accuracy': test_metrics['accuracy'],
                    'Precision': test_metrics['precision'],
                    'Recall': test_metrics['recall'],
                    'F1-Score': test_metrics['f1_score'],
                    'ROC-AUC': test_metrics['roc_auc'],
                    'Avg Performance': avg_score,
                    'CV Rank': model_info['rank']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Add overall ranking based on average performance
            comparison_df['Overall Rank'] = comparison_df['Avg Performance'].rank(ascending=False, method='min').astype(int)
            
            # Reorder columns for better display
            column_order = ['Model', 'Overall Rank', 'Avg Performance', 'CV Rank', 'CV Score', 
                          'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            comparison_df = comparison_df[column_order]
            
            st.dataframe(
                comparison_df,
                use_container_width=True,
                column_config={
                    "CV Score": st.column_config.NumberColumn("CV Score", format="%.4f"),
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                    "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.3f"),
                    "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                    "Avg Performance": st.column_config.NumberColumn("Avg Performance", format="%.4f"),
                    "Overall Rank": st.column_config.NumberColumn("Overall Rank"),
                    "CV Rank": st.column_config.NumberColumn("CV Rank"),
                }
            )
            
            # Add explanation
            st.info("""
            **📊 Ranking Explanation:**
            - **Overall Rank**: Based on average of Accuracy, Precision, Recall, and F1-Score
            - **CV Rank**: Based on Cross-Validation Score during training
            - Both rankings are valuable - CV Score shows training performance, while Overall Rank shows test performance
            """)
    elif 'comparison' in results:
        # Fallback to original ranking if test results not available
        st.subheader("🏆 Model Ranking (Cross-Validation Based)")
        comparison_df = pd.DataFrame(results['comparison'])
        comparison_df['model'] = comparison_df['model'].str.replace('_', ' ').str.title()
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config={
                "cv_score": st.column_config.NumberColumn("CV Score", format="%.4f"),
                "rank": st.column_config.NumberColumn("Rank"),
            }
        )
        st.info("💡 This ranking is based on Cross-Validation scores during model training.")

def about_tab():
    """About/Info tab"""
    st.title("ℹ️ About This Application")
    st.markdown("---")
    
    # Project overview
    st.subheader("🎯 Project Overview")
    st.markdown("""
    This **Employee Churn Prediction System** is an advanced machine learning application 
    designed to predict whether an employee will leave the company based on various 
    employment-related factors.
    
    The system uses two powerful machine learning algorithms:
    - **Random Forest**: An ensemble method that builds multiple decision trees
    - **XGBoost**: An optimized gradient boosting framework
    """)
    
    # Dataset information
    st.subheader("📊 Dataset Information")
    data = load_data()
    
    if data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dataset Statistics:**
            - **Total Records**: {:,}
            - **Features**: {}
            - **Target**: Employee Quit (Binary)
            - **Churn Rate**: {:.1f}%
            """.format(
                len(data), 
                len(data.columns) - 1,
                data['quit'].mean() * 100
            ))
        
        with col2:
            st.markdown("""
            **Key Features:**
            - Satisfaction Level
            - Last Evaluation Score
            - Number of Projects
            - Average Monthly Hours
            - Time Spent at Company
            - Work Accident History
            - Promotion History
            - Department
            - Salary Level
            """)
    
    # Preprocessing steps
    st.subheader("🔧 Data Preprocessing")
    st.markdown("""
    The dataset underwent comprehensive preprocessing:
    
    1. **Feature Engineering**: Created scaled versions of numerical features
    2. **Encoding**: One-hot encoding for categorical variables (department, salary)
    3. **Scaling**: Standardized numerical features for better model performance
    4. **Data Quality**: Handled missing values and outliers
    5. **Feature Selection**: Retained most predictive features
    """)
    
    # Model information
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Random Forest**
        - **Type**: Ensemble Learning
        - **Algorithm**: Bagging with Decision Trees
        - **Strengths**: 
          - High accuracy
          - Handles overfitting well
          - Feature importance ranking
        - **Use Case**: Stable, interpretable predictions
        """)
    
    with col2:
        st.markdown("""
        **XGBoost**
        - **Type**: Gradient Boosting
        - **Algorithm**: Optimized Gradient Boosting
        - **Strengths**:
          - High performance
          - Built-in regularization
          - Handles missing values
        - **Use Case**: Maximum predictive accuracy
        """)
    
    # Technical specifications
    st.subheader("⚙️ Technical Specifications")
    st.markdown("""
    **Technologies Used:**
    - **Frontend**: Streamlit
    - **ML Framework**: Scikit-learn, XGBoost
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    - **Model Persistence**: Joblib
    
    **Model Training:**
    - **Cross-Validation**: 5-fold StratifiedKFold
    - **Hyperparameter Optimization**: RandomizedSearchCV
    - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - **Regularization**: Applied to prevent overfitting
    """)
    
    # Performance summary
    results = load_results()
    if 'test_results' in results:
        st.subheader("🏆 Model Performance Summary")
        
        test_results = results['test_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'random_forest' in test_results:
                rf_results = test_results['random_forest']
                st.markdown(f"""
                **Random Forest Performance:**
                - **Accuracy**: {rf_results['accuracy']:.3f}
                - **Precision**: {rf_results['precision']:.3f}
                - **Recall**: {rf_results['recall']:.3f}
                - **F1-Score**: {rf_results['f1_score']:.3f}
                - **ROC-AUC**: {rf_results['roc_auc']:.3f}
                """)
        
        with col2:
            if 'xgboost' in test_results:
                xgb_results = test_results['xgboost']
                st.markdown(f"""
                **XGBoost Performance:**
                - **Accuracy**: {xgb_results['accuracy']:.3f}
                - **Precision**: {xgb_results['precision']:.3f}
                - **Recall**: {xgb_results['recall']:.3f}
                - **F1-Score**: {xgb_results['f1_score']:.3f}
                - **ROC-AUC**: {xgb_results['roc_auc']:.3f}
                """)
    
    # Developer information
    st.subheader("👨‍💻 Developer Information")
    st.markdown("""
    **Author**: Sharawey74  
    **Project**: AI Project - Employee Turnover Prediction  
    **Version**: 2.0.0  
    **Date**: August 2025  
    **Repository**: [GitHub - AI-Project](https://github.com/Sharawey74/AI-Project)
    
    This application is part of a comprehensive machine learning pipeline designed 
    to help organizations predict and prevent employee churn through data-driven insights.
    """)

def main():
    """Main application function"""
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    st.sidebar.markdown("---")
    
    # Tab selection
    tab_selection = st.sidebar.radio(
        "Choose a section:",
        options=[
            "🏠 Main Dashboard",
            "📊 Visualization", 
            "⚖️ Model Comparison",
            "ℹ️ About"
        ]
    )
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 Quick Help
    
    **Main Dashboard**: Make predictions for individual employees
    
    **Visualization**: Explore dataset patterns and insights
    
    **Model Comparison**: Compare Random Forest vs XGBoost performance
    
    **About**: Learn about the project and models
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version**: 2.0.0")
    st.sidebar.markdown("**Author**: Sharawey74")
    
    # Route to selected tab
    if tab_selection == "🏠 Main Dashboard":
        main_dashboard()
    elif tab_selection == "📊 Visualization":
        visualization_tab()
    elif tab_selection == "⚖️ Model Comparison":
        model_comparison_tab()
    elif tab_selection == "ℹ️ About":
        about_tab()

if __name__ == "__main__":
    main()
