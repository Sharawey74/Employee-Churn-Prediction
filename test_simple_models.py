#!/usr/bin/env python3
"""
Simple test to verify models can train with encoded data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

try:
    print("🧪 Testing models with properly encoded data...")
    
    # Load and prepare data
    from src.feature_engineering import FeatureEngineer
    from src.data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.load_raw_data("data/raw/employee_data.csv")
    print(f"✅ Data loaded: {data.shape}")
    
    # Apply feature engineering
    fe = FeatureEngineer()
    X, y = fe.prepare_features_and_target(data, 'quit')
    print(f"✅ Features prepared: {X.shape}")
    
    # Check for categorical data
    categorical_remaining = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_remaining) == 0:
        print("✅ All categorical data properly encoded!")
    else:
        print(f"❌ Still have categorical columns: {categorical_remaining}")
        sys.exit(1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split: Train({X_train.shape[0]}), Test({X_test.shape[0]})")
    
    # Test simple Random Forest
    print("\n🌲 Testing Random Forest...")
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"✅ Random Forest Accuracy: {acc_rf:.3f}")
    
    # Test simple Logistic Regression
    print("\n📊 Testing Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"✅ Logistic Regression Accuracy: {acc_lr:.3f}")
    
    print(f"\n🎉 SUCCESS! Both models trained successfully!")
    print(f"📊 Random Forest: {acc_rf:.3f} accuracy")
    print(f"📊 Logistic Regression: {acc_lr:.3f} accuracy")
    print("\n✅ The categorical encoding fix is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
