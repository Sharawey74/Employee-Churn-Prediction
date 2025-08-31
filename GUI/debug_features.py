#!/usr/bin/env python3
"""Debug the feature preparation process"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Load data and models
print("Loading data and models...")
DATA_PATH = parent_dir / "data" / "processed" / "feature_engineered_data.csv"
MODELS_PATH = parent_dir / "models"

data = pd.read_csv(DATA_PATH)
rf_model = joblib.load(MODELS_PATH / "random_forest_model.pkl")

print(f"Data shape: {data.shape}")
print(f"RF model expects: {rf_model.n_features_in_} features")

# Test the prepare_input_data function
from app import prepare_input_data

test_input = {
    'satisfaction_level': 0.38,
    'last_evaluation': 0.53,
    'number_project': 2,
    'average_montly_hours': 157,
    'time_spend_company': 3,
    'Work_accident': 0,
    'promotion_last_5years': 0,
    'department': 'sales',
    'salary': 'low'
}

print("\nTesting prepare_input_data function...")
model_input = prepare_input_data(test_input, data)

if model_input is not None:
    print(f"Model input shape: {model_input.shape}")
    print(f"Expected: (1, 18), Got: {model_input.shape}")
    print(f"Model input columns ({len(model_input.columns)}):")
    for i, col in enumerate(model_input.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Test prediction
    try:
        prediction = rf_model.predict(model_input)
        probabilities = rf_model.predict_proba(model_input)
        print(f"\n✅ Prediction works!")
        print(f"   Prediction: {prediction[0]}")
        print(f"   Probabilities: Stay={probabilities[0][0]:.3f}, Leave={probabilities[0][1]:.3f}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
else:
    print("❌ prepare_input_data returned None")
