#!/usr/bin/env python3
"""
Quick test to verify the scoring fix
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from Testing/ to project root
sys.path.append(str(PROJECT_ROOT))

try:
    print("🧪 Testing scoring fix...")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Set up model and parameters
    model = RandomForestClassifier(random_state=42)
    param_distributions = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, None]
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Test with string scorer (new approach)
    print("Testing with 'roc_auc' string scorer...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=5,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        verbose=0
    )
    
    search.fit(X, y)
    print(f"✅ Scoring fix works! Best score: {search.best_score_:.3f}")
    print(f"✅ Best params: {search.best_params_}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
