#!/usr/bin/env python3
"""
Cross-Validation Implementation Analysis
Analyzes CV configuration and implementation in the project
"""

import pickle
import joblib
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def analyze_cv_implementation():
    """Analyze cross-validation implementation in the project"""
    
    print("=== CROSS-VALIDATION IMPLEMENTATION ANALYSIS ===")
    
    # Load the evaluation data to see the CV implementation
    try:
        evaluation_data_path = PROJECT_ROOT / "models" / "evaluation_data.pkl"
        evaluation_data = joblib.load(evaluation_data_path)
        
        print("\nEvaluation data keys:", list(evaluation_data.keys()))
        
        if 'X_train' in evaluation_data:
            print(f"Training set size: {evaluation_data['X_train'].shape}")
        if 'X_test' in evaluation_data:
            print(f"Test set size: {evaluation_data['X_test'].shape}")
            
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
    
    print("\n=== CV CONFIGURATION FROM CONFIG ===")
    try:
        config_path = PROJECT_ROOT / "src" / "config.py"
        with open(config_path, 'r') as f:
            config_content = f.read()

        # Extract CV config section
        cv_section_start = config_content.find("# Cross-Validation Configuration")
        cv_section_end = config_content.find("# Evaluation Metrics")
        if cv_section_start != -1 and cv_section_end != -1:
            cv_section = config_content[cv_section_start:cv_section_end]
            print(cv_section)
        else:
            print("CV configuration section not found in config.py")
            # Print entire config for debugging
            print("Full config content:")
            print(config_content[:1000] + "..." if len(config_content) > 1000 else config_content)
            
    except Exception as e:
        print(f"Error reading config: {e}")

if __name__ == "__main__":
    analyze_cv_implementation()
