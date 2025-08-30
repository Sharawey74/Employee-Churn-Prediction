#!/usr/bin/env python3
"""
Regularization Parameter Analysis
Analyzes regularization techniques used in the trained models
"""

import pickle
import json
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def analyze_regularization():
    """Analyze regularization parameters in best_parameters.pkl"""
    
    best_params_path = PROJECT_ROOT / "models" / "best_parameters.pkl"
    
    try:
        # Load and examine the best parameters
        with open(best_params_path, 'rb') as f:
            best_params = pickle.load(f)

        print("=== BEST PARAMETERS ANALYSIS ===")
        print(json.dumps(best_params, indent=2))

        # Check for regularization parameters
        print("\n=== REGULARIZATION ANALYSIS ===")
        for model_name, params in best_params.items():
            print(f"\n{model_name.upper()}:")
            regularization_found = False
            
            for param_name, param_value in params.items():
                # Check for regularization-related parameters
                if any(reg_term in param_name.lower() for reg_term in ['c', 'penalty', 'reg_alpha', 'reg_lambda', 'l1_ratio', 'alpha']):
                    print(f"  🔧 {param_name}: {param_value} (Regularization)")
                    regularization_found = True
                elif param_name in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample']:
                    print(f"  🌲 {param_name}: {param_value} (Tree Regularization)")
                    regularization_found = True
                else:
                    print(f"  ⚙️  {param_name}: {param_value}")
            
            if not regularization_found:
                print("  ❌ No explicit regularization parameters found")
                
    except Exception as e:
        print(f"Error analyzing regularization: {e}")

if __name__ == "__main__":
    analyze_regularization()
