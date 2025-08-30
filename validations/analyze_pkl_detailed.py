#!/usr/bin/env python3
"""
Detailed PKL File Analysis
Analyzes .pkl files in the models directory with comprehensive information
"""

import pickle
import joblib
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def analyze_pkl_file_with_joblib(file_path):
    """Analyze a .pkl file trying both pickle and joblib"""
    try:
        # Try pickle first
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        loader = "pickle"
    except:
        try:
            # Try joblib
            data = joblib.load(file_path)
            loader = "joblib"
        except Exception as e:
            return {'file_name': os.path.basename(file_path), 'error': str(e)}, None, None
    
    file_info = {
        'file_name': os.path.basename(file_path),
        'file_path': file_path,
        'data_type': type(data).__name__,
        'size_bytes': os.path.getsize(file_path),
        'loader_used': loader
    }
    
    # Get more specific information based on data type
    if hasattr(data, '__class__'):
        file_info['class_name'] = data.__class__.__name__
        file_info['module'] = data.__class__.__module__
    
    # For sklearn models
    if hasattr(data, 'get_params'):
        try:
            file_info['model_params'] = data.get_params()
        except:
            file_info['model_params'] = "Unable to get parameters"
    
    # For dictionaries
    if isinstance(data, dict):
        file_info['dict_keys'] = list(data.keys())
        file_info['dict_size'] = len(data)
        # Show some sample values for small dicts
        if len(data) <= 10:
            file_info['sample_values'] = {k: str(v)[:100] for k, v in list(data.items())[:3]}
    
    # For DataFrames
    if isinstance(data, pd.DataFrame):
        file_info['shape'] = data.shape
        file_info['columns'] = list(data.columns)
    
    # For lists/arrays
    if isinstance(data, (list, tuple)):
        file_info['length'] = len(data)
    
    return file_info, data, loader

def main():
    """Main analysis function"""
    models_dir = PROJECT_ROOT / "models"
    pkl_files = []
    
    # Find all .pkl files
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    
    print("=== DETAILED PKL FILE ANALYSIS ===")
    print(f"Found {len(pkl_files)} .pkl files\n")
    
    for pkl_file in sorted(pkl_files):
        print(f"{'='*60}")
        print(f"FILE: {os.path.basename(pkl_file)}")
        print(f"PATH: {pkl_file}")
        
        info, data, loader = analyze_pkl_file_with_joblib(pkl_file)
        
        if 'error' in info:
            print(f"ERROR: {info['error']}")
            continue
        
        print(f"Loaded with: {info['loader_used']}")
        print(f"Data Type: {info['data_type']}")
        print(f"File Size: {info['size_bytes']:,} bytes")
        
        if 'class_name' in info:
            print(f"Class: {info['class_name']}")
            print(f"Module: {info['module']}")
        
        if 'model_params' in info and isinstance(info['model_params'], dict):
            print("Key Model Parameters:")
            important_params = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'gamma', 'kernel']
            for param in important_params:
                if param in info['model_params']:
                    print(f"  {param}: {info['model_params'][param]}")
        
        if 'dict_keys' in info:
            print(f"Dictionary Keys: {info['dict_keys']}")
            if 'sample_values' in info:
                print("Sample Values:")
                for k, v in info['sample_values'].items():
                    print(f"  {k}: {v}")
        
        if 'shape' in info:
            print(f"DataFrame Shape: {info['shape']}")
            print(f"Columns: {info['columns']}")
        
        if 'length' in info:
            print(f"Length: {info['length']}")
        
        print()

if __name__ == "__main__":
    main()
