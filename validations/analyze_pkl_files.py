#!/usr/bin/env python3
"""
PKL File Analysis
Basic analysis of .pkl files in the models directory
"""

import pickle
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def analyze_pkl_file(file_path):
    """Analyze a single .pkl file and return its contents info"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        file_info = {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'data_type': type(data).__name__,
            'size_bytes': os.path.getsize(file_path)
        }
        
        # Get more specific information based on data type
        if hasattr(data, '__class__'):
            file_info['class_name'] = data.__class__.__name__
            file_info['module'] = data.__class__.__module__
        
        # For sklearn models
        if hasattr(data, 'get_params'):
            file_info['model_params'] = data.get_params()
        
        # For dictionaries
        if isinstance(data, dict):
            file_info['dict_keys'] = list(data.keys())
            file_info['dict_size'] = len(data)
        
        # For DataFrames
        if isinstance(data, pd.DataFrame):
            file_info['shape'] = data.shape
            file_info['columns'] = list(data.columns)
        
        # For lists/arrays
        if isinstance(data, (list, tuple)):
            file_info['length'] = len(data)
        
        return file_info, data
    
    except Exception as e:
        return {'file_name': os.path.basename(file_path), 'error': str(e)}, None

def main():
    """Main analysis function"""
    models_dir = PROJECT_ROOT / "models"
    pkl_files = []
    
    # Find all .pkl files
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    
    print("=== PKL FILE ANALYSIS ===")
    print(f"Found {len(pkl_files)} .pkl files\n")
    
    for pkl_file in sorted(pkl_files):
        print(f"{'='*50}")
        print(f"FILE: {os.path.basename(pkl_file)}")
        print(f"PATH: {pkl_file}")
        
        info, data = analyze_pkl_file(pkl_file)
        
        if 'error' in info:
            print(f"ERROR: {info['error']}")
            continue
        
        print(f"Data Type: {info['data_type']}")
        print(f"File Size: {info['size_bytes']:,} bytes")
        
        if 'class_name' in info:
            print(f"Class: {info['class_name']}")
            print(f"Module: {info['module']}")
        
        if 'model_params' in info:
            print("Model Parameters:")
            for key, value in list(info['model_params'].items())[:5]:  # Show first 5 params
                print(f"  {key}: {value}")
            if len(info['model_params']) > 5:
                print(f"  ... and {len(info['model_params']) - 5} more parameters")
        
        if 'dict_keys' in info:
            print(f"Dictionary Keys: {info['dict_keys']}")
        
        if 'shape' in info:
            print(f"DataFrame Shape: {info['shape']}")
            print(f"Columns: {info['columns']}")
        
        if 'length' in info:
            print(f"Length: {info['length']}")
        
        print()

if __name__ == "__main__":
    main()
