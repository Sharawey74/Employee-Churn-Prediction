#!/usr/bin/env python3
"""
Debug script to check path issues
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Absolute path: {PROJECT_ROOT.absolute()}")

required_dirs = [
    "data/raw",
    "data/processed", 
    "models",
    "results",
    "json",
    "src",
    "main"
]

for dir_path in required_dirs:
    full_path = PROJECT_ROOT / dir_path
    exists = full_path.exists()
    print(f"{dir_path}: {exists} (Path: {full_path.absolute()})")

key_files = [
    "src/config.py",
    "src/model_trainer.py",
    "main/main.py",
    "main/rf_xgb_trainer.py",
    "README_RF_XGB.md"
]

for file_path in key_files:
    full_path = PROJECT_ROOT / file_path
    exists = full_path.exists()
    print(f"{file_path}: {exists} (Path: {full_path.absolute()})")
