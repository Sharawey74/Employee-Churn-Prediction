#!/usr/bin/env python3
"""
Setup script for AI Project - Employee Turnover Prediction
Usage: 
    python setup.py install          # Install the package
    python setup.py develop          # Install in development mode
    python setup.py --help           # Show help
    pip install -e .                 # Recommended: Install in editable mode
"""

import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read requirements from requirements.txt if it exists
def read_requirements():
    """Read requirements from requirements.txt if it exists"""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Use requirements from file or fallback to default list
requirements = read_requirements() or [
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.1.0",
    "imbalanced-learn>=0.9.0",
    "xgboost>=1.6.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.10.0",
    "jupyterlab>=3.4.0",
    "ipywidgets>=8.0.0",
    "tqdm>=4.64.0",
    "joblib>=1.1.0",
    "optuna>=3.0.0",
    "shap>=0.41.0",
]

# Check if running without commands
if len(sys.argv) == 1:
    print("\n🚀 AI Project - Employee Turnover Prediction Setup")
    print("=" * 55)
    print("\nUsage options:")
    print("  pip install -e .                    # Recommended: Editable install")
    print("  python setup.py install            # Standard install")
    print("  python setup.py develop            # Development install")
    print("  python setup.py --help             # Show all options")
    print("\nProject structure:")
    print("  📁 main/         - Training scripts")
    print("  📁 src/          - Core modules") 
    print("  📁 validations/  - Validation scripts")
    print("  📁 data/         - Datasets")
    print("  📁 models/       - Trained models")
    print("  📁 results/      - Training results")
    print("\nFor quick start:")
    print("  python main/rf_xgb_trainer.py      # Train RF & XGBoost models")
    print("  python -m validations.debug_paths  # Check project structure")
    print("")
    sys.exit(0)

setup(
    name="ai-project-employee-turnover",
    version="2.0.0",
    description="Employee Turnover Prediction using Random Forest and XGBoost",
    long_description="Machine Learning pipeline for employee turnover prediction, optimized for Random Forest and XGBoost algorithms with comprehensive evaluation and hyperparameter optimization.",
    author="Sharawey74",
    author_email="",
    url="https://github.com/Sharawey74/AI-Project",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        'console_scripts': [
            'ai-project-train=main.rf_xgb_trainer:main',
            'ai-project-full=main.main:main',
            'ai-project-validate=validations.validate_restructure:run_all_tests',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
        ],
        'notebook': [
            'notebook>=6.4.0',
            'ipykernel>=6.15.0',
        ],
    },
)