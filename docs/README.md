# Employee Turnover Prediction - Project Documentation

## Overview

This project implements a complete machine learning pipeline for predicting employee turnover using Decision Trees and Random Forests. The implementation follows all 9 tasks outlined in the project requirements and demonstrates software engineering best practices.

## Project Structure

```
AI-Project/
├── README.md                      # Original project description
├── config.py                      # Project configuration
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation setup
├── main.py                       # Main execution script
├── interactive_demo.py           # Interactive demonstration
├── .gitignore                    # Git ignore rules
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── feature_encoder.py       # Categorical feature encoding
│   ├── visualizer.py            # Visualization functions
│   └── models/                  # Machine learning models
│       ├── __init__.py
│       ├── decision_tree.py     # Decision Tree implementation
│       ├── random_forest.py     # Random Forest implementation
│       └── evaluator.py         # Model evaluation and metrics
│
├── data/                        # Data directory
│   ├── raw/                     # Original datasets
│   │   └── employee_data.csv    # Employee dataset
│   ├── processed/               # Cleaned and encoded data
│   └── sample_data.py          # Sample data generator
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── constants.py            # Project constants
│   └── helpers.py              # Helper functions
│
├── docs/                       # Documentation
├── tests/                      # Unit tests
└── outputs/                    # Generated outputs and plots
```

## Features

### Core Functionality

1. **Data Loading and Preprocessing** (`src/data_loader.py`)
   - Robust data loading with validation
   - Data quality checks and statistics
   - Sample data generation capabilities

2. **Feature Encoding** (`src/feature_encoder.py`)
   - Categorical variable encoding using one-hot encoding
   - Handles department and salary categories
   - Reversible encoding with proper validation

3. **Data Visualization** (`src/visualizer.py`)
   - Comprehensive exploratory data analysis
   - Class imbalance visualization using Yellowbrick
   - Feature importance plots and model comparisons
   - Interactive dashboards

4. **Decision Tree Model** (`src/models/decision_tree.py`)
   - Full Decision Tree implementation
   - Interactive parameter tuning with ipywidgets
   - Tree visualization and text representation
   - Comprehensive model evaluation

5. **Random Forest Model** (`src/models/random_forest.py`)
   - Random Forest classifier implementation
   - Interactive parameter optimization
   - Individual tree analysis and visualization
   - Feature importance comparison across trees

6. **Model Evaluation** (`src/models/evaluator.py`)
   - Comprehensive evaluation metrics
   - ROC curves and Precision-Recall curves
   - Learning curves and validation curves
   - Model comparison visualizations

### Interactive Features

- **Real-time Parameter Tuning**: Interactive widgets for adjusting model parameters
- **Live Visualizations**: Immediate feedback on model performance
- **Feature Exploration**: Interactive feature importance analysis
- **Model Comparison**: Side-by-side model performance comparison

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package (optional):**
   ```bash
   pip install -e .
   ```

## Usage

### Quick Start

Run the complete pipeline with all 9 tasks:

```bash
python main.py
```

This will:
- Load and analyze the employee dataset
- Perform exploratory data analysis
- Encode categorical features
- Visualize class imbalance
- Split data into train/validation/test sets
- Train Decision Tree and Random Forest models
- Generate comprehensive evaluation reports
- Create feature importance plots
- Save all visualizations to the outputs/ directory

### Interactive Demo

For interactive model building and parameter tuning:

```bash
# Menu-driven interface
python interactive_demo.py

# Full interactive demo
python interactive_demo.py --full

# Menu-driven demo
python interactive_demo.py --menu
```

### Individual Components

You can also use individual components:

```python
from src.data_loader import DataLoader
from src.models.decision_tree import DecisionTreeModel
from src.visualizer import DataVisualizer

# Load data
loader = DataLoader()
data = loader.load_data()

# Create visualizations
visualizer = DataVisualizer()
figures = visualizer.exploratory_data_analysis(data)

# Train model
model = DecisionTreeModel()
results = model.train_model(X_train, y_train, X_val, y_val)
```

## The 9 Tasks Implementation

### Task 1: Import Libraries
- ✅ Implemented in module imports across all files
- Essential modules: NumPy, Pandas, Matplotlib, scikit-learn, Yellowbrick

### Task 2: Exploratory Data Analysis
- ✅ Implemented in `src/visualizer.py`
- Comprehensive EDA with multiple visualization types
- Feature distribution analysis by target variable

### Task 3: Encode Categorical Features
- ✅ Implemented in `src/feature_encoder.py`
- One-hot encoding for department and salary variables
- Robust encoding with validation and reverse transformation

### Task 4: Visualize Class Imbalance
- ✅ Implemented in `src/visualizer.py`
- Yellowbrick Class Balance visualizer
- Custom frequency plots with statistics

### Task 5: Create Training and Validation Sets
- ✅ Implemented in `main.py` and `interactive_demo.py`
- 80/20 training/validation split with stratified sampling
- Additional test set for final evaluation

### Tasks 6 & 7: Decision Tree with Interactive Controls
- ✅ Implemented in `src/models/decision_tree.py`
- Interactive parameter tuning with ipywidgets
- Real-time accuracy calculation and tree visualization
- Comprehensive model analysis

### Task 8: Random Forest with Interactive Controls
- ✅ Implemented in `src/models/random_forest.py`
- Interactive Random Forest building
- Individual tree analysis and ensemble statistics
- Feature importance comparison across trees

### Task 9: Feature Importance and Evaluation Metrics
- ✅ Implemented in `src/models/evaluator.py`
- Feature importance ranking and visualization
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, AUC)
- Model comparison and performance analysis

## Key Features

### Robustness
- Comprehensive error handling and logging
- Data validation at multiple stages
- Fallback options for missing dependencies

### Modularity
- Clean separation of concerns
- Reusable components
- Easy to extend and modify

### Interactivity
- ipywidgets integration for Jupyter notebooks
- Real-time parameter adjustment
- Interactive visualizations

### Visualization
- Professional matplotlib and seaborn plots
- Yellowbrick integration for ML-specific visualizations
- Comprehensive dashboards and reports

### Documentation
- Detailed docstrings with type hints
- Comprehensive examples and usage guides
- Clear code structure and comments

## Performance Optimization

- Efficient data loading and processing
- Parallel processing for Random Forest (n_jobs=-1)
- Optimized visualization rendering
- Memory-efficient data handling

## Extension Points

The project is designed to be easily extensible:

1. **Add New Models**: Implement new model classes following the same interface
2. **Custom Visualizations**: Extend the visualizer with new plot types
3. **Additional Features**: Add new feature engineering techniques
4. **Advanced Evaluation**: Implement additional evaluation metrics
5. **Data Sources**: Support for different data formats and sources

## Troubleshooting

### Common Issues

1. **Data not found**: Ensure `employee_data.csv` is in `data/raw/` directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Interactive widgets not working**: Install ipywidgets: `pip install ipywidgets`
4. **Plots not displaying**: Ensure matplotlib backend is properly configured

### Generating Sample Data

If the original dataset is not available:

```python
from data.sample_data import generate_sample_files
generate_sample_files()
```

This creates synthetic employee data that matches the expected format.

## Contributing

1. Follow the existing code structure and documentation style
2. Add comprehensive tests for new features
3. Update documentation for any new functionality
4. Ensure all code passes linting and tests

## License

This project is part of an AI/ML educational exercise. Please respect any applicable licenses and attribution requirements.