"""
Model Training Module for Customer Churn Prediction
Handles training, hyperparameter tuning, and cross-validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                   cross_val_score, StratifiedKFold)
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, CV_CONFIG, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training and hyperparameter optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ModelTrainer with configuration
        
        Args:
            config: Configuration dictionary, defaults to MODEL_CONFIG
        """
        self.config = config or MODEL_CONFIG
        self.cv_config = CV_CONFIG
        self.trained_models = {}
        self.best_params = {}
        self.cv_results = {}
        
        logger.info("ModelTrainer initialized")
    
    def get_model_instance(self, model_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Get model instance with specified parameters
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
        """
        model_config = self.config['models'][model_name]
        model_class = model_config['class']
        default_params = model_config['params'].copy()
        
        if params:
            default_params.update(params)
        
        # Map class names to actual classes
        model_classes = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'XGBClassifier': XGBClassifier
        }
        
        if model_class not in model_classes:
            raise ValueError(f"Unknown model class: {model_class}")
        
        return model_classes[model_class](**default_params)
    
    def perform_grid_search(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform grid search for hyperparameter optimization
        
        Args:
            model_name: Name of the model
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger.info(f"Starting grid search for {model_name}")
        
        # Get base model and hyperparameters
        base_model = self.get_model_instance(model_name)
        param_grid = self.config['models'][model_name]['hyperparameters']
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_config['cv_folds'],
            shuffle=True,
            random_state=self.cv_config['random_state']
        )
        
        # Set up scorer
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=self.cv_config['n_jobs'],
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X, y)
        end_time = time.time()
        
        logger.info(f"Grid search for {model_name} completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def perform_random_search(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform randomized search for hyperparameter optimization
        
        Args:
            model_name: Name of the model
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger.info(f"Starting randomized search for {model_name}")
        
        # Get base model and hyperparameters
        base_model = self.get_model_instance(model_name)
        param_distributions = self.config['models'][model_name]['hyperparameters']
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_config['cv_folds'],
            shuffle=True,
            random_state=self.cv_config['random_state']
        )
        
        # Set up scorer
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.cv_config['n_iter'],
            cv=cv,
            scoring=scorer,
            n_jobs=self.cv_config['n_jobs'],
            random_state=self.cv_config['random_state'],
            verbose=1
        )
        
        start_time = time.time()
        random_search.fit(X, y)
        end_time = time.time()
        
        logger.info(f"Randomized search for {model_name} completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best params: {random_search.best_params_}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    def perform_optuna_optimization(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                                  n_trials: int = 100) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform Optuna-based hyperparameter optimization
        
        Args:
            model_name: Name of the model
            X: Training features
            y: Training target
            n_trials: Number of optimization trials
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger.info(f"Starting Optuna optimization for {model_name} with {n_trials} trials")
        
        def objective(trial):
            # Define parameter space based on model
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                }
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 10, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
                }
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
            else:
                raise ValueError(f"Optuna optimization not supported for {model_name}")
            
            # Create model with suggested parameters
            model = self.get_model_instance(model_name, params)
            
            # Perform cross-validation
            cv = StratifiedKFold(
                n_splits=self.cv_config['cv_folds'],
                shuffle=True,
                random_state=self.cv_config['random_state']
            )
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.cv_config['random_state'])
        )
        
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        end_time = time.time()
        
        # Get best parameters and train final model
        best_params = study.best_params
        best_model = self.get_model_instance(model_name, best_params)
        best_model.fit(X, y)
        
        logger.info(f"Optuna optimization for {model_name} completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_model, best_params, study.best_value
    
    def train_model_with_cv(self, model_name: str, X: np.ndarray, y: np.ndarray,
                           optimization_method: str = 'random_search') -> Dict[str, Any]:
        """
        Train model with cross-validation and hyperparameter optimization
        
        Args:
            model_name: Name of the model
            X: Training features
            y: Training target
            optimization_method: Method for hyperparameter optimization
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {model_name} with {optimization_method}")
        
        # Perform hyperparameter optimization
        if optimization_method == 'grid_search':
            best_model, best_params, best_score = self.perform_grid_search(model_name, X, y)
        elif optimization_method == 'random_search':
            best_model, best_params, best_score = self.perform_random_search(model_name, X, y)
        elif optimization_method == 'optuna':
            best_model, best_params, best_score = self.perform_optuna_optimization(model_name, X, y)
        else:
            # Train with default parameters
            logger.info(f"Training {model_name} with default parameters")
            best_model = self.get_model_instance(model_name)
            best_model.fit(X, y)
            best_params = self.config['models'][model_name]['params']
            
            # Evaluate with cross-validation
            cv = StratifiedKFold(
                n_splits=self.cv_config['cv_folds'],
                shuffle=True,
                random_state=self.cv_config['random_state']
            )
            cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc')
            best_score = cv_scores.mean()
        
        # Store results
        self.trained_models[model_name] = best_model
        self.best_params[model_name] = best_params
        self.cv_results[model_name] = best_score
        
        # Training summary
        training_results = {
            'model': best_model,
            'best_params': best_params,
            'cv_score': best_score,
            'optimization_method': optimization_method
        }
        
        logger.info(f"Training completed for {model_name}")
        logger.info(f"CV Score: {best_score:.4f}")
        
        return training_results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        optimization_method: str = 'random_search') -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models
        
        Args:
            X: Training features
            y: Training target
            optimization_method: Method for hyperparameter optimization
            
        Returns:
            Dictionary containing results for all models
        """
        logger.info("Starting training for all models")
        
        all_results = {}
        total_start_time = time.time()
        
        for model_name in self.config['models'].keys():
            try:
                results = self.train_model_with_cv(model_name, X, y, optimization_method)
                all_results[model_name] = results
                logger.info(f"✓ {model_name} training completed successfully")
            except Exception as e:
                logger.error(f"✗ Error training {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        logger.info(f"All models training completed in {total_time:.2f} seconds")
        
        # Print summary
        self.print_training_summary(all_results)
        
        return all_results
    
    def print_training_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Print training summary for all models
        
        Args:
            results: Training results dictionary
        """
        print("\n" + "="*80)
        print("MODEL TRAINING SUMMARY")
        print("="*80)
        
        successful_models = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_models:
            # Sort by CV score
            sorted_models = sorted(successful_models.items(), 
                                 key=lambda x: x[1]['cv_score'], reverse=True)
            
            print(f"{'Model':<20} {'CV Score':<12} {'Optimization':<15}")
            print("-" * 50)
            
            for model_name, result in sorted_models:
                cv_score = result['cv_score']
                opt_method = result['optimization_method']
                print(f"{model_name:<20} {cv_score:<12.4f} {opt_method:<15}")
            
            # Best model
            best_model_name, best_result = sorted_models[0]
            print(f"\n🏆 Best Model: {best_model_name} (CV Score: {best_result['cv_score']:.4f})")
        
        # Failed models
        failed_models = {k: v for k, v in results.items() if 'error' in v}
        if failed_models:
            print(f"\n❌ Failed Models: {list(failed_models.keys())}")
        
        print("="*80)
    
    def save_models(self, save_all: bool = True) -> None:
        """
        Save trained models to disk
        
        Args:
            save_all: Whether to save all models or just the best one
        """
        if not self.trained_models:
            logger.warning("No trained models to save")
            return
        
        logger.info("Saving trained models")
        
        # Ensure models directory exists
        MODELS_DIR.mkdir(exist_ok=True)
        
        if save_all:
            # Save all models
            for model_name, model in self.trained_models.items():
                filename = f"{model_name}_model.pkl"
                filepath = MODELS_DIR / filename
                joblib.dump(model, filepath)
                logger.info(f"Saved {model_name} to {filepath}")
            
            # Save best parameters
            params_filepath = MODELS_DIR / "best_parameters.pkl"
            joblib.dump(self.best_params, params_filepath)
            logger.info(f"Saved best parameters to {params_filepath}")
            
        else:
            # Save only the best model
            if self.cv_results:
                best_model_name = max(self.cv_results.keys(), key=lambda k: self.cv_results[k])
                best_model = self.trained_models[best_model_name]
                
                filename = f"{best_model_name}_model.pkl"
                filepath = MODELS_DIR / filename
                joblib.dump(best_model, filepath)
                logger.info(f"Saved {best_model_name} to {filepath}")
        logger.info("Model saving completed")
        logger.info("All models saved successfully")
        print("Models saved successfully.")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Best parameters saved to: {MODELS_DIR / 'best_parameters.pkl'}")
        print("Training summary printed to console.")
        print("Training completed successfully.")
        print("You can now proceed to evaluate the models or use them for predictions.")
        print("="*80)
        print("Thank you for using the Model Trainer!")
        