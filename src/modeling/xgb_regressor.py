"""XGBoost regressor for RUL prediction."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Dict, Any, Tuple
import logging
import pickle
import os


class XGBoostRULRegressor:
    """
    XGBoost model for Remaining Useful Life (RUL) regression.
    
    This class handles training, prediction, and evaluation of RUL regression models.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost regressor.
        
        Args:
            params: XGBoost parameters (uses defaults if None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        self.params = params if params else default_params
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              early_stopping_rounds: Optional[int] = 10) -> None:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target (RUL)
            X_val: Validation features (for early stopping)
            y_val: Validation target
            early_stopping_rounds: Early stopping patience
        """
        self.logger.info("Training XGBoost RUL regression model...")
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Parameters: {self.params}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=True
            )
            self.logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train, verbose=True)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self,
                 X_test: pd.DataFrame,
                 y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target (RUL)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Create model with same parameters
        model = xgb.XGBRegressor(**self.params)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        results = {
            'cv_scores': -scores,  # Convert back to positive RMSE
            'mean_score': -scores.mean(),
            'std_score': scores.std()
        }
        
        self.logger.info(f"CV RMSE: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def tune_hyperparameters(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            param_grid: Optional[Dict[str, list]] = None,
                            cv: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        self.logger.info("Tuning hyperparameters...")
        
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV RMSE: {-grid_search.best_score_:.4f}")
        
        # Update model parameters
        self.params.update(grid_search.best_params_)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance')
        
        self.logger.info(f"Model loaded from {filepath}")
