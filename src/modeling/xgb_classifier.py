"""XGBoost classifier for failure prediction."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from typing import Optional, Dict, Any, Tuple
import logging
import pickle
import os


class XGBoostFailureClassifier:
    """
    XGBoost model for failure prediction (binary classification).
    
    This class handles training, prediction, and evaluation of failure classification models.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost classifier.
        
        Args:
            params: XGBoost parameters (uses defaults if None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        default_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        self.params = params if params else default_params
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.threshold = 0.5
    
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
            y_train: Training target (binary labels)
            X_val: Validation features (for early stopping)
            y_val: Validation target
            early_stopping_rounds: Early stopping patience
        """
        self.logger.info("Training XGBoost failure classification model...")
        self.logger.info(f"Training data shape: {X_train.shape}")
        
        # Check class distribution
        class_dist = y_train.value_counts()
        self.logger.info(f"Class distribution:\n{class_dist}")
        
        # Adjust scale_pos_weight for imbalanced classes
        if 0 in class_dist.index and 1 in class_dist.index:
            scale_pos_weight = class_dist[0] / class_dist[1]
            self.params['scale_pos_weight'] = scale_pos_weight
            self.logger.info(f"Set scale_pos_weight to {scale_pos_weight:.2f}")
        
        self.logger.info(f"Parameters: {self.params}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
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
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            threshold: Classification threshold (uses default if None)
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        threshold = threshold if threshold is not None else self.threshold
        
        # Get probabilities
        proba = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (proba >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target (binary labels)
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model...")
        
        # Get predictions
        y_pred = self.predict(X_test, threshold)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        self.logger.info(f"\nClassification Report:\n{report}")
        
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
        model = xgb.XGBClassifier(**self.params)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        
        results = {
            'cv_scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        
        self.logger.info(f"CV F1: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
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
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")
        
        # Update model parameters
        self.params.update(grid_search.best_params_)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def optimize_threshold(self,
                          X_val: pd.DataFrame,
                          y_val: pd.Series,
                          metric: str = 'f1') -> float:
        """
        Optimize classification threshold.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        self.logger.info(f"Optimizing threshold for {metric}...")
        
        # Get probabilities
        y_proba = self.predict_proba(X_val)
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_score:.4f})")
        
        return best_threshold
    
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
            'feature_importance': self.feature_importance,
            'threshold': self.threshold
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
        self.threshold = model_data.get('threshold', 0.5)
        
        self.logger.info(f"Model loaded from {filepath}")
