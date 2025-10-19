"""Modeling module initialization."""

from .xgb_regressor import XGBoostRULRegressor
from .xgb_classifier import XGBoostFailureClassifier

__all__ = ['XGBoostRULRegressor', 'XGBoostFailureClassifier']
