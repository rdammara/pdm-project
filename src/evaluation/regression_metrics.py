"""Regression model evaluation metrics."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from typing import Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class RegressionEvaluator:
    """
    Evaluate regression models for RUL prediction.
    """
    
    def __init__(self):
        """Initialize regression evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self,
                 y_true: pd.Series,
                 y_pred: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Computing regression metrics...")
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
        
        # Compute metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean, sample_weight=sample_weight)),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean, sample_weight=sample_weight),
            'median_ae': median_absolute_error(y_true_clean, y_pred_clean),
            'r2': r2_score(y_true_clean, y_pred_clean, sample_weight=sample_weight),
            'max_error': np.max(np.abs(y_true_clean - y_pred_clean))
        }
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mape_mask = y_true_clean != 0
        if mape_mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / 
                                            y_true_clean[mape_mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        # Residuals statistics
        residuals = y_true_clean - y_pred_clean
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        self.logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def plot_predictions(self,
                        y_true: pd.Series,
                        y_pred: np.ndarray,
                        save_path: Optional[str] = None,
                        title: str = 'RUL Predictions vs Actual') -> None:
        """
        Plot predicted vs actual RUL values.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            save_path: Path to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title(f'{title} - Scatter Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted RUL')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_residuals_distribution(self,
                                   y_true: pd.Series,
                                   y_pred: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot residuals distribution.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            save_path: Path to save figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residuals Histogram')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_error_by_rul(self,
                         y_true: pd.Series,
                         y_pred: np.ndarray,
                         bins: int = 10,
                         save_path: Optional[str] = None) -> None:
        """
        Plot error metrics by RUL bins.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            bins: Number of bins for RUL ranges
            save_path: Path to save figure
        """
        # Create RUL bins
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'error': np.abs(y_true - y_pred)
        })
        
        df['rul_bin'] = pd.cut(df['true'], bins=bins)
        
        # Compute mean error per bin
        bin_stats = df.groupby('rul_bin')['error'].agg(['mean', 'std', 'count'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(bin_stats))
        ax.bar(x, bin_stats['mean'], yerr=bin_stats['std'], 
               capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([str(label) for label in bin_stats.index], rotation=45, ha='right')
        ax.set_xlabel('RUL Range')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Prediction Error by RUL Range')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()


def evaluate_regression(y_true: pd.Series,
                       y_pred: np.ndarray,
                       plot_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Convenience function for complete regression evaluation.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        plot_dir: Directory to save plots (None = no plots)
        
    Returns:
        Dictionary of metrics
    """
    evaluator = RegressionEvaluator()
    
    # Compute metrics
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Generate plots if requested
    if plot_dir:
        import os
        os.makedirs(plot_dir, exist_ok=True)
        
        evaluator.plot_predictions(
            y_true, y_pred,
            save_path=os.path.join(plot_dir, 'predictions_vs_actual.png')
        )
        
        evaluator.plot_residuals_distribution(
            y_true, y_pred,
            save_path=os.path.join(plot_dir, 'residuals_distribution.png')
        )
        
        evaluator.plot_error_by_rul(
            y_true, y_pred,
            save_path=os.path.join(plot_dir, 'error_by_rul.png')
        )
    
    return metrics
