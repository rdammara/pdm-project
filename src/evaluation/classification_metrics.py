"""Classification model evaluation metrics."""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, matthews_corrcoef)
from typing import Dict, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationEvaluator:
    """
    Evaluate classification models for failure prediction.
    """
    
    def __init__(self):
        """Initialize classification evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self,
                 y_true: pd.Series,
                 y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Computing classification metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # ROC-AUC (requires probabilities)
        if y_proba is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Compute derived metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # False positive rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False negative rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        self.logger.info(f"Metrics: {metrics}")
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        self.logger.info(f"\nClassification Report:\n{report}")
        
        return metrics
    
    def plot_confusion_matrix(self,
                             y_true: pd.Series,
                             y_pred: np.ndarray,
                             save_path: Optional[str] = None,
                             normalize: bool = False) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            normalize: Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['No Failure', 'Failure'],
                   yticklabels=['No Failure', 'Failure'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self,
                      y_true: pd.Series,
                      y_proba: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            save_path: Path to save figure
        """
        if len(np.unique(y_true)) < 2:
            self.logger.warning("Cannot plot ROC curve with single class")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self,
                                    y_true: pd.Series,
                                    y_proba: np.ndarray,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            save_path: Path to save figure
        """
        if len(np.unique(y_true)) < 2:
            self.logger.warning("Cannot plot PR curve with single class")
            return
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_threshold_analysis(self,
                               y_true: pd.Series,
                               y_proba: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        Plot how metrics vary with classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            save_path: Path to save figure
        """
        thresholds = np.arange(0.0, 1.01, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Classification Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def compute_pdm_metrics(self,
                           y_true: pd.Series,
                           y_pred: np.ndarray,
                           time_to_failure: Optional[pd.Series] = None,
                           early_detection_window: int = 10) -> Dict[str, float]:
        """
        Compute PdM-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            time_to_failure: Time to failure values
            early_detection_window: Window for early detection (in time units)
            
        Returns:
            Dictionary of PdM metrics
        """
        self.logger.info("Computing PdM-specific metrics...")
        
        pdm_metrics = {}
        
        # False alarm rate (FPR)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            pdm_metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Early detection rate
        if time_to_failure is not None:
            early_detections = ((y_pred == 1) & (y_true == 1) & 
                              (time_to_failure > early_detection_window)).sum()
            total_failures = (y_true == 1).sum()
            pdm_metrics['early_detection_rate'] = early_detections / total_failures if total_failures > 0 else 0
            
            # Average advance warning time
            correct_predictions = (y_pred == 1) & (y_true == 1)
            if correct_predictions.sum() > 0:
                pdm_metrics['avg_advance_warning'] = time_to_failure[correct_predictions].mean()
            else:
                pdm_metrics['avg_advance_warning'] = 0
        
        self.logger.info(f"PdM metrics: {pdm_metrics}")
        
        return pdm_metrics


def evaluate_classification(y_true: pd.Series,
                           y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray] = None,
                           plot_dir: Optional[str] = None) -> Dict[str, any]:
    """
    Convenience function for complete classification evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        plot_dir: Directory to save plots (None = no plots)
        
    Returns:
        Dictionary of metrics
    """
    evaluator = ClassificationEvaluator()
    
    # Compute metrics
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)
    
    # Generate plots if requested
    if plot_dir:
        import os
        os.makedirs(plot_dir, exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(plot_dir, 'confusion_matrix.png')
        )
        
        if y_proba is not None:
            evaluator.plot_roc_curve(
                y_true, y_proba,
                save_path=os.path.join(plot_dir, 'roc_curve.png')
            )
            
            evaluator.plot_precision_recall_curve(
                y_true, y_proba,
                save_path=os.path.join(plot_dir, 'precision_recall_curve.png')
            )
            
            evaluator.plot_threshold_analysis(
                y_true, y_proba,
                save_path=os.path.join(plot_dir, 'threshold_analysis.png')
            )
    
    return metrics
