"""Report generation module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class ReportGenerator:
    """
    Generate comprehensive reports for predictive maintenance models.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_model_report(self,
                             model_name: str,
                             metrics: Dict[str, Any],
                             feature_importance: Optional[pd.DataFrame] = None,
                             plots_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive model evaluation report.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            feature_importance: DataFrame with feature importance
            plots_dir: Directory containing evaluation plots
            
        Returns:
            Path to generated report
        """
        self.logger.info(f"Generating report for {model_name}...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'{model_name}_report_{timestamp}.html')
        
        # Build HTML report
        html_content = self._build_html_header(model_name)
        html_content += self._build_metrics_section(metrics)
        
        if feature_importance is not None:
            html_content += self._build_feature_importance_section(feature_importance)
        
        if plots_dir and os.path.exists(plots_dir):
            html_content += self._build_plots_section(plots_dir)
        
        html_content += self._build_html_footer()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Report saved to {report_path}")
        
        return report_path
    
    def _build_html_header(self, model_name: str) -> str:
        """Build HTML header."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{model_name} - Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-value {{
            font-weight: bold;
            color: #4CAF50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{model_name} - Evaluation Report</h1>
    <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""
    
    def _build_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Build metrics section."""
        html = "<h2>Performance Metrics</h2>\n<table>\n"
        html += "<tr><th>Metric</th><th>Value</th></tr>\n"
        
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                continue  # Handle separately
            
            if isinstance(value, (int, float, np.integer, np.floating)):
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                html += f"<tr><td>{key}</td><td class='metric-value'>{formatted_value}</td></tr>\n"
        
        html += "</table>\n"
        
        # Add confusion matrix if present
        if 'confusion_matrix' in metrics:
            html += "<h3>Confusion Matrix</h3>\n"
            cm = metrics['confusion_matrix']
            html += "<table>\n"
            html += "<tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>\n"
            html += f"<tr><td><strong>Actual Negative</strong></td><td>{cm[0, 0]}</td><td>{cm[0, 1]}</td></tr>\n"
            html += f"<tr><td><strong>Actual Positive</strong></td><td>{cm[1, 0]}</td><td>{cm[1, 1]}</td></tr>\n"
            html += "</table>\n"
        
        return html
    
    def _build_feature_importance_section(self, feature_importance: pd.DataFrame, top_n: int = 20) -> str:
        """Build feature importance section."""
        html = f"<h2>Top {top_n} Feature Importance</h2>\n<table>\n"
        html += "<tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>\n"
        
        for idx, row in feature_importance.head(top_n).iterrows():
            html += f"<tr><td>{idx + 1}</td><td>{row['feature']}</td>"
            html += f"<td class='metric-value'>{row['importance']:.6f}</td></tr>\n"
        
        html += "</table>\n"
        
        return html
    
    def _build_plots_section(self, plots_dir: str) -> str:
        """Build plots section."""
        html = "<h2>Visualizations</h2>\n"
        
        # Find all PNG files in the plots directory
        if os.path.exists(plots_dir):
            for filename in sorted(os.listdir(plots_dir)):
                if filename.endswith('.png'):
                    plot_path = os.path.join(plots_dir, filename)
                    # Use relative path
                    rel_path = os.path.relpath(plot_path, self.output_dir)
                    html += f"<h3>{filename.replace('_', ' ').replace('.png', '').title()}</h3>\n"
                    html += f"<img src='{rel_path}' alt='{filename}'>\n"
        
        return html
    
    def _build_html_footer(self) -> str:
        """Build HTML footer."""
        return """
</body>
</html>
"""
    
    def generate_comparison_report(self,
                                   models_metrics: Dict[str, Dict[str, Any]],
                                   report_name: str = 'model_comparison') -> str:
        """
        Generate a comparison report for multiple models.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            report_name: Name for the comparison report
            
        Returns:
            Path to generated report
        """
        self.logger.info("Generating model comparison report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'{report_name}_{timestamp}.html')
        
        html_content = self._build_html_header("Model Comparison")
        html_content += "<h2>Model Performance Comparison</h2>\n"
        
        # Build comparison table
        html_content += "<table>\n"
        
        # Get all unique metrics
        all_metrics = set()
        for metrics in models_metrics.values():
            all_metrics.update([k for k in metrics.keys() if k != 'confusion_matrix'])
        
        # Header
        html_content += "<tr><th>Metric</th>"
        for model_name in models_metrics.keys():
            html_content += f"<th>{model_name}</th>"
        html_content += "</tr>\n"
        
        # Rows
        for metric in sorted(all_metrics):
            html_content += f"<tr><td><strong>{metric}</strong></td>"
            for model_name in models_metrics.keys():
                value = models_metrics[model_name].get(metric, 'N/A')
                if isinstance(value, (int, float, np.integer, np.floating)):
                    formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                else:
                    formatted_value = str(value)
                html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>\n"
        
        html_content += "</table>\n"
        html_content += self._build_html_footer()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Comparison report saved to {report_path}")
        
        return report_path
    
    def export_metrics_to_excel(self,
                               model_name: str,
                               metrics: Dict[str, Any],
                               feature_importance: Optional[pd.DataFrame] = None) -> str:
        """
        Export metrics to Excel file.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            feature_importance: DataFrame with feature importance
            
        Returns:
            Path to generated Excel file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.output_dir, f'{model_name}_metrics_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Metrics sheet
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': v}
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.integer, np.floating))
            ])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Feature importance sheet
            if feature_importance is not None:
                feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        
        self.logger.info(f"Metrics exported to {excel_path}")
        
        return excel_path
    
    def plot_feature_importance(self,
                               feature_importance: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature importance
            top_n: Number of top features to plot
            save_path: Path to save figure
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
