"""Utility functions for predictive maintenance project."""

import os
import yaml
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('log_file')
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save DataFrame to various formats based on file extension.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the file
        **kwargs: Additional arguments to pass to the save function
    """
    ensure_dir(os.path.dirname(filepath))
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        df.to_csv(filepath, **kwargs)
    elif ext == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        df.to_excel(filepath, **kwargs)
    elif ext == '.pkl':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from various formats based on file extension.
    
    Args:
        filepath: Path to the file to load
        **kwargs: Additional arguments to pass to the load function
        
    Returns:
        Loaded DataFrame
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif ext == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    elif ext == '.pkl':
        return pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def compute_time_delta_features(df: pd.DataFrame, 
                                  timestamp_col: str = 'timestamp',
                                  group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Compute time delta features from timestamps.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        group_col: Optional grouping column (e.g., machine_id)
        
    Returns:
        DataFrame with added time delta features
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    if group_col:
        df = df.sort_values([group_col, timestamp_col])
        df['time_delta'] = df.groupby(group_col)[timestamp_col].diff().dt.total_seconds()
    else:
        df = df.sort_values(timestamp_col)
        df['time_delta'] = df[timestamp_col].diff().dt.total_seconds()
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'interpolate',
                          columns: Optional[list] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame with potential missing values
        strategy: Strategy for handling missing values ('interpolate', 'forward_fill', 'drop')
        columns: Columns to apply strategy to (None = all numeric columns)
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'interpolate':
        df[columns] = df[columns].interpolate(method='linear', limit_direction='both')
    elif strategy == 'forward_fill':
        df[columns] = df[columns].fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna(subset=columns)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def detect_outliers(df: pd.DataFrame, 
                    columns: Optional[list] = None,
                    method: str = 'iqr',
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in DataFrame.
    
    Args:
        df: DataFrame to check for outliers
        columns: Columns to check (None = all numeric columns)
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with boolean columns indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_df = pd.DataFrame(index=df.index)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_df[f'{col}_outlier'] = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return outlier_df


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def build_path(*args) -> str:
    """
    Build a path relative to project root.
    
    Args:
        *args: Path components
        
    Returns:
        Full path as string
    """
    return str(get_project_root() / Path(*args))
