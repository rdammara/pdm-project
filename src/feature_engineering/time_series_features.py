"""Time-series feature engineering module."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from scipy import stats


class TimeSeriesFeatureGenerator:
    """
    Generate features from time-series sensor data.
    
    This class creates rolling window features, lag features, and other
    time-series specific transformations for predictive maintenance.
    """
    
    def __init__(self, 
                 window_sizes: List[int] = [10, 20, 50],
                 lag_sizes: List[int] = [1, 2, 5, 10]):
        """
        Initialize feature generator.
        
        Args:
            window_sizes: List of rolling window sizes
            lag_sizes: List of lag sizes for lag features
        """
        self.logger = logging.getLogger(__name__)
        self.window_sizes = window_sizes
        self.lag_sizes = lag_sizes
    
    def generate_features(self,
                         df: pd.DataFrame,
                         sensor_cols: Optional[List[str]] = None,
                         machine_id_col: str = 'machine_id',
                         timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Generate all time-series features.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns to generate features for
                        (None = all numeric columns)
            machine_id_col: Name of machine ID column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with added features
        """
        self.logger.info("Generating time-series features...")
        
        df = df.copy()
        
        # Auto-detect sensor columns if not provided
        if sensor_cols is None:
            exclude_cols = [machine_id_col, timestamp_col, 'RUL', 'failure_label', 
                          'failure_type', 'time_to_failure', 'RUL_bucket']
            sensor_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_cols]
        
        self.logger.info(f"Generating features for {len(sensor_cols)} sensor columns")
        
        # Generate rolling window features
        df = self._add_rolling_features(df, sensor_cols, machine_id_col)
        
        # Generate lag features
        df = self._add_lag_features(df, sensor_cols, machine_id_col)
        
        # Generate rate of change features
        df = self._add_rate_of_change(df, sensor_cols, machine_id_col)
        
        # Generate trend features
        df = self._add_trend_features(df, sensor_cols, machine_id_col)
        
        self.logger.info(f"Generated features. New shape: {df.shape}")
        
        return df
    
    def _add_rolling_features(self,
                             df: pd.DataFrame,
                             sensor_cols: List[str],
                             machine_id_col: str) -> pd.DataFrame:
        """Add rolling window statistical features."""
        self.logger.info("Adding rolling window features...")
        
        for col in sensor_cols:
            for window in self.window_sizes:
                # Group by machine to ensure rolling windows don't cross machines
                grouped = df.groupby(machine_id_col)[col]
                
                # Mean
                df[f'{col}_rolling_mean_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Standard deviation
                df[f'{col}_rolling_std_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                
                # Min
                df[f'{col}_rolling_min_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                
                # Max
                df[f'{col}_rolling_max_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                
                # Range
                df[f'{col}_rolling_range_{window}'] = (
                    df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']
                )
        
        return df
    
    def _add_lag_features(self,
                         df: pd.DataFrame,
                         sensor_cols: List[str],
                         machine_id_col: str) -> pd.DataFrame:
        """Add lag features."""
        self.logger.info("Adding lag features...")
        
        for col in sensor_cols:
            for lag in self.lag_sizes:
                df[f'{col}_lag_{lag}'] = df.groupby(machine_id_col)[col].shift(lag)
        
        return df
    
    def _add_rate_of_change(self,
                           df: pd.DataFrame,
                           sensor_cols: List[str],
                           machine_id_col: str) -> pd.DataFrame:
        """Add rate of change (derivative) features."""
        self.logger.info("Adding rate of change features...")
        
        for col in sensor_cols:
            # First difference (rate of change)
            df[f'{col}_diff'] = df.groupby(machine_id_col)[col].diff()
            
            # Second difference (acceleration)
            df[f'{col}_diff2'] = df.groupby(machine_id_col)[f'{col}_diff'].diff()
            
            # Percentage change
            df[f'{col}_pct_change'] = df.groupby(machine_id_col)[col].pct_change()
        
        return df
    
    def _add_trend_features(self,
                           df: pd.DataFrame,
                           sensor_cols: List[str],
                           machine_id_col: str,
                           window: int = 20) -> pd.DataFrame:
        """Add trend features (slope of linear fit)."""
        self.logger.info("Adding trend features...")
        
        def compute_slope(x):
            if len(x) < 2 or x.isna().all():
                return np.nan
            y = x.values
            X = np.arange(len(y))
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            slope, _ = np.polyfit(X[mask], y[mask], 1)
            return slope
        
        for col in sensor_cols:
            df[f'{col}_trend_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(compute_slope, raw=False)
            )
        
        return df
    
    def add_fft_features(self,
                        df: pd.DataFrame,
                        sensor_cols: List[str],
                        machine_id_col: str,
                        window: int = 50,
                        n_components: int = 5) -> pd.DataFrame:
        """
        Add Fourier Transform features for frequency domain analysis.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            machine_id_col: Machine ID column
            window: Window size for FFT
            n_components: Number of FFT components to extract
            
        Returns:
            DataFrame with FFT features
        """
        self.logger.info("Adding FFT features...")
        
        def compute_fft_features(x, n_comp):
            if len(x) < window or x.isna().all():
                return [np.nan] * n_comp
            
            y = x.values
            # Handle NaN
            if np.isnan(y).any():
                y = pd.Series(y).interpolate(method='linear').fillna(0).values
            
            # Compute FFT
            fft_vals = np.fft.fft(y)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            
            # Get top components
            if len(fft_mag) < n_comp:
                result = list(fft_mag) + [0] * (n_comp - len(fft_mag))
            else:
                result = fft_mag[:n_comp].tolist()
            
            return result
        
        for col in sensor_cols:
            # Compute FFT for each machine
            fft_results = df.groupby(machine_id_col)[col].transform(
                lambda x: pd.Series(
                    x.rolling(window=window, min_periods=window).apply(
                        lambda y: compute_fft_features(y, n_components)[0], raw=False
                    )
                )
            )
            
            # Add FFT component columns
            for i in range(n_components):
                df[f'{col}_fft_{i}'] = df.groupby(machine_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=window).apply(
                        lambda y: compute_fft_features(y, n_components)[i], raw=False
                    )
                )
        
        return df
    
    def add_statistical_moments(self,
                               df: pd.DataFrame,
                               sensor_cols: List[str],
                               machine_id_col: str,
                               window: int = 50) -> pd.DataFrame:
        """
        Add statistical moment features (skewness, kurtosis).
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            machine_id_col: Machine ID column
            window: Window size for computation
            
        Returns:
            DataFrame with statistical moment features
        """
        self.logger.info("Adding statistical moment features...")
        
        for col in sensor_cols:
            # Skewness
            df[f'{col}_skew_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=3).skew()
            )
            
            # Kurtosis
            df[f'{col}_kurt_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=4).kurt()
            )
        
        return df
    
    def add_crossing_features(self,
                             df: pd.DataFrame,
                             sensor_cols: List[str],
                             machine_id_col: str,
                             window: int = 50) -> pd.DataFrame:
        """
        Add zero-crossing and mean-crossing rate features.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            machine_id_col: Machine ID column
            window: Window size for computation
            
        Returns:
            DataFrame with crossing rate features
        """
        self.logger.info("Adding crossing rate features...")
        
        def mean_crossing_rate(x):
            if len(x) < 2 or x.isna().all():
                return np.nan
            mean_val = x.mean()
            crosses = ((x[:-1] - mean_val) * (x[1:].values - mean_val) < 0).sum()
            return crosses / len(x)
        
        for col in sensor_cols:
            df[f'{col}_mean_cross_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(mean_crossing_rate, raw=False)
            )
        
        return df
