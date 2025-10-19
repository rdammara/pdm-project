"""Statistical feature engineering module."""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging


class StatisticalFeatureGenerator:
    """
    Generate statistical features from sensor data.
    
    This class creates domain-specific and statistical features
    for predictive maintenance tasks.
    """
    
    def __init__(self):
        """Initialize statistical feature generator."""
        self.logger = logging.getLogger(__name__)
    
    def generate_features(self,
                         df: pd.DataFrame,
                         sensor_cols: Optional[List[str]] = None,
                         machine_id_col: str = 'machine_id') -> pd.DataFrame:
        """
        Generate all statistical features.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns to generate features for
            machine_id_col: Name of machine ID column
            
        Returns:
            DataFrame with added statistical features
        """
        self.logger.info("Generating statistical features...")
        
        df = df.copy()
        
        if sensor_cols is None:
            exclude_cols = [machine_id_col, 'timestamp', 'RUL', 'failure_label']
            sensor_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_cols]
        
        # Add interaction features
        df = self._add_interaction_features(df, sensor_cols)
        
        # Add ratio features
        df = self._add_ratio_features(df, sensor_cols)
        
        # Add power features
        df = self._add_power_features(df, sensor_cols)
        
        self.logger.info(f"Generated statistical features. New shape: {df.shape}")
        
        return df
    
    def _add_interaction_features(self,
                                 df: pd.DataFrame,
                                 sensor_cols: List[str],
                                 max_interactions: int = 10) -> pd.DataFrame:
        """
        Add interaction features between sensors.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            max_interactions: Maximum number of interaction pairs to create
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Adding interaction features...")
        
        # Select most important pairs (you can customize this logic)
        # For now, just take first N pairs
        count = 0
        for i, col1 in enumerate(sensor_cols):
            if count >= max_interactions:
                break
            for col2 in sensor_cols[i+1:]:
                if count >= max_interactions:
                    break
                
                # Product interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                count += 1
        
        return df
    
    def _add_ratio_features(self,
                           df: pd.DataFrame,
                           sensor_cols: List[str],
                           max_ratios: int = 10) -> pd.DataFrame:
        """
        Add ratio features between sensors.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            max_ratios: Maximum number of ratio pairs to create
            
        Returns:
            DataFrame with ratio features
        """
        self.logger.info("Adding ratio features...")
        
        count = 0
        for i, col1 in enumerate(sensor_cols):
            if count >= max_ratios:
                break
            for col2 in sensor_cols[i+1:]:
                if count >= max_ratios:
                    break
                
                # Avoid division by zero
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-10)
                count += 1
        
        return df
    
    def _add_power_features(self,
                           df: pd.DataFrame,
                           sensor_cols: List[str],
                           powers: List[int] = [2]) -> pd.DataFrame:
        """
        Add polynomial features.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            powers: List of powers to compute
            
        Returns:
            DataFrame with power features
        """
        self.logger.info("Adding power features...")
        
        for col in sensor_cols:
            for power in powers:
                df[f'{col}_pow{power}'] = df[col] ** power
        
        return df
    
    def add_domain_features(self,
                           df: pd.DataFrame,
                           machine_id_col: str = 'machine_id') -> pd.DataFrame:
        """
        Add domain-specific features for predictive maintenance.
        
        These features are based on common patterns in industrial equipment.
        
        Args:
            df: DataFrame with sensor data
            machine_id_col: Machine ID column
            
        Returns:
            DataFrame with domain features
        """
        self.logger.info("Adding domain-specific features...")
        
        # Operating time (cumulative count per machine)
        df['operating_time'] = df.groupby(machine_id_col).cumcount()
        
        # Time since last measurement
        if 'time_delta' in df.columns:
            df['time_delta_log'] = np.log1p(df['time_delta'])
        
        return df
    
    def add_cumulative_features(self,
                               df: pd.DataFrame,
                               sensor_cols: List[str],
                               machine_id_col: str = 'machine_id') -> pd.DataFrame:
        """
        Add cumulative features (running sums, running means).
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            machine_id_col: Machine ID column
            
        Returns:
            DataFrame with cumulative features
        """
        self.logger.info("Adding cumulative features...")
        
        for col in sensor_cols[:5]:  # Limit to first 5 sensors to avoid too many features
            # Cumulative sum
            df[f'{col}_cumsum'] = df.groupby(machine_id_col)[col].cumsum()
            
            # Cumulative mean
            df[f'{col}_cummean'] = df.groupby(machine_id_col)[col].expanding().mean().reset_index(drop=True)
        
        return df
    
    def add_percentile_features(self,
                               df: pd.DataFrame,
                               sensor_cols: List[str],
                               machine_id_col: str = 'machine_id',
                               window: int = 50) -> pd.DataFrame:
        """
        Add percentile features within rolling windows.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            machine_id_col: Machine ID column
            window: Window size
            
        Returns:
            DataFrame with percentile features
        """
        self.logger.info("Adding percentile features...")
        
        for col in sensor_cols[:5]:  # Limit to avoid too many features
            # 25th percentile
            df[f'{col}_q25_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).quantile(0.25)
            )
            
            # 75th percentile
            df[f'{col}_q75_{window}'] = df.groupby(machine_id_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).quantile(0.75)
            )
            
            # IQR
            df[f'{col}_iqr_{window}'] = df[f'{col}_q75_{window}'] - df[f'{col}_q25_{window}']
        
        return df
