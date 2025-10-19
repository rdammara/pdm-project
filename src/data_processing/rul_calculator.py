"""RUL (Remaining Useful Life) calculator module."""

import pandas as pd
import numpy as np
from typing import Optional, Union
import logging


class RULCalculator:
    """
    Calculate Remaining Useful Life (RUL) for time-series sensor data.
    
    This class computes RUL labels for predictive maintenance by working
    backward from known failure events to assign RUL values to each timestamp.
    """
    
    def __init__(self, clip_max: Optional[int] = None):
        """
        Initialize RUL calculator.
        
        Args:
            clip_max: Maximum RUL value to clip (None = no clipping)
        """
        self.logger = logging.getLogger(__name__)
        self.clip_max = clip_max
    
    def compute_rul(self,
                    sensor_data: pd.DataFrame,
                    failure_events: pd.DataFrame,
                    machine_id_col: str = 'machine_id',
                    timestamp_col: str = 'timestamp',
                    failure_timestamp_col: str = 'failure_timestamp',
                    time_unit: str = 'hours') -> pd.DataFrame:
        """
        Compute RUL for sensor data based on failure events.
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with failure events
            machine_id_col: Name of machine ID column
            timestamp_col: Name of timestamp column in sensor data
            failure_timestamp_col: Name of failure timestamp column in failure events
            time_unit: Time unit for RUL ('seconds', 'hours', 'days')
            
        Returns:
            DataFrame with added RUL column
        """
        self.logger.info("Computing RUL labels...")
        
        # Make copies to avoid modifying originals
        sensor_data = sensor_data.copy()
        failure_events = failure_events.copy()
        
        # Ensure timestamps are datetime
        sensor_data[timestamp_col] = pd.to_datetime(sensor_data[timestamp_col])
        failure_events[failure_timestamp_col] = pd.to_datetime(failure_events[failure_timestamp_col])
        
        # Sort data
        sensor_data = sensor_data.sort_values([machine_id_col, timestamp_col])
        failure_events = failure_events.sort_values([machine_id_col, failure_timestamp_col])
        
        # Initialize RUL column
        sensor_data['RUL'] = np.nan
        
        # Get unique machines
        machines = sensor_data[machine_id_col].unique()
        
        for machine in machines:
            # Get sensor data for this machine
            machine_mask = sensor_data[machine_id_col] == machine
            machine_data = sensor_data[machine_mask].copy()
            
            # Get failure events for this machine
            machine_failures = failure_events[failure_events[machine_id_col] == machine]
            
            if len(machine_failures) == 0:
                # No failures recorded for this machine
                # Assign a large RUL or NaN
                sensor_data.loc[machine_mask, 'RUL'] = self.clip_max if self.clip_max else np.nan
                continue
            
            # Process each failure event
            for idx, failure_row in machine_failures.iterrows():
                failure_time = failure_row[failure_timestamp_col]
                
                # Find data points before this failure
                before_failure = machine_data[timestamp_col] <= failure_time
                
                # Compute time difference (RUL)
                time_diff = failure_time - machine_data.loc[before_failure, timestamp_col]
                
                # Convert to desired time unit
                if time_unit == 'seconds':
                    rul_values = time_diff.dt.total_seconds()
                elif time_unit == 'hours':
                    rul_values = time_diff.dt.total_seconds() / 3600
                elif time_unit == 'days':
                    rul_values = time_diff.dt.total_seconds() / 86400
                else:
                    raise ValueError(f"Unknown time unit: {time_unit}")
                
                # Update RUL values (use minimum if multiple failures apply)
                current_rul = sensor_data.loc[machine_mask & before_failure, 'RUL']
                new_rul = rul_values.values
                
                # Keep minimum RUL if already set
                mask_to_update = machine_mask & before_failure
                sensor_data.loc[mask_to_update, 'RUL'] = np.where(
                    current_rul.isna(),
                    new_rul,
                    np.minimum(current_rul, new_rul)
                )
        
        # Clip RUL if specified
        if self.clip_max is not None:
            sensor_data['RUL'] = sensor_data['RUL'].clip(upper=self.clip_max)
        
        # Log statistics
        rul_stats = sensor_data['RUL'].describe()
        self.logger.info(f"RUL statistics:\n{rul_stats}")
        
        return sensor_data
    
    def compute_rul_by_cycles(self,
                              sensor_data: pd.DataFrame,
                              failure_events: pd.DataFrame,
                              machine_id_col: str = 'machine_id',
                              cycle_col: str = 'cycle',
                              failure_cycle_col: str = 'failure_cycle') -> pd.DataFrame:
        """
        Compute RUL based on cycle numbers instead of timestamps.
        
        Useful for datasets where cycles are more meaningful than time.
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with failure events
            machine_id_col: Name of machine ID column
            cycle_col: Name of cycle column in sensor data
            failure_cycle_col: Name of failure cycle column in failure events
            
        Returns:
            DataFrame with added RUL column
        """
        self.logger.info("Computing RUL labels by cycles...")
        
        sensor_data = sensor_data.copy()
        failure_events = failure_events.copy()
        
        # Sort data
        sensor_data = sensor_data.sort_values([machine_id_col, cycle_col])
        failure_events = failure_events.sort_values([machine_id_col, failure_cycle_col])
        
        # Initialize RUL column
        sensor_data['RUL'] = np.nan
        
        # Get unique machines
        machines = sensor_data[machine_id_col].unique()
        
        for machine in machines:
            # Get sensor data for this machine
            machine_mask = sensor_data[machine_id_col] == machine
            machine_data = sensor_data[machine_mask].copy()
            
            # Get failure events for this machine
            machine_failures = failure_events[failure_events[machine_id_col] == machine]
            
            if len(machine_failures) == 0:
                sensor_data.loc[machine_mask, 'RUL'] = self.clip_max if self.clip_max else np.nan
                continue
            
            # Process each failure event
            for idx, failure_row in machine_failures.iterrows():
                failure_cycle = failure_row[failure_cycle_col]
                
                # Find data points before this failure
                before_failure = machine_data[cycle_col] <= failure_cycle
                
                # Compute RUL in cycles
                rul_values = failure_cycle - machine_data.loc[before_failure, cycle_col]
                
                # Update RUL values
                current_rul = sensor_data.loc[machine_mask & before_failure, 'RUL']
                new_rul = rul_values.values
                
                mask_to_update = machine_mask & before_failure
                sensor_data.loc[mask_to_update, 'RUL'] = np.where(
                    current_rul.isna(),
                    new_rul,
                    np.minimum(current_rul, new_rul)
                )
        
        # Clip RUL if specified
        if self.clip_max is not None:
            sensor_data['RUL'] = sensor_data['RUL'].clip(upper=self.clip_max)
        
        return sensor_data
    
    def add_rul_buckets(self,
                        df: pd.DataFrame,
                        rul_col: str = 'RUL',
                        buckets: Optional[list] = None) -> pd.DataFrame:
        """
        Add categorical RUL buckets for classification tasks.
        
        Args:
            df: DataFrame with RUL column
            rul_col: Name of RUL column
            buckets: List of bucket thresholds (e.g., [10, 30, 50])
                    Creates buckets: [0-10), [10-30), [30-50), [50+]
            
        Returns:
            DataFrame with added RUL_bucket column
        """
        df = df.copy()
        
        if buckets is None:
            buckets = [10, 30, 50, 100]
        
        # Create bucket labels
        labels = []
        for i in range(len(buckets)):
            if i == 0:
                labels.append(f'0-{buckets[i]}')
            else:
                labels.append(f'{buckets[i-1]}-{buckets[i]}')
        labels.append(f'{buckets[-1]}+')
        
        # Create buckets
        bins = [0] + buckets + [np.inf]
        df['RUL_bucket'] = pd.cut(df[rul_col], bins=bins, labels=labels, right=False)
        
        return df
