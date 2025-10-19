"""Event labeling module for failure prediction."""

import pandas as pd
import numpy as np
from typing import Optional, Union
import logging


class EventLabeler:
    """
    Label time-series data for failure prediction.
    
    This class creates binary labels for classification tasks by identifying
    time windows before failures where we want to predict failure events.
    """
    
    def __init__(self, prediction_horizon: int = 30, time_unit: str = 'hours'):
        """
        Initialize event labeler.
        
        Args:
            prediction_horizon: Time window before failure to label as positive
            time_unit: Time unit for prediction horizon ('seconds', 'hours', 'days')
        """
        self.logger = logging.getLogger(__name__)
        self.prediction_horizon = prediction_horizon
        self.time_unit = time_unit
    
    def label_failure_window(self,
                            sensor_data: pd.DataFrame,
                            failure_events: pd.DataFrame,
                            machine_id_col: str = 'machine_id',
                            timestamp_col: str = 'timestamp',
                            failure_timestamp_col: str = 'failure_timestamp') -> pd.DataFrame:
        """
        Label data points within prediction horizon of failure as positive class.
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with failure events
            machine_id_col: Name of machine ID column
            timestamp_col: Name of timestamp column in sensor data
            failure_timestamp_col: Name of failure timestamp column in failure events
            
        Returns:
            DataFrame with added 'failure_label' binary column
        """
        self.logger.info(f"Labeling failure windows with {self.prediction_horizon} {self.time_unit} horizon...")
        
        sensor_data = sensor_data.copy()
        failure_events = failure_events.copy()
        
        # Ensure timestamps are datetime
        sensor_data[timestamp_col] = pd.to_datetime(sensor_data[timestamp_col])
        failure_events[failure_timestamp_col] = pd.to_datetime(failure_events[failure_timestamp_col])
        
        # Initialize label column
        sensor_data['failure_label'] = 0
        
        # Convert prediction horizon to timedelta
        if self.time_unit == 'seconds':
            horizon_delta = pd.Timedelta(seconds=self.prediction_horizon)
        elif self.time_unit == 'hours':
            horizon_delta = pd.Timedelta(hours=self.prediction_horizon)
        elif self.time_unit == 'days':
            horizon_delta = pd.Timedelta(days=self.prediction_horizon)
        else:
            raise ValueError(f"Unknown time unit: {self.time_unit}")
        
        # Get unique machines
        machines = sensor_data[machine_id_col].unique()
        
        for machine in machines:
            # Get sensor data for this machine
            machine_mask = sensor_data[machine_id_col] == machine
            machine_data = sensor_data[machine_mask].copy()
            
            # Get failure events for this machine
            machine_failures = failure_events[failure_events[machine_id_col] == machine]
            
            if len(machine_failures) == 0:
                continue
            
            # Process each failure event
            for idx, failure_row in machine_failures.iterrows():
                failure_time = failure_row[failure_timestamp_col]
                
                # Define the prediction window
                window_start = failure_time - horizon_delta
                
                # Label points within the window
                in_window = (
                    (machine_data[timestamp_col] >= window_start) &
                    (machine_data[timestamp_col] < failure_time)
                )
                
                sensor_data.loc[machine_mask & in_window, 'failure_label'] = 1
        
        # Log class distribution
        class_counts = sensor_data['failure_label'].value_counts()
        self.logger.info(f"Failure label distribution:\n{class_counts}")
        self.logger.info(f"Positive class ratio: {class_counts.get(1, 0) / len(sensor_data):.4f}")
        
        return sensor_data
    
    def label_by_rul_threshold(self,
                               sensor_data: pd.DataFrame,
                               rul_col: str = 'RUL',
                               threshold: float = 30) -> pd.DataFrame:
        """
        Create binary labels based on RUL threshold.
        
        Args:
            sensor_data: DataFrame with RUL column
            rul_col: Name of RUL column
            threshold: RUL threshold (below = 1, above = 0)
            
        Returns:
            DataFrame with added 'failure_label' binary column
        """
        self.logger.info(f"Labeling based on RUL threshold: {threshold}")
        
        sensor_data = sensor_data.copy()
        sensor_data['failure_label'] = (sensor_data[rul_col] <= threshold).astype(int)
        
        # Log class distribution
        class_counts = sensor_data['failure_label'].value_counts()
        self.logger.info(f"Failure label distribution:\n{class_counts}")
        
        return sensor_data
    
    def label_multi_class_failures(self,
                                   sensor_data: pd.DataFrame,
                                   failure_events: pd.DataFrame,
                                   machine_id_col: str = 'machine_id',
                                   timestamp_col: str = 'timestamp',
                                   failure_timestamp_col: str = 'failure_timestamp',
                                   failure_type_col: str = 'failure_type') -> pd.DataFrame:
        """
        Label data points with failure types for multi-class classification.
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with failure events including failure types
            machine_id_col: Name of machine ID column
            timestamp_col: Name of timestamp column in sensor data
            failure_timestamp_col: Name of failure timestamp column in failure events
            failure_type_col: Name of failure type column in failure events
            
        Returns:
            DataFrame with added 'failure_type' column
        """
        self.logger.info("Labeling multi-class failure types...")
        
        sensor_data = sensor_data.copy()
        failure_events = failure_events.copy()
        
        # Ensure timestamps are datetime
        sensor_data[timestamp_col] = pd.to_datetime(sensor_data[timestamp_col])
        failure_events[failure_timestamp_col] = pd.to_datetime(failure_events[failure_timestamp_col])
        
        # Initialize label column
        sensor_data['failure_type'] = 'no_failure'
        
        # Convert prediction horizon to timedelta
        if self.time_unit == 'seconds':
            horizon_delta = pd.Timedelta(seconds=self.prediction_horizon)
        elif self.time_unit == 'hours':
            horizon_delta = pd.Timedelta(hours=self.prediction_horizon)
        elif self.time_unit == 'days':
            horizon_delta = pd.Timedelta(days=self.prediction_horizon)
        else:
            raise ValueError(f"Unknown time unit: {self.time_unit}")
        
        # Get unique machines
        machines = sensor_data[machine_id_col].unique()
        
        for machine in machines:
            # Get sensor data for this machine
            machine_mask = sensor_data[machine_id_col] == machine
            machine_data = sensor_data[machine_mask].copy()
            
            # Get failure events for this machine
            machine_failures = failure_events[failure_events[machine_id_col] == machine]
            
            if len(machine_failures) == 0:
                continue
            
            # Process each failure event
            for idx, failure_row in machine_failures.iterrows():
                failure_time = failure_row[failure_timestamp_col]
                failure_type = failure_row[failure_type_col]
                
                # Define the prediction window
                window_start = failure_time - horizon_delta
                
                # Label points within the window
                in_window = (
                    (machine_data[timestamp_col] >= window_start) &
                    (machine_data[timestamp_col] < failure_time)
                )
                
                sensor_data.loc[machine_mask & in_window, 'failure_type'] = failure_type
        
        # Log class distribution
        class_counts = sensor_data['failure_type'].value_counts()
        self.logger.info(f"Failure type distribution:\n{class_counts}")
        
        return sensor_data
    
    def add_time_to_failure(self,
                           sensor_data: pd.DataFrame,
                           failure_events: pd.DataFrame,
                           machine_id_col: str = 'machine_id',
                           timestamp_col: str = 'timestamp',
                           failure_timestamp_col: str = 'failure_timestamp') -> pd.DataFrame:
        """
        Add time-to-failure feature (similar to RUL but unlimited).
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with failure events
            machine_id_col: Name of machine ID column
            timestamp_col: Name of timestamp column in sensor data
            failure_timestamp_col: Name of failure timestamp column in failure events
            
        Returns:
            DataFrame with added 'time_to_failure' column
        """
        self.logger.info("Computing time to failure...")
        
        sensor_data = sensor_data.copy()
        failure_events = failure_events.copy()
        
        # Ensure timestamps are datetime
        sensor_data[timestamp_col] = pd.to_datetime(sensor_data[timestamp_col])
        failure_events[failure_timestamp_col] = pd.to_datetime(failure_events[failure_timestamp_col])
        
        # Initialize column
        sensor_data['time_to_failure'] = np.nan
        
        # Get unique machines
        machines = sensor_data[machine_id_col].unique()
        
        for machine in machines:
            machine_mask = sensor_data[machine_id_col] == machine
            machine_data = sensor_data[machine_mask].copy()
            machine_failures = failure_events[failure_events[machine_id_col] == machine]
            
            if len(machine_failures) == 0:
                continue
            
            for idx, failure_row in machine_failures.iterrows():
                failure_time = failure_row[failure_timestamp_col]
                before_failure = machine_data[timestamp_col] <= failure_time
                
                # Compute time difference
                time_diff = failure_time - machine_data.loc[before_failure, timestamp_col]
                
                if self.time_unit == 'seconds':
                    ttf_values = time_diff.dt.total_seconds()
                elif self.time_unit == 'hours':
                    ttf_values = time_diff.dt.total_seconds() / 3600
                elif self.time_unit == 'days':
                    ttf_values = time_diff.dt.total_seconds() / 86400
                
                # Update values (keep minimum if multiple failures)
                current_ttf = sensor_data.loc[machine_mask & before_failure, 'time_to_failure']
                new_ttf = ttf_values.values
                
                mask_to_update = machine_mask & before_failure
                sensor_data.loc[mask_to_update, 'time_to_failure'] = np.where(
                    current_ttf.isna(),
                    new_ttf,
                    np.minimum(current_ttf, new_ttf)
                )
        
        return sensor_data
