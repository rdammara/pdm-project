"""SQL Server data extraction module."""

import pyodbc
import pandas as pd
import yaml
from typing import Optional, Dict, Any
import logging
from sqlalchemy import create_engine
from urllib.parse import quote_plus


class SQLExtractor:
    """
    Extract data from SQL Server database.
    
    This class handles connection to SQL Server and extraction of sensor data,
    failure events, and maintenance logs for predictive maintenance analysis.
    """
    
    def __init__(self, config_path: str = 'config/database_config.yaml'):
        """
        Initialize SQL extractor with database configuration.
        
        Args:
            config_path: Path to database configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.connection = None
        self.engine = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('database', {})
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def connect(self) -> None:
        """Establish connection to SQL Server."""
        try:
            # Build connection string
            if self.config.get('trusted_connection'):
                conn_str = (
                    f"DRIVER={{{self.config['driver']}}};"
                    f"SERVER={self.config['server']};"
                    f"DATABASE={self.config['database']};"
                    f"Trusted_Connection=yes;"
                )
            else:
                conn_str = (
                    f"DRIVER={{{self.config['driver']}}};"
                    f"SERVER={self.config['server']};"
                    f"DATABASE={self.config['database']};"
                    f"UID={self.config.get('username', '')};"
                    f"PWD={self.config.get('password', '')};"
                )
            
            # Create pyodbc connection
            self.connection = pyodbc.connect(conn_str)
            
            # Create SQLAlchemy engine for pandas integration
            quoted_conn_str = quote_plus(conn_str)
            self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}")
            
            self.logger.info("Successfully connected to SQL Server")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to SQL Server: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Disconnected from SQL Server")
    
    def extract_sensor_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           machine_ids: Optional[list] = None,
                           output_path: str = 'data/raw/sensor_data.csv') -> pd.DataFrame:
        """
        Extract sensor data from SQL Server.
        
        Args:
            start_date: Start date for data extraction (format: 'YYYY-MM-DD')
            end_date: End date for data extraction (format: 'YYYY-MM-DD')
            machine_ids: List of machine IDs to extract (None = all machines)
            output_path: Path to save extracted data
            
        Returns:
            DataFrame containing sensor data
        """
        if not self.engine:
            self.connect()
        
        # Build query
        table_name = self.config.get('tables', {}).get('sensor_data', 'sensor_readings')
        query = f"SELECT * FROM {table_name}"
        
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if machine_ids:
            machine_list = "','".join(map(str, machine_ids))
            conditions.append(f"machine_id IN ('{machine_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY machine_id, timestamp"
        
        self.logger.info(f"Executing query: {query}")
        
        # Execute query with chunking for large datasets
        chunk_size = self.config.get('chunk_size', 10000)
        chunks = []
        
        for chunk in pd.read_sql(query, self.engine, chunksize=chunk_size):
            chunks.append(chunk)
            self.logger.debug(f"Loaded chunk with {len(chunk)} rows")
        
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Extracted {len(df)} rows of sensor data")
        
        # Save to file
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved sensor data to {output_path}")
        
        return df
    
    def extract_failure_events(self,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               machine_ids: Optional[list] = None,
                               output_path: str = 'data/raw/failure_events.csv') -> pd.DataFrame:
        """
        Extract failure events from SQL Server.
        
        Args:
            start_date: Start date for data extraction (format: 'YYYY-MM-DD')
            end_date: End date for data extraction (format: 'YYYY-MM-DD')
            machine_ids: List of machine IDs to extract (None = all machines)
            output_path: Path to save extracted data
            
        Returns:
            DataFrame containing failure events
        """
        if not self.engine:
            self.connect()
        
        # Build query
        table_name = self.config.get('tables', {}).get('failure_events', 'failure_events')
        query = f"SELECT * FROM {table_name}"
        
        conditions = []
        if start_date:
            conditions.append(f"failure_timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"failure_timestamp <= '{end_date}'")
        if machine_ids:
            machine_list = "','".join(map(str, machine_ids))
            conditions.append(f"machine_id IN ('{machine_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY machine_id, failure_timestamp"
        
        self.logger.info(f"Executing query: {query}")
        
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Extracted {len(df)} failure events")
        
        # Save to file
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved failure events to {output_path}")
        
        return df
    
    def extract_maintenance_logs(self,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 machine_ids: Optional[list] = None,
                                 output_path: str = 'data/raw/maintenance_logs.csv') -> pd.DataFrame:
        """
        Extract maintenance logs from SQL Server.
        
        Args:
            start_date: Start date for data extraction (format: 'YYYY-MM-DD')
            end_date: End date for data extraction (format: 'YYYY-MM-DD')
            machine_ids: List of machine IDs to extract (None = all machines)
            output_path: Path to save extracted data
            
        Returns:
            DataFrame containing maintenance logs
        """
        if not self.engine:
            self.connect()
        
        # Build query
        table_name = self.config.get('tables', {}).get('maintenance_logs', 'maintenance_logs')
        query = f"SELECT * FROM {table_name}"
        
        conditions = []
        if start_date:
            conditions.append(f"maintenance_timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"maintenance_timestamp <= '{end_date}'")
        if machine_ids:
            machine_list = "','".join(map(str, machine_ids))
            conditions.append(f"machine_id IN ('{machine_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY machine_id, maintenance_timestamp"
        
        self.logger.info(f"Executing query: {query}")
        
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Extracted {len(df)} maintenance logs")
        
        # Save to file
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved maintenance logs to {output_path}")
        
        return df
    
    def execute_custom_query(self, query: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query to execute
            output_path: Optional path to save results
            
        Returns:
            DataFrame containing query results
        """
        if not self.engine:
            self.connect()
        
        self.logger.info(f"Executing custom query")
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Query returned {len(df)} rows")
        
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved query results to {output_path}")
        
        return df
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
