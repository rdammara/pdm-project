# Data Schema Documentation

This document describes the expected data schema for the predictive maintenance project.

## Sensor Data Schema

### Table: `sensor_readings`

Expected structure for sensor data:

| Column Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `machine_id` | VARCHAR/INT | Unique identifier for each machine | M001, 12345 |
| `timestamp` | DATETIME | Recording timestamp | 2023-01-15 14:30:00 |
| `sensor_1` | FLOAT | Temperature sensor (°C) | 75.5 |
| `sensor_2` | FLOAT | Vibration sensor (mm/s) | 2.3 |
| `sensor_3` | FLOAT | Pressure sensor (bar) | 120.5 |
| `sensor_4` | FLOAT | RPM (revolutions per minute) | 1500 |
| `sensor_5` | FLOAT | Current (Amperes) | 15.2 |
| `sensor_6` | FLOAT | Voltage (Volts) | 220.5 |
| `operational_setting_1` | FLOAT | Load setting | 0.75 |
| `operational_setting_2` | FLOAT | Speed setting | 0.85 |
| `operational_setting_3` | FLOAT | Environment temperature (°C) | 25.0 |

### Notes:
- Timestamps should be in UTC or clearly documented timezone
- Sensor readings should be numeric (float/double)
- Missing values should be represented as NULL, not special codes
- Frequency: Typically 1 reading per minute to 1 reading per hour

## Failure Events Schema

### Table: `failure_events`

Expected structure for failure events:

| Column Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `event_id` | INT | Unique event identifier | 1001 |
| `machine_id` | VARCHAR/INT | Machine identifier (FK to sensor_readings) | M001 |
| `failure_timestamp` | DATETIME | When failure occurred | 2023-02-20 10:15:00 |
| `failure_type` | VARCHAR | Type/component of failure | Bearing, Motor, Pump |
| `failure_mode` | VARCHAR | Mode of failure (optional) | Wear, Crack, Overheating |
| `downtime_hours` | FLOAT | Hours of downtime | 4.5 |
| `severity` | VARCHAR | Severity level (optional) | Critical, Major, Minor |
| `root_cause` | TEXT | Description of root cause (optional) | Bearing degradation due to... |

### Notes:
- Each failure event must have a valid machine_id
- failure_timestamp must be >= earliest sensor reading for that machine
- failure_type is used for multi-class classification (optional)
- downtime_hours helps assess business impact

## Maintenance Logs Schema (Optional)

### Table: `maintenance_logs`

If maintenance logs are available:

| Column Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `maintenance_id` | INT | Unique maintenance identifier | 5001 |
| `machine_id` | VARCHAR/INT | Machine identifier | M001 |
| `maintenance_timestamp` | DATETIME | When maintenance was performed | 2023-01-10 08:00:00 |
| `maintenance_type` | VARCHAR | Type of maintenance | Preventive, Corrective, Inspection |
| `parts_replaced` | VARCHAR | Parts replaced (comma-separated) | Bearing, Oil filter |
| `cost` | FLOAT | Cost of maintenance (optional) | 1500.00 |
| `duration_hours` | FLOAT | Duration of maintenance | 2.0 |
| `notes` | TEXT | Additional notes | Routine inspection, all OK |

## Machine Metadata Schema (Optional)

### Table: `machine_metadata`

Static information about machines:

| Column Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `machine_id` | VARCHAR/INT | Machine identifier (PK) | M001 |
| `machine_type` | VARCHAR | Type/model of machine | Centrifugal Pump XY-200 |
| `manufacturer` | VARCHAR | Manufacturer name | ACME Corp |
| `installation_date` | DATE | When machine was installed | 2020-05-15 |
| `location` | VARCHAR | Physical location | Building A, Floor 2 |
| `capacity` | FLOAT | Rated capacity | 1000.0 |
| `criticality` | VARCHAR | Business criticality | High, Medium, Low |

## Data Quality Requirements

### Sensor Data
- **Completeness**: At least 80% of expected readings present
- **Consistency**: Sensor values within expected physical ranges
- **Timeliness**: Regular sampling intervals (no large gaps)
- **Accuracy**: Calibrated sensors with known error margins

### Failure Events
- **Completeness**: All significant failures recorded
- **Consistency**: Timestamps align with sensor data timeline
- **Accuracy**: Failure types correctly classified
- **Traceability**: Link to root cause analysis where available

## Sample Data Format

### CSV Format for Sensor Data
```csv
machine_id,timestamp,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6
M001,2023-01-01 00:00:00,75.2,2.1,120.3,1500,15.1,220.2
M001,2023-01-01 00:01:00,75.4,2.2,120.5,1502,15.2,220.3
M002,2023-01-01 00:00:00,72.1,1.9,118.5,1480,14.8,219.8
```

### CSV Format for Failure Events
```csv
machine_id,failure_timestamp,failure_type,downtime_hours
M001,2023-02-15 14:30:00,Bearing,4.5
M003,2023-03-20 09:15:00,Motor,8.0
M001,2023-05-10 16:45:00,Pump,3.5
```

## Data Extraction Guidelines

### SQL Query Examples

#### Extract Sensor Data
```sql
SELECT 
    machine_id,
    timestamp,
    sensor_1,
    sensor_2,
    sensor_3,
    sensor_4,
    sensor_5,
    sensor_6,
    operational_setting_1,
    operational_setting_2,
    operational_setting_3
FROM sensor_readings
WHERE timestamp BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY machine_id, timestamp;
```

#### Extract Failure Events
```sql
SELECT 
    machine_id,
    failure_timestamp,
    failure_type,
    downtime_hours
FROM failure_events
WHERE failure_timestamp BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY machine_id, failure_timestamp;
```

## Data Validation Checklist

Before using data for modeling:
- [ ] All machine_ids in failure_events exist in sensor_readings
- [ ] No future timestamps (all timestamps <= current time)
- [ ] Sensor values within reasonable ranges
- [ ] No duplicate timestamps for same machine
- [ ] Failure timestamps align with sensor data timeline
- [ ] Missing values handled appropriately
- [ ] Outliers investigated and documented
- [ ] Data types are correct and consistent

## Contact

For questions about data schema or data quality issues, please contact the data engineering team.
