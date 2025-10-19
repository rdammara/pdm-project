# Model Documentation

This document provides comprehensive documentation for the predictive maintenance models.

## Overview

The predictive maintenance system consists of two complementary models:

1. **RUL Regression Model**: Predicts the Remaining Useful Life (RUL) in time units
2. **Failure Classification Model**: Predicts binary failure risk within a prediction horizon

Both models use XGBoost (Extreme Gradient Boosting) algorithm with carefully engineered features from sensor time-series data.

---

## Model 1: RUL Regression

### Purpose
Predict the continuous remaining useful life (RUL) of equipment to support long-term maintenance planning.

### Algorithm
- **Type**: XGBoost Regressor
- **Objective**: reg:squarederror (minimizing squared error)
- **Loss Function**: Mean Squared Error (MSE)

### Default Hyperparameters
```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'objective': 'reg:squarederror',
    'random_state': 42
}
```

### Target Variable
- **Name**: RUL (Remaining Useful Life)
- **Type**: Continuous numeric
- **Unit**: Hours (or cycles, depending on data)
- **Range**: 0 to rul_clip_max (default: 150)
- **Calculation**: Time difference between current timestamp and next failure event

### Input Features
Features are automatically generated from raw sensor data:

#### Time-Series Features
- Rolling statistics (mean, std, min, max, range)
- Lag features (previous values)
- Rate of change (first and second derivatives)
- Trend features (linear slope over window)

#### Statistical Features
- Interaction features (sensor cross-products)
- Ratio features
- Polynomial features
- Cumulative statistics

#### Domain Features
- Operating time
- Time since last measurement

### Performance Metrics
- **RMSE** (Root Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Interpretable error
- **R²** (Coefficient of Determination): Variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative error

### Target Performance
- RMSE < 15 time units
- R² > 0.7
- MAE < 10 time units

### Use Cases
- Long-term maintenance scheduling
- Budget planning for repairs
- Spare parts inventory management
- Equipment lifecycle analysis

---

## Model 2: Failure Classification

### Purpose
Predict binary failure risk within a defined prediction horizon to enable short-term preventive actions.

### Algorithm
- **Type**: XGBoost Classifier
- **Objective**: binary:logistic
- **Loss Function**: Binary cross-entropy

### Default Hyperparameters
```python
{
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,  # Adjusted for class imbalance
    'objective': 'binary:logistic',
    'random_state': 42
}
```

### Target Variable
- **Name**: failure_label
- **Type**: Binary (0 or 1)
- **Definition**: 
  - 1 = Failure expected within prediction horizon
  - 0 = No imminent failure expected
- **Prediction Horizon**: Configurable (default: 30 days)

### Input Features
Same feature set as RUL regression model for consistency.

### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

#### Custom PdM Metrics
- **False Alarm Rate**: Rate of false positives
- **Early Detection Rate**: Failures detected with adequate advance warning
- **Average Advance Warning Time**: Mean time before failure that prediction was made

### Target Performance
- F1-Score > 0.75
- Recall > 0.80 (minimize false negatives)
- ROC-AUC > 0.85
- False Alarm Rate < 10%

### Classification Threshold
- Default: 0.5
- Optimizable based on business priorities:
  - Higher threshold → fewer false alarms, more missed failures
  - Lower threshold → fewer missed failures, more false alarms

### Use Cases
- Immediate maintenance alerts
- Emergency response planning
- Short-term resource allocation
- Safety-critical failure prevention

---

## Feature Engineering

### Rolling Window Features
For each sensor, compute statistics over multiple window sizes:
- **Window sizes**: [10, 20, 50] time steps
- **Statistics**: mean, std, min, max, range

Example features:
- `sensor_1_rolling_mean_50`
- `sensor_2_rolling_std_20`
- `sensor_3_rolling_range_10`

### Lag Features
Previous values to capture temporal dependencies:
- **Lag sizes**: [1, 2, 5, 10] time steps

Example features:
- `sensor_1_lag_1` (previous value)
- `sensor_1_lag_10` (value 10 steps ago)

### Rate of Change Features
Derivatives to capture trends:
- **First difference**: `sensor_1_diff`
- **Second difference**: `sensor_1_diff2`
- **Percentage change**: `sensor_1_pct_change`

### Trend Features
Linear slope over rolling windows:
- `sensor_1_trend_20`

### Interaction Features
Cross-products between sensors:
- `sensor_1_x_sensor_2`
- `sensor_3_x_sensor_4`

### Ratio Features
Ratios between sensors:
- `sensor_1_div_sensor_2`

---

## Model Training Pipeline

### 1. Data Preparation
```python
# Load data
sensor_data = pd.read_csv('data/raw/sensor_data.csv')
failure_events = pd.read_csv('data/raw/failure_events.csv')

# Compute RUL labels
rul_calculator = RULCalculator(clip_max=150)
sensor_data = rul_calculator.compute_rul(sensor_data, failure_events)

# Compute binary labels
labeler = EventLabeler(prediction_horizon=30, time_unit='days')
sensor_data = labeler.label_failure_window(sensor_data, failure_events)
```

### 2. Feature Engineering
```python
# Generate features
feature_generator = TimeSeriesFeatureGenerator(
    window_sizes=[10, 20, 50],
    lag_sizes=[1, 2, 5, 10]
)
sensor_data = feature_generator.generate_features(sensor_data)
```

### 3. Train/Validation/Test Split
- Training: 70%
- Validation: 10% (for early stopping and threshold optimization)
- Test: 20% (final evaluation)

### 4. Model Training
```python
# RUL Regression
rul_model = XGBoostRULRegressor(params=rul_params)
rul_model.train(X_train, y_train_rul, X_val, y_val_rul)
rul_model.save_model('models/rul_regressor.pkl')

# Failure Classification
classifier = XGBoostFailureClassifier(params=class_params)
classifier.train(X_train, y_train_class, X_val, y_val_class)
classifier.save_model('models/failure_classifier.pkl')
```

---

## Model Evaluation

### Cross-Validation
5-fold cross-validation used during hyperparameter tuning to ensure generalization.

### Holdout Test Set
Final evaluation on completely unseen test set (20% of data).

### Evaluation Visualizations
- **RUL Model**: Predictions vs actual, residual plots, error by RUL range
- **Classification Model**: Confusion matrix, ROC curve, precision-recall curve, threshold analysis

---

## Model Inference

### Making Predictions
```python
# Load models
rul_model = XGBoostRULRegressor()
rul_model.load_model('models/rul_regressor.pkl')

classifier = XGBoostFailureClassifier()
classifier.load_model('models/failure_classifier.pkl')

# Prepare new data (with feature engineering)
new_data = prepare_features(raw_sensor_data)

# Make predictions
rul_predictions = rul_model.predict(new_data)
failure_probabilities = classifier.predict_proba(new_data)
failure_labels = classifier.predict(new_data)
```

### Prediction Outputs
For each timestamp and machine:
- **RUL**: Estimated remaining useful life (numeric)
- **Failure Probability**: Probability of failure within horizon (0-1)
- **Failure Label**: Binary prediction (0 or 1)

---

## Model Maintenance

### Monitoring
- Track prediction accuracy over time
- Monitor feature distributions for drift
- Log false positives and false negatives
- Collect feedback from maintenance team

### Retraining
Recommended retraining schedule:
- **Monthly**: If significant new data available
- **Quarterly**: Standard retraining cycle
- **Ad-hoc**: If performance degradation detected

### Model Versioning
- Use semantic versioning (e.g., v1.2.3)
- Store model artifacts with metadata
- Track training data and hyperparameters
- Document changes between versions

---

## Limitations and Considerations

### Known Limitations
1. **Data Quality**: Model performance depends on sensor data quality
2. **Class Imbalance**: Failures are rare events, affecting classification
3. **Concept Drift**: Machine behavior may change over time
4. **Feature Engineering**: Requires domain knowledge for sensor selection

### Assumptions
- Sensor data is representative of machine health
- Failure events are accurately recorded
- Past failure patterns predict future failures
- Operating conditions remain relatively stable

### Edge Cases
- **New machines**: No historical failure data
- **Rare failure types**: Insufficient training examples
- **Sensor failures**: Missing or incorrect readings
- **Maintenance effects**: Post-maintenance behavior changes

---

## Business Impact

### Expected Benefits
- **Reduced Downtime**: 30% reduction in unplanned downtime
- **Cost Savings**: Lower maintenance costs through optimization
- **Extended Lifespan**: 15% increase in equipment life
- **Safety**: Prevent catastrophic failures

### Key Metrics for ROI
- Maintenance cost per machine
- Downtime hours avoided
- Parts replacement optimization
- Labor efficiency improvement

---

## References

### Technical Papers
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- NASA Prognostics Center of Excellence datasets
- PHM (Prognostics and Health Management) best practices

### Related Resources
- Project GitHub: [rdammara/pdm-project](https://github.com/rdammara/pdm-project)
- XGBoost Documentation: https://xgboost.readthedocs.io/

---

## Contact

For questions about the models or technical support:
- Data Science Team
- GitHub Issues: https://github.com/rdammara/pdm-project/issues
