# Predictive Maintenance Project Template

A comprehensive Python + Jupyter Notebook project template for predictive maintenance on industrial machines, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## Project Overview

This project template provides a structured approach to building predictive maintenance models for industrial equipment. It includes:

- **Data Extraction**: SQL Server connectivity and data extraction scripts
- **Event-to-Time-Series Labeling**: Tools for computing Remaining Useful Life (RUL) and failure prediction labels
- **Feature Engineering**: Automated feature generation from sensor time-series data
- **Machine Learning Models**: XGBoost-based regression (RUL prediction) and classification (failure prediction)
- **Evaluation & Reporting**: Comprehensive model evaluation metrics and visualization tools

## Project Structure

```
pdm-project/
│
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration
│   └── database_config.yaml    # Database connection settings (template)
│
├── data/                       # Data directory (gitignored except .gitkeep)
│   ├── raw/                    # Raw data from SQL Server
│   ├── interim/                # Intermediate processed data
│   ├── processed/              # Final processed data ready for modeling
│   └── external/               # External reference data
│
├── notebooks/                  # Jupyter notebooks following CRISP-DM phases
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
│
├── src/                        # Source code
│   ├── data_extraction/        # SQL Server data extraction
│   │   ├── __init__.py
│   │   └── sql_extractor.py
│   │
│   ├── data_processing/        # Data cleaning and labeling
│   │   ├── __init__.py
│   │   ├── event_labeler.py
│   │   └── rul_calculator.py
│   │
│   ├── feature_engineering/    # Feature generation
│   │   ├── __init__.py
│   │   ├── time_series_features.py
│   │   └── statistical_features.py
│   │
│   ├── modeling/               # Model training and prediction
│   │   ├── __init__.py
│   │   ├── xgb_regressor.py
│   │   └── xgb_classifier.py
│   │
│   ├── evaluation/             # Model evaluation
│   │   ├── __init__.py
│   │   ├── regression_metrics.py
│   │   └── classification_metrics.py
│   │
│   ├── reporting/              # Reporting utilities
│   │   ├── __init__.py
│   │   └── report_generator.py
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── models/                     # Saved models (gitignored)
│
├── reports/                    # Generated reports and figures
│   ├── figures/               # Plots and visualizations
│   └── evaluation/            # Evaluation reports
│
├── docs/                       # Documentation
│   ├── data_schema.md
│   └── model_documentation.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## CRISP-DM Workflow

This project follows the CRISP-DM methodology:

1. **Business Understanding**: Define predictive maintenance objectives and requirements
2. **Data Understanding**: Explore sensor data, failure events, and maintenance logs
3. **Data Preparation**: Clean data, compute RUL labels, handle missing values
4. **Modeling**: Train XGBoost models for regression and classification tasks
5. **Evaluation**: Assess model performance using appropriate metrics
6. **Deployment**: Package models and create inference pipelines

## Getting Started

### Prerequisites

- Python 3.8 or higher
- SQL Server (or compatible database with ODBC support)
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rdammara/pdm-project.git
cd pdm-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure database connection:
```bash
cp config/database_config.yaml.example config/database_config.yaml
# Edit database_config.yaml with your SQL Server credentials
```

### Usage

#### 1. Data Extraction

Extract data from SQL Server:
```python
from src.data_extraction.sql_extractor import SQLExtractor

extractor = SQLExtractor(config_path='config/database_config.yaml')
extractor.extract_sensor_data(start_date='2023-01-01', end_date='2023-12-31')
extractor.extract_failure_events()
```

#### 2. Data Processing and Labeling

Compute RUL labels for time-series data:
```python
from src.data_processing.rul_calculator import RULCalculator

rul_calc = RULCalculator()
labeled_data = rul_calc.compute_rul(sensor_data, failure_events)
```

#### 3. Feature Engineering

Generate features from time-series data:
```python
from src.feature_engineering.time_series_features import TimeSeriesFeatureGenerator

feature_gen = TimeSeriesFeatureGenerator()
features = feature_gen.generate_features(labeled_data, window_size=50)
```

#### 4. Model Training

Train XGBoost models:
```python
from src.modeling.xgb_regressor import XGBoostRULRegressor
from src.modeling.xgb_classifier import XGBoostFailureClassifier

# RUL Regression
rul_model = XGBoostRULRegressor()
rul_model.train(X_train, y_train)

# Failure Classification
classifier = XGBoostFailureClassifier()
classifier.train(X_train, y_train_binary)
```

#### 5. Evaluation

Evaluate model performance:
```python
from src.evaluation.regression_metrics import evaluate_regression
from src.evaluation.classification_metrics import evaluate_classification

# Regression metrics
regression_results = evaluate_regression(y_true, y_pred)

# Classification metrics
classification_results = evaluate_classification(y_true, y_pred_class)
```

#### 6. Using Jupyter Notebooks

Work through the CRISP-DM workflow interactively:
```bash
jupyter notebook notebooks/
```

Start with `01_business_understanding.ipynb` and proceed through each phase sequentially.

## Key Features

### Event-to-Time-Series Labeling

The project includes sophisticated labeling mechanisms:
- **RUL (Remaining Useful Life)**: Continuous target for regression models
- **Failure Window Labeling**: Binary labels for classification within a prediction horizon
- **Multi-class Failure Types**: Support for different failure modes

### Feature Engineering

Automated feature generation including:
- Rolling statistics (mean, std, min, max)
- Trend features (slopes, rates of change)
- Frequency domain features (FFT-based)
- Lag features
- Domain-specific sensor features

### XGBoost Models

Pre-configured XGBoost models with:
- Hyperparameter optimization via grid search
- Cross-validation
- Feature importance analysis
- SHAP value computation for interpretability

### Comprehensive Evaluation

Evaluation metrics include:
- **Regression**: RMSE, MAE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Custom PdM Metrics**: Early detection rate, false alarm rate, advance warning time

## Configuration

Edit `config/config.yaml` to customize:
- Feature engineering parameters
- Model hyperparameters
- Data processing settings
- Prediction horizons and thresholds

Example configuration:
```yaml
feature_engineering:
  window_size: 50
  rolling_windows: [10, 20, 50]
  
modeling:
  rul_regression:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
  
  failure_classification:
    max_depth: 4
    learning_rate: 0.05
    n_estimators: 200
    
prediction:
  horizon_days: 30
  threshold_probability: 0.7
```

## Data Schema

Expected data schema for sensor data and failure events is documented in `docs/data_schema.md`.

### Sensor Data Format
- `machine_id`: Unique machine identifier
- `timestamp`: Recording timestamp
- `sensor_1`, `sensor_2`, ...: Sensor readings
- `operational_setting_1`, ...`: Operating conditions

### Failure Events Format
- `machine_id`: Unique machine identifier
- `failure_timestamp`: When failure occurred
- `failure_type`: Type of failure (optional)
- `downtime_hours`: Duration of downtime

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This template is designed for industrial predictive maintenance applications and follows best practices from:
- CRISP-DM methodology
- NASA's Turbofan Engine Degradation dataset patterns
- Industry standards for PHM (Prognostics and Health Management)

## Contact

For questions or support, please open an issue on GitHub.
