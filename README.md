
# 🧠 Predictive Maintenance (PDM) — PVC Extruder Line 10 & 20

This repository implements a **Predictive Maintenance (PDM)** system for PVC pipe extruder machines using both **Regression (RUL)** and **Classification (CoF)** approaches.

It follows the **CRISP-DM methodology**, and compares three machine learning algorithms:
- **XGBoost**
- **LSTM (Long Short-Term Memory)**
- **CNN (Convolutional Neural Network)**

---

## 📂 Folder Structure

```
pdm-project/
│
├── data/
│   ├── raw/                 # Line10 & Line20 source data (CSV/SQL/Influx)
│   ├── interim/             # Temporary merged/cleaned data
│   └── processed/           # Final labeled, scaled, and split datasets
│
├── notebooks/
│   ├── 00_overview.ipynb    # Project setup & configuration
│   ├── RUL/                 # Regression (Remaining Useful Life)
│   │   ├── 01_eda_data_prep_RUL.ipynb
│   │   ├── 02_feature_engineering_RUL.ipynb
│   │   ├── 03_train_eval_RUL.ipynb
│   │   └── 04_model_comparison_RUL.ipynb
│   └── CoF/                 # Classification (Chance of Failure)
│       ├── 01_eda_data_prep_CoF.ipynb
│       ├── 02_feature_engineering_CoF.ipynb
│       ├── 03_train_eval_CoF.ipynb
│       └── 04_model_comparison_CoF.ipynb
│
├── src/
│   ├── io_/                 # Data loaders (CSV, SQL, InfluxDB)
│   ├── prep/                # Cleaning, resampling, alignment
│   ├── features/            # Feature engineering (lags, rolling stats, diffs)
│   ├── labels/              # Label builders (RUL & CoF)
│   ├── models/              # Model definitions (XGB, LSTM, CNN)
│   ├── train/               # Training loops, metrics, utilities
│   └── viz/                 # Visualization helpers
│
├── configs/                 # YAML-based configuration files
├── experiments/             # Logged results & artifacts for each task
│   ├── RUL/
│   └── CoF/
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Create Environment
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2️⃣ Prepare Data
Place your raw data here:
```
data/raw/Line10/DM_Machine_Learning_Line_10.csv
data/raw/Line20/DM_Machine_Learning_Line_20.csv
```

### 3️⃣ Run the Pipeline
1. Open **`notebooks/00_overview.ipynb`** to initialize configs and folders.
2. Run each phase:
   - **RUL**: notebooks in `/RUL`
   - **CoF**: notebooks in `/CoF`
3. Results (metrics, artifacts) are saved under `/experiments/`.

---

## 📊 Evaluation Metrics

| Task | Algorithm | Metrics |
|------|------------|----------|
| **RUL (Regression)** | CNN, LSTM, XGBoost | RMSE, MAE, R², NASA Score, Silhouette |
| **CoF (Classification)** | CNN, LSTM, XGBoost | F1-Score, Recall, ROC-AUC |

---

## 🧩 Key Features

- Consistent preprocessing and feature engineering across models  
- Unified training + logging for reproducible experiments  
- Configurable YAML for tasks, algorithms, and data sources  
- Modular codebase (src/) for scalability and maintenance  
- Ready for cloud deployment and MLflow tracking

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch (`feature/add-model-x`)  
3. Commit your changes  
4. Push to your branch and open a PR  

---

```
