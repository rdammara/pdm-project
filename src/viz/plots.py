
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_residuals(y_true, y_pred, title='Residuals'):
    res = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, res, alpha=0.3)
    plt.axhline(0, linestyle='--')
    plt.title(title)
    plt.xlabel('True'); plt.ylabel('Residual (pred-true)')
    plt.tight_layout()
    plt.show()

def bar_metrics(df: pd.DataFrame, metrics=('rmse','mae','r2')):
    agg = df.groupby('algo')[list(metrics)].mean()
    plt.figure()
    agg.plot(kind='bar')
    plt.title('Metrics by Algorithm')
    plt.tight_layout()
    plt.show()
