
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, recall_score, roc_auc_score

def nasa_score(y_true, y_pred, c=10.0):
    e = y_pred - y_true
    over = e >= 0
    under = ~over
    s = np.zeros_like(e, dtype=float)
    s[over]  = np.exp(-e[over] / c) - 1.0
    s[under] = np.exp(e[under] / c) - 1.0
    return float(np.mean(s**2))

def silhouette_safe(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return float(silhouette_score(X, labels))
    except Exception:
        pass
    return float('nan')

def regression_metrics(y_true, y_pred, X_for_sil=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    nasa = float(nasa_score(y_true, y_pred))
    sil  = float('nan')
    if X_for_sil is not None:
        q = np.quantile(y_true, [0.33, 0.66])
        labels = np.digitize(y_true, q)
        sil = silhouette_safe(X_for_sil, labels)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nasa': nasa, 'silhouette': sil}

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float('nan')
    return {'f1': f1, 'recall': rec, 'roc_auc': auc}
