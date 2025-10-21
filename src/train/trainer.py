
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .metrics import regression_metrics, classification_metrics
from ..models.xgboost_ import make_xgb
from ..models.lstm import make_lstm
from ..models.cnn import make_cnn

def build_windows(X: pd.DataFrame, y: np.ndarray, window=20, stride=1, id_col='machine_id'):
    feats = [c for c in X.columns if c not in [id_col, 'timestamp','__line']]
    Xs, ys = [], []
    for gid, gX in X.groupby(id_col):
        idx = gX.index.values
        for start in range(0, len(gX) - window + 1, stride):
            end = start + window
            sl = gX.iloc[start:end]
            Xs.append(sl[feats].values)
            ys.append(y[idx[end-1]])
    Xs = np.stack(Xs, axis=0) if Xs else np.empty((0, window, len(feats)))
    ys = np.array(ys)
    return Xs, ys, feats

def train_regression(algo: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray, window=20, stride=1, seed=42) -> Dict:
    if algo == 'xgboost':
        model = make_xgb('regression', random_state=seed)
        feats = [c for c in Xtr.columns if c not in ['machine_id','timestamp','__line']]
        model.fit(Xtr[feats], ytr)
        pred = model.predict(Xte[feats])
        return {'pred': pred, 'feats': feats}
    elif algo in {'lstm','cnn'}:
        Xtr_w, ytr_w, feats = build_windows(Xtr, ytr, window, stride)
        Xva_w, yva_w, _    = build_windows(Xva, yva, window, stride)
        Xte_w, yte_w, _    = build_windows(Xte, yte, window, stride)
        input_shape = (window, len(feats))
        if algo == 'lstm':
            model = make_lstm(input_shape, task='regression')
        else:
            model = make_cnn(input_shape, task='regression')
        model.fit(Xtr_w, ytr_w, validation_data=(Xva_w, yva_w), epochs=20, batch_size=128, verbose=0)
        pred = model.predict(Xte_w).ravel()
        return {'pred': pred, 'feats': feats, 'y_true_seq': yte_w}
    else:
        raise ValueError('Unknown algo')

def train_classification(algo: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray, window=20, stride=1, seed=42) -> Dict:
    if algo == 'xgboost':
        feats = [c for c in Xtr.columns if c not in ['machine_id','timestamp','__line']]
        pos = max(1.0, (len(ytr) - ytr.sum()) / max(1.0, ytr.sum()))
        model = make_xgb('classification', random_state=seed, scale_pos_weight=pos)
        model.fit(Xtr[feats], ytr)
        prob = model.predict_proba(Xte[feats])[:,1]
        return {'prob': prob, 'feats': feats}
    elif algo in {'lstm','cnn'}:
        Xtr_w, ytr_w, feats = build_windows(Xtr, ytr, window, stride)
        Xva_w, yva_w, _    = build_windows(Xva, yva, window, stride)
        Xte_w, yte_w, _    = build_windows(Xte, yte, window, stride)
        input_shape = (window, len(feats))
        if algo == 'lstm':
            from ..models.lstm import make_lstm
            model = make_lstm(input_shape, task='classification')
        else:
            from ..models.cnn import make_cnn
            model = make_cnn(input_shape, task='classification')
        model.fit(Xtr_w, ytr_w, validation_data=(Xva_w, yva_w), epochs=20, batch_size=128, verbose=0)
        prob = model.predict(Xte_w).ravel()
        return {'prob': prob, 'feats': feats, 'y_true_seq': yte_w}
    else:
        raise ValueError('Unknown algo')
