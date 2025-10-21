
import pandas as pd
import numpy as np
from typing import Iterable

def add_lags(g: pd.DataFrame, cols: Iterable[str], lags=(1,2,3)) -> pd.DataFrame:
    g = g.copy()
    for L in lags:
        for c in cols:
            g[f'{c}_lag{L}'] = g[c].shift(L)
    return g

def add_roll_stats(g: pd.DataFrame, cols: Iterable[str], windows=(3,5,15)) -> pd.DataFrame:
    g = g.copy()
    for w in windows:
        for c in cols:
            g[f'{c}_roll{w}_mean'] = g[c].rolling(w).mean()
            g[f'{c}_roll{w}_std']  = g[c].rolling(w).std()
            g[f'{c}_roll{w}_min']  = g[c].rolling(w).min()
            g[f'{c}_roll{w}_max']  = g[c].rolling(w).max()
    return g

def add_diffs(g: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    g = g.copy()
    for c in cols:
        g[f'{c}_diff1'] = g[c].diff(1)
    return g

def per_machine_apply(df: pd.DataFrame, id_col: str, fn) -> pd.DataFrame:
    return pd.concat([fn(g.copy()) for _, g in df.groupby(id_col)], ignore_index=True)
