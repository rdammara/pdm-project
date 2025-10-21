
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

def parse_and_sort(df: pd.DataFrame, time_col: str='timestamp', id_col: str='machine_id') -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df = df.dropna(subset=[time_col]).sort_values([id_col, time_col]).reset_index(drop=True)
    return df

def resample_regular(df: pd.DataFrame, freq: Optional[str], time_col='timestamp', id_col='machine_id') -> pd.DataFrame:
    if not freq:
        return df.copy()
    non_num = df.select_dtypes(exclude='number').columns.tolist()
    non_num = list(dict.fromkeys([c for c in non_num if c != time_col] + [id_col, '__line' if '__line' in df.columns else None]))
    non_num = [c for c in non_num if c is not None]
    num = [c for c in df.select_dtypes(include='number').columns if c not in [id_col]]
    out = []
    for gid, g in df.groupby(id_col):
        g = g.set_index(time_col).sort_index()
        g_num = g[num].resample(freq).mean()
        g_non = g[non_num].resample(freq).ffill().bfill()
        g_out = pd.concat([g_non, g_num], axis=1)
        g_out[id_col] = gid
        out.append(g_out.reset_index())
    return pd.concat(out, ignore_index=True).sort_values([id_col, time_col]).reset_index(drop=True)

def align_oee_events(df: pd.DataFrame, oee_df: pd.DataFrame, time_col='timestamp', id_col='machine_id', event_col='breakdown') -> pd.DataFrame:
    """Left-join nearest OEE event within same timestamp (assumes oee_df has same id/time granularity)."""
    if oee_df is None or oee_df.empty:
        return df
    oee = oee_df[[id_col, time_col, event_col]].copy()
    return df.merge(oee, on=[id_col, time_col], how='left')

def fill_missing_numeric(df: pd.DataFrame, method: str='ffill') -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include='number').columns
    if method == 'ffill':
        df[num_cols] = df.groupby(0, group_keys=False).apply(lambda g: g[num_cols].ffill().bfill())
    elif method == 'median':
        med = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(med)
    else:
        df[num_cols] = df[num_cols].fillna(0)
    return df
