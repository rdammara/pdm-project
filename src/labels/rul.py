
import pandas as pd
import numpy as np

def compute_rul(df: pd.DataFrame, time_col='timestamp', id_col='machine_id', failure_col='breakdown') -> pd.DataFrame:
    df = df.copy()
    if failure_col not in df.columns:
        df['RUL'] = np.nan
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df = df.sort_values([id_col, time_col])
    def minutes_until_next_failure(g: pd.DataFrame):
        times = g[time_col].view('int64') // 10**9
        nxt = np.full(len(g), np.nan)
        next_t = None
        for i in range(len(g)-1, -1, -1):
            if g.iloc[i][failure_col] in [1, True]:
                next_t = times[i]
            if next_t is not None:
                nxt[i] = (next_t - times[i]) / 60.0
        return nxt
    df['RUL'] = df.groupby(id_col, group_keys=False).apply(lambda g: pd.Series(minutes_until_next_failure(g), index=g.index))
    return df
