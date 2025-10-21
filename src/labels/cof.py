
import pandas as pd
import numpy as np

def label_future_breakdown(df: pd.DataFrame, time_col='timestamp', id_col='machine_id', breakdown_col='breakdown', horizon_minutes=30) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df = df.sort_values([id_col, time_col])
    def lab(g: pd.DataFrame):
        times = g[time_col].values
        bd = g[breakdown_col].astype(int).values if breakdown_col in g.columns else np.zeros(len(g), dtype=int)
        out = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            t0 = times[i]
            j = i + 1
            while j < len(g) and (times[j] - t0) <= np.timedelta64(horizon_minutes, 'm'):
                if bd[j] == 1:
                    out[i] = 1
                    break
                j += 1
        return out
    df['CoF'] = df.groupby(id_col, group_keys=False).apply(lambda g: pd.Series(lab(g), index=g.index)).astype(int)
    return df
