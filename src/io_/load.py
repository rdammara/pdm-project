
from pathlib import Path
import pandas as pd
from typing import Optional, Dict

def load_csv(path: str | Path, line: Optional[int]=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    if line is not None:
        df['__line'] = int(line)
    return df

def load_sql(conn_str: str, query: str, line: Optional[int]=None) -> pd.DataFrame:
    """Placeholder using pandas read_sql; user must have appropriate driver installed."""
    import pandas as pd
    import sqlalchemy
    engine = sqlalchemy.create_engine(conn_str)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if line is not None:
        df['__line'] = int(line)
    return df

def load_influx(url: str, token: str, org: str, bucket: str, flux_query: str, line: Optional[int]=None) -> pd.DataFrame:
    """Return a tidy DataFrame from InfluxDB 2.x. Requires 'influxdb-client' package."""
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.flux_table import FluxStructureEncoder
    client = InfluxDBClient(url=url, token=token, org=org)
    tables = client.query_api().query(flux_query, org=org)
    # Convert to pandas
    df = pd.DataFrame([
        {**r.values, "_time": r.get_time(), "_value": r.get_value()}
        for t in tables for r in t.records
    ])
    if line is not None:
        df['__line'] = int(line)
    return df

def load_from_config(cfg: Dict) -> pd.DataFrame:
    source = cfg.get('source', 'csv').lower()
    line = cfg.get('line')
    if source == 'csv':
        return load_csv(cfg['path'], line=line)
    elif source == 'sql':
        return load_sql(cfg['conn'], cfg['query'], line=line)
    elif source == 'influx':
        return load_influx(cfg['url'], cfg['token'], cfg['org'], cfg['bucket'], cfg['query'], line=line)
    else:
        raise ValueError(f"Unknown source: {source}")
