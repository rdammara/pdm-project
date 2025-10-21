"""SQL Server data extraction module."""

"""
Extract DM_Machine_Learning dataset from SQL Server using .env + YAML config.
- Matches the columns seen in DM_Machine_Learning_Line_10.csv
- Filters by Mesin (line), date range, and writes CSV (optionally Parquet)
"""

import os
import yaml
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

RAW_DIR = "data/raw"

# --------- config helpers ----------
def load_cfg(path="config/database_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def build_engine():
    load_dotenv()  # loads .env in project root

    cfg = load_cfg()

    driver   = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    server   = os.getenv("DB_SERVER")              # e.g. DGP-DBWHSVR-01\\DBWHouse
    port     = os.getenv("DB_PORT", "1433")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")

    if not all([server, database, username, password]):
        raise RuntimeError("Missing one or more env vars: DB_SERVER, DB_NAME, DB_USER, DB_PASS")

    encrypt = "yes" if cfg.get("encrypt", True) else "no"
    tsc     = "yes" if cfg.get("trust_server_certificate", False) else "no"

    # For a named instance like HOST\\INSTANCE, do NOT append ,port
    server_part = f"{server}" if ("\\" in server) else f"{server},{port}"

    odbc = (
        "DRIVER={driver};SERVER={server};DATABASE={db};UID={uid};PWD={pwd};"
        "Encrypt={enc};TrustServerCertificate={tsc};Timeout=5;Connection Timeout=5;"
    ).format(
        driver=driver, server=server_part, db=database,
        uid=username, pwd=password, enc=encrypt, tsc=tsc
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")

# --------- column list (matches your CSV) ----------
COLS = [
    "Timestamp",
    "Mesin",
    "EXT_10.Extruder_Load",
    "EXT_10.Machine_Hour",
    "EXT_10.Machine_Run",
    "EXT_10.Melt_Press",
    "EXT_10.Melt_Temp_1",
    "EXT_10.Motor_Extruder_Run",
    "EXT_10.Motor_Gearbox_Line_Run",
    "EXT_10.Motor_Gearbox_Run",
    "EXT_10.Motor_Grooved_Sleeve_Run",
    "EXT_10.Motor_Haul_off_Run",
    "EXT_10.Motor_Winder_Run",
    "EXT_10.Screw_Speed",
    "EXT_10.Setpoint_Haul_off_Speed",
    "EXT_10.Setpoint_Line_Speed",
    "EXT_10.Setpoint_Winder_Speed",
    "EXT_10.Temperature_Barrel_1",
    "EXT_10.Temperature_Barrel_2",
    "EXT_10.Temperature_Barrel_3",
    "EXT_10.Temperature_Barrel_4",
    "EXT_10.Temperature_Barrel_5",
    "EXT_10.Temperature_Die_1",
    "EXT_10.Temperature_Die_2",
    "EXT_10.Temperature_Die_3",
    "EXT_10.Temperature_Die_4",
    "EXT_10.Temperature_Die_5",
    "EXT_10.Temperature_Grooved_Sleeve",
    "EXT_10.Temperature_Melt",
    "EXT_10.Vacuum_Press",
    "EXT_10.Vacuum_Press_Setpoint",
    "EXT_10.Vacuum_Pump_Run",
    "PM_Extruder_10.A_avg",
    "PM_Extruder_10.Frequency",
    "PM_Extruder_10.P",
    "PM_Extruder_10.Power_Factor",
    "PM_Extruder_10.V_avg",
    "PM_Extruder_10.W_tot",
    "Start_Time",
    "End_Time",
    "Level_1",
    "Level_2",
    "Level_3",
    "Detail",
    "rn",
    "Breakdown",
]

def bracket(c: str) -> str:
    """Bracket-escape SQL identifiers (handles dots/spaces)."""
    return f"[{c}]"

SELECT_COLS = ",\n  ".join(bracket(c) for c in COLS)

# --------- extraction ----------
def extract_dm(
    source_fq="dbo.DM_Machine_Learning_Line_10",     # ← CHANGE to your actual table/view
    line:int = 10,                      # Mesin filter (10 or 20)
    start_ts:str = "2025-01-01 00:00:00",
    end_ts:str   = "2025-02-01 23:59:59.999",
    out_csv:str  = "DM_Machine_Learning_Line_10_extract.csv",
    chunksize:int|None = 200_000,
    write_parquet: bool = False
):
    os.makedirs(RAW_DIR, exist_ok=True)
    engine = build_engine()

    # Ensure Timestamp comes first as datetime2
    sql = f"""
    SELECT
      CAST([Timestamp] AS datetime2) AS [Timestamp],
      {SELECT_COLS.replace('[Timestamp],', '').strip()}
    FROM {source_fq}
    WHERE [Mesin] = 10
      AND [Timestamp] BETWEEN '2025-01-01 00:00:00' AND '2025-02-01 23:59:59.999'
    ORDER BY [Timestamp] ASC
    """

    out_path = os.path.join(RAW_DIR, out_csv)
    total = 0
    last_chunk = None

    with engine.connect() as conn:
        if chunksize:
            iter_df = pd.read_sql_query(text(sql), con=conn,
                                        params={"line": line, "start": start_ts, "end": end_ts},
                                        chunksize=chunksize)
            first = True
            for i, chunk in enumerate(iter_df, 1):
                total += len(chunk)
                # write CSV
                chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
                first = False
                last_chunk = chunk
                if i % 5 == 0:
                    print(f"...written ~{total:,} rows")
        else:
            df = pd.read_sql_query(text(sql), con=conn,
                                   params={"line": line, "start": start_ts, "end": end_ts})
            total = len(df)
            df.to_csv(out_path, index=False)
            last_chunk = df

    if total == 0:
        print("⚠️ No rows matched filters.")
        return

    print(f"✅ Done. Rows: {total:,} → {out_path}")

    # Optional Parquet alongside CSV (requires pyarrow)
    if write_parquet and last_chunk is not None:
        try:
            import pyarrow  # noqa: F401
            parquet_path = out_path.replace(".csv", ".parquet")
            # If we streamed by chunks, re-read the CSV once and write Parquet
            if chunksize:
                full = pd.read_csv(out_path, parse_dates=["Timestamp"])
                full.to_parquet(parquet_path, index=False)
            else:
                last_chunk.to_parquet(parquet_path, index=False)
            print(f"📦 Parquet written → {parquet_path}")
        except Exception as e:
            print(f"ℹ️ Parquet skipped (install pyarrow to enable): {e}")

if __name__ == "__main__":
    # Example for Line 10 (matches your sample)
    extract_dm(
        source_fq="dbo.DM_Machine_Learning_Line_10",
        line=10,
        start_ts="2025-01-01 00:00:00",
        end_ts="2025-02-01 23:59:59.999",
        out_csv="DM_Machine_Learning_Line_10_extract.csv",
        chunksize=200_000,
        write_parquet=False,                 # set True if you added pyarrow
    )