import os
from dotenv import load_dotenv
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

def load_cfg(cfg_path="config/database_config.yaml"):
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def build_engine():
    load_dotenv()  # reads .env in project root
    cfg = load_cfg()

    driver = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    server = os.getenv("MSSQL_SERVER")
    port = os.getenv("MSSQL_PORT")
    database = os.getenv("MSSQL_DATABASE")
    username = os.getenv("MSSQL_USERNAME")
    password = os.getenv("MSSQL_PASSWORD")

    if not all([server, database, username, password]):
        raise RuntimeError("Missing one or more env vars: MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USERNAME, MSSQL_PASSWORD")

    odbc = (
    f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};"
    "Encrypt=yes;TrustServerCertificate=yes;Timeout=5;Connection Timeout=5;"
)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")

def main():
    engine = build_engine()
    print("🔌 Connecting...")
    with engine.connect() as conn:
        # 1) Minimal ping
        one = conn.execute(text("SELECT 1 AS ok")).scalar()
        print(f"✅ Ping SELECT 1 -> {one}")

        # 2) Server/version info
        ver = conn.execute(text("SELECT @@SERVERNAME AS server_name, CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(50)) AS version")).mappings().first()
        print(f"🖥️  Server: {ver['server_name']} | SQL Server version: {ver['version']}")

        # 3) Current DB/user
        ctx = conn.execute(text("SELECT DB_NAME() AS db_name, SUSER_SNAME() AS login_name")).mappings().first()
        print(f"📦 Database: {ctx['db_name']} | Login: {ctx['login_name']}")

        # 4) List a few tables/views
        tables = pd.read_sql_query(
            text("""
            SELECT TOP 10 TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """),
            conn
        )
        print("📚 Sample tables/views:")
        if tables.empty:
            print("   (No tables visible; check permissions/schema.)")
        else:
            for _, r in tables.iterrows():
                print(f"   - {r['TABLE_SCHEMA']}.{r['TABLE_NAME']} ({r['TABLE_TYPE']})")

        # 5) Optional: sample from a known view/table if you have it
        # Replace dbo.SensorOEE_View with your actual object, or comment this block out.
        try:
            sample = pd.read_sql_query(
                text("SELECT TOP 10 * FROM dbo.DM_Machine_Learning_Line_10 ORDER BY 1"),
                conn
            )
            print(f"🔎 Sample rows from dbo.SensorOEE_View: {len(sample)}")
            print(sample.head(3).to_string(index=False))
        except Exception as e:
            print(f"ℹ️ Skipping view sample (not found or no permission): {e}")

    print("🎉 Connection test completed.")

if __name__ == "__main__":
    main()