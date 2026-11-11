import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing from .env")

# Connect to database
engine = create_engine(DATABASE_URL)

# Query sensor readings in chronological order
query = """
    SELECT 
        temperature_c, 
        humidity_pct, 
        pressure_hpa, 
        rain_raw, 
        received_at_utc 
    FROM sensor_readings 
    ORDER BY received_at_utc ASC;
"""

# Load into pandas DataFrame and save to CSV
df = pd.read_sql(query, con=engine)
df.to_csv("weather_logs.csv", index=False)

print(f"Successfully exported {len(df)} rows from PostgreSQL to weather_logs.csv!")